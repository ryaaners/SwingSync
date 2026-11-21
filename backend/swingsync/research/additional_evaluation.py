"""Registered additional-fold outer evaluation, with no-forward artifact replay.

Only the frozen selected best_model is loaded. No chronology repair, tuning,
calibration, model promotion or product advice is performed here.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import uuid

import numpy as np
import torch

from .additional_study import admit_completed_training, recheck_study  # noqa: F401
from .cache_job import DEFAULT_LOCK_PATH, _check_previous_process, _git_base_commit, _process_start
from .evaluation import evaluate_clip, events_from_labels, summarize_evaluations
from .fold_cache_audit import _check_clip
from .fold_protocol import DEFAULT_PROTOCOL, ROOT, _same, file_sha256
from .fold_training import _runtime, _tensor_hash
from .heldout import load_selected_head, predict_probabilities

CONTEXTS = ('reference-64-reset', 'full-clip')
HEADS = ('frame', 'temporal')
CLOSED_GATES = {'normal_model_available': False, 'validated_for_serving': False,
                'pose_available': False, 'outcomes_available': False, 'advice': []}


def _read_state(raw):
    if not raw or len(raw) > 64 * 1024 * 1024:
        raise ValueError('Evaluation JSON is empty or exceeds 64 MiB')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate evaluation JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite evaluation JSON value: ' + value)
    result = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(result, dict):
        raise ValueError('Evaluation JSON must be an object')
    return result


def _now():
    return datetime.now(timezone.utc).isoformat()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _sources():
    names = ('additional_evaluation.py', 'additional_study.py', 'fold_admission.py',
             'full_fold_admission.py', 'fold_training.py', 'fold_protocol.py', 'fold_cache_audit.py',
             'heldout.py', 'features.py', 'models.py', 'evaluation.py', 'golfdb.py')
    paths = [ROOT / 'backend/swingsync/research' / name for name in names]
    paths += [ROOT / 'scripts/evaluate_additional_folds.py', ROOT / 'uv.lock']
    return {str(p.resolve()): file_sha256(p) for p in paths}


def _unchanged(paths):
    if any(Path(p).is_symlink() or file_sha256(p) != digest for p, digest in paths.items()):
        raise ValueError('An immutable input, source or published output changed')


def _stat_identity(path):
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Immutable path is not a regular file: ' + str(path))
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


class _PinnedFiles:
    """Authenticate bytes between stable stat snapshots, then reject any write.

    ctime is deliberately included: restoring bytes and mtime cannot restore
    the original filesystem change identity. Full hashes are checked again at
    the invocation boundary. This assumes the local OS filesystem is trusted.
    """

    def __init__(self, paths=None):
        self.hashes, self.identities = {}, {}
        paths = paths or {}
        before = {str(path): _stat_identity(path) for path in paths}
        _unchanged(paths)
        if any(_stat_identity(path) != identity for path, identity in before.items()):
            raise ValueError('Immutable file changed during initial byte authentication')
        self.hashes.update({str(path): digest for path, digest in paths.items()})
        self.identities.update(before)

    def add_known(self, path, digest, identity):
        # Called only after byte authentication bracketed by matching stats.
        path = str(path)
        if _stat_identity(path) != identity:
            raise ValueError('Published file changed before identity pinning')
        self.hashes[path], self.identities[path] = digest, identity

    def bind(self, path, digest):
        verified = _PinnedFiles({str(path): digest})
        self.add_known(path, digest, verified.identities[str(path)])

    def check(self):
        if any(_stat_identity(path) != identity for path, identity in self.identities.items()):
            raise ValueError('An immutable input or published output stat identity changed')

    def verify(self):
        self.check()
        _unchanged(self.hashes)
        self.check()


def _publish(path, raw, *, expected=None, pins=None):
    """Compare predecessor before publication; never clobber an unknown file."""
    path = Path(path)
    def check():
        if expected is None:
            if path.exists() or path.is_symlink():
                raise ValueError('Unknown publication predecessor')
        elif path.is_symlink() or file_sha256(path) != expected:
            raise ValueError('Publication predecessor drift')
    check()
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.part')
    try:
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        check()
        if expected is None:
            os.link(temporary, path)
        else:
            temporary.replace(path)
        temporary.unlink(missing_ok=True)  # Remove hard-link alias before pinning inode ctime.
        identity = _stat_identity(path)
        digest = hashlib.sha256(raw).hexdigest()
        if file_sha256(path) != digest or _stat_identity(path) != identity:
            raise ValueError('Published output drift')
        if pins is not None:
            pins.add_known(path, digest, identity)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _preflight(study, clips_dir):
    if set(study.folds) != {2, 3, 4} or set(study.training) != {2, 3, 4}:
        raise ValueError('All three additional folds are required')
    records = {r.clip_id: r for r in study.dataset.records}
    items, watched = [], {}
    for fold in (2, 3, 4):
        ids = study.folds[fold].manifest['partitions']['test']['clip_ids']
        if len(ids) != 350 or len(set(ids)) != 350 or set(study.training[fold]) != set(HEADS):
            raise ValueError('All six heads and exactly 350 outer clips per fold are required')
        for clip_id in sorted(ids):
            record = records[clip_id]
            row = study.audit_report['clips'][str(clip_id)]
            if record.outer_split != fold or row['official_outer_split'] != fold:
                raise ValueError('Outer split identity differs')
            cache = Path(row['cache_path'])
            source = clips_dir / f'{clip_id}.mp4'
            item = {'fold': fold, 'clip_id': clip_id, 'source_id': record.source_id,
                    'cache_path': str(cache), 'cache_sha256': row['cache_sha256'],
                    'source_path': str(source), 'source_sha256': row['source_video_sha256'],
                    'cache_key': row['cache_key'], 'native_pts_sha256': row['native_pts_sha256']}
            _load_clip(study, item)
            watched[str(cache)] = row['cache_sha256']
            watched[str(source)] = row['source_video_sha256']
            items.append(item)
    return items, watched


def _load_clip(study, item):
    record = next(r for r in study.dataset.records if r.clip_id == item['clip_id'])
    cache = Path(item['cache_path'])
    if cache.is_symlink():
        raise ValueError('Cache symlink refused')
    clip = _check_clip(record, study.dataset.sha256,
                       {'cache': cache, 'video': Path(item['source_path']),
                        'video_sha256': item['source_sha256'],
                        'entry': {'cache_sha256': item['cache_sha256'], 'cache_key': item['cache_key']}},
                       study.folds[item['fold']].feature_pipeline)
    pts = clip.pts_seconds
    if (clip.features.dtype != np.float32 or pts.dtype != np.float64
            or pts.shape != (len(clip.features),) or not np.isfinite(pts).all()
            or np.any(np.diff(pts) <= 0)
            or hashlib.sha256(pts.tobytes()).hexdigest() != item['native_pts_sha256']):
        raise ValueError('Native PTS or feature dtype differs from frozen audit')
    events_from_labels(clip.labels)
    return clip


def _array_key(head, context):
    return head + '__' + context.replace('-', '_')


def _metrics(arrays, events, pts):
    results = {}
    for head in HEADS:
        for context in CONTEXTS:
            key = _array_key(head, context)
            scores = arrays[key]
            if scores.dtype != np.float32:
                raise ValueError('Saved probabilities must retain full float32 precision')
            result = evaluate_clip(scores, events, timestamps_seconds=pts).to_dict()
            result['strictly_ordered_predictions'] = bool(np.all(np.diff(result['predicted_frames']) > 0))
            results[key] = json.loads(json.dumps(result, allow_nan=False))
    return results


def _identity(item, fingerprint):
    return {'input': item, 'evaluation_fingerprint': fingerprint,
            'timestamp_scope': 'native-reencoded-clip-PTS-no-original-source-equivalence'}


def _validate_row(root, row, item, fingerprint, clip):
    expected_path = root / f"fold{item['fold']}-clip{item['clip_id']}.npz"
    if row.get('path') != str(expected_path) or expected_path.is_symlink():
        raise ValueError('Unexpected prediction artifact path')
    _unchanged({str(expected_path): row['sha256']})
    with np.load(expected_path, allow_pickle=False) as saved:
        keys = {_array_key(h, c) for h in HEADS for c in CONTEXTS}
        if set(saved.files) != keys | {'pts_seconds', 'metadata'}:
            raise ValueError('Prediction artifact array keys differ')
        metadata = saved['metadata']
        if metadata.ndim != 0 or metadata.dtype.kind != 'U':
            raise ValueError('Prediction metadata must be a scalar JSON string')
        if not _same(json.loads(metadata.item()), _identity(item, fingerprint)):
            raise ValueError('Prediction identity differs')
        pts = saved['pts_seconds']
        if pts.dtype != np.float64 or not np.array_equal(pts, clip.pts_seconds):
            raise ValueError('Prediction artifact native PTS differs')
        metrics = _metrics(saved, events_from_labels(clip.labels), pts)
    if not _same(metrics, row.get('metrics')):
        raise ValueError('Saved metrics differ from independent probability replay')
    _unchanged({str(expected_path): row['sha256']})
    return metrics


def _summaries(rows):
    # Reconstruct official result objects from replayed dictionaries, excluding diagnostics.
    from .evaluation import ClipEvaluation
    groups = {}
    for fold in (2, 3, 4, 'additional-three-folds'):
        selected = rows if isinstance(fold, str) else [r for r in rows if r['fold'] == fold]
        groups[str(fold)] = {}
        for head in HEADS:
            for context in CONTEXTS:
                key = _array_key(head, context)
                values = [r['metrics'][key] for r in selected]
                fields = ClipEvaluation.__dataclass_fields__
                results = [ClipEvaluation(**{k: v[k] for k in fields}) for v in values]
                summary = summarize_evaluations(results)
                summary['strictly_ordered_prediction_count'] = sum(v['strictly_ordered_predictions'] for v in values)
                groups[str(fold)][key] = summary
    variation = {}
    for key in groups['2']:
        variation[key] = {}
        for metric in ('pce_percent', 'mae_frames', 'mae_ms'):
            values = [groups[str(f)][key][metric] for f in (2, 3, 4)]
            variation[key][metric] = {'values_by_fold_2_3_4': values, 'min': min(values),
                                    'max': max(values), 'population_sd_descriptive': float(np.std(values))}
    return {'groups': groups, 'descriptive_fold_variation': variation,
            'primary_context': CONTEXTS[0], 'full_clip_context_reported_separately': True,
            'scope': 'Additional three official outer folds; historical fold1 already observed. '
                     'Source-recording grouping, not verified unseen golfers or upload-domain validation.',
            **CLOSED_GATES}


def run_additional_evaluation(*, annotations, manifests_dir, audit_run, smokes_dir,
                              training_runs_dir, training_receipt, expected_receipt_sha256,
                              clips_dir, run_dir, protocol_path=DEFAULT_PROTOCOL,
                              lock_path=DEFAULT_LOCK_PATH, resume=False, max_new_clips=None):
    """Evaluate a bounded increment or authenticate a completed no-forward resume."""
    if max_new_clips is not None and (type(max_new_clips) is not int or max_new_clips < 1):
        raise ValueError('max_new_clips must be a positive integer')
    root = Path(run_dir).absolute()
    clips_dir = Path(clips_dir).absolute()
    if root.is_symlink() or clips_dir.is_symlink():
        raise ValueError('Job/source directory symlink refused')
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another job owns the shared expensive-work lock') from exc
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        study = admit_completed_training(annotations=annotations, manifests_dir=manifests_dir,
            audit_run=audit_run, smokes_dir=smokes_dir, training_runs_dir=training_runs_dir,
            training_receipt=training_receipt, expected_receipt_sha256=expected_receipt_sha256,
            protocol_path=protocol_path)
        items, watched = _preflight(study, clips_dir)
        sources, runtime = _sources(), _runtime()
        bound = dict(study.watched)
        for mapping in (watched, sources):
            for path, digest in mapping.items():
                if path in bound and bound[path] != digest:
                    raise ValueError('Conflicting immutable path bindings')
                bound[path] = digest
        input_pins = _PinnedFiles(bound)
        output_pins, state_pins = _PinnedFiles(), _PinnedFiles()
        study_identity = _bytes(study.immutable)
        def selected_identity():
            return _bytes({str(f): {h: {k: str(row[k]) for k in
                ('checkpoint_path', 'checkpoint_sha256')} for h, row in heads.items()}
                for f, heads in study.training.items()})
        selected_before = selected_identity()
        immutable = {'study': study.immutable, 'inputs': items, 'sources': sources, 'runtime': runtime,
                     'contexts': list(CONTEXTS), 'heads': list(HEADS), 'seed': 20261105,
                     'selection_state': 'best_model', 'hidden_size': 64}
        fingerprint = hashlib.sha256(_bytes(immutable)).hexdigest()
        state_path = root / 'run.json'
        previous = None
        outputs = {}
        models, model_hashes = {}, {}

        def guard():
            input_pins.check()
            output_pins.check()
            state_pins.check()
            if _bytes(study.immutable) != study_identity or selected_identity() != selected_before:
                raise ValueError('Admitted study or selected checkpoint identity changed')
            if any(_tensor_hash(model) != model_hashes[key] for key, model in models.items()):
                raise ValueError('Selected in-memory model changed')
            if not _same(runtime, _runtime()):
                raise ValueError('Evaluation runtime changed')

        def full_verify():
            guard()
            recheck_study(study)
            input_pins.verify()
            output_pins.verify()
            state_pins.verify()
            guard()

        def save():
            nonlocal previous
            guard()
            previous = _publish(state_path, _bytes(state), expected=previous, pins=state_pins)
            guard()

        if resume:
            identity = _stat_identity(state_path)
            raw = state_path.read_bytes()
            state = _read_state(raw)
            previous = hashlib.sha256(raw).hexdigest()
            if _stat_identity(state_path) != identity:
                raise ValueError('Evaluation ledger changed during reading')
            state_pins.add_known(state_path, previous, identity)
            if (not _same(state.get('immutable'), immutable) or state.get('fingerprint') != fingerprint
                    or state.get('state') not in ('running', 'partial', 'completed')
                    or any(not _same(state.get(k), v) for k, v in CLOSED_GATES.items())):
                raise ValueError('Resume identity or lifecycle differs')
            _check_previous_process(state)
            rows = state['clips']
            if len(rows) > len(items):
                raise ValueError('Unexpected progress length')
            for row, item in zip(rows, items):
                if row.get('fold') != item['fold'] or row.get('clip_id') != item['clip_id']:
                    raise ValueError('Progress is not the exact immutable prefix')
                _validate_row(root, row, item, fingerprint, _load_clip(study, item))
                outputs[row['path']] = row['sha256']
                output_pins.bind(row['path'], row['sha256'])
            pending = state.get('pending')
            in_flight = state.get('in_flight')
            if in_flight is not None:
                if len(rows) == len(items) or not _same(in_flight, _identity(items[len(rows)], fingerprint)):
                    raise ValueError('In-flight clip identity differs')
                if pending is None:
                    raise ValueError('An outer clip started without known durable probabilities; fail closed')
            elif pending is not None:
                raise ValueError('Pending probabilities lack a durable in-flight identity')
            known = {Path(p).name for p in outputs}
            if pending:
                if len(rows) == len(items):
                    raise ValueError('Unexpected pending clip after all clips')
                item = items[len(rows)]
                if (pending.get('fold'), pending.get('clip_id')) != (item['fold'], item['clip_id']):
                    raise ValueError('Pending progress differs')
                path = Path(pending['path'])
                if path.exists():
                    _validate_row(root, pending, item, fingerprint, _load_clip(study, item))
                    outputs[str(path)] = pending['sha256']
                    output_pins.bind(path, pending['sha256'])
                    known.add(path.name)
                    rows.append(pending)
                    state['pending'] = None
                    state['in_flight'] = None
                else:
                    raise ValueError('Pending artifact absent; fail closed rather than repeat a forward')
            expected_files = {'run.json'} | known
            pending_report = state.get('pending_report')
            if pending_report is not None:
                report_path = root / 'report.json'
                if len(rows) != len(items) or state.get('in_flight') is not None:
                    raise ValueError('Pending report lacks complete clip coverage')
                expected_report = _bytes(_summaries(rows))
                expected_pending = {'path': str(report_path),
                                    'sha256': hashlib.sha256(expected_report).hexdigest()}
                if not _same(pending_report, expected_pending):
                    raise ValueError('Pending report differs from independent probability replay')
                if report_path.exists():
                    output_pins.bind(report_path, pending_report['sha256'])
                    if report_path.read_bytes() != expected_report:
                        raise ValueError('Known report orphan differs from replay')
                    outputs[str(report_path)] = pending_report['sha256']
                    expected_files.add('report.json')
            if state.get('state') == 'completed':
                expected_files.add('report.json')
                if (len(rows) != len(items) or state.get('pending')
                        or state.get('in_flight') is not None or pending_report is not None):
                    raise ValueError('Completed evaluation coverage differs')
                report_path = root / 'report.json'
                outputs[str(report_path)] = state['report_sha256']
                output_pins.bind(report_path, state['report_sha256'])
                guard()
                if not _same(_read_state(report_path.read_bytes()), _summaries(rows)):
                    raise ValueError('Completed report differs from probability replay')
            if {p.name for p in root.iterdir()} != expected_files:
                raise ValueError('Unknown job artifact or orphan refused')
            guard()
            if state['state'] == 'completed':
                full_verify()
                return state
        else:
            root.mkdir(parents=True, exist_ok=False)
            state = {'schema_version': 1, 'job_id': uuid.uuid4().hex, 'created_at': _now(),
                     'git_base_commit': _git_base_commit(), 'immutable': immutable,
                     'fingerprint': fingerprint, 'clips': [], 'pending': None, 'in_flight': None,
                     'pending_report': None, 'attempts': [],
                     **CLOSED_GATES}
        attempt = {'pid': os.getpid(), 'process_start_identity': _process_start(os.getpid()),
                   'started_at': _now()}
        state['attempts'].append(attempt)
        state.update(state='running', pid=attempt['pid'],
                     process_start_identity=attempt['process_start_identity'])
        save()  # Persist identities BEFORE constructing any selected head.
        remaining = items[len(state['clips']):]
        for item in remaining[:max_new_clips]:
            guard()
            clip = _load_clip(study, item)
            state['in_flight'] = _identity(item, fingerprint)
            save()  # A crash after any forward must never silently repeat that outer clip.
            arrays = {}
            fold = item['fold']
            for head in HEADS:
                if (fold, head) not in models:
                    selected = study.training[fold][head]
                    guard()
                    models[fold, head] = load_selected_head(selected['checkpoint_path'],
                        selected['checkpoint_sha256'], head, hidden_size=64)
                    model_hashes[fold, head] = _tensor_hash(models[fold, head])
                    guard()
                for context in CONTEXTS:
                    guard()
                    arrays[_array_key(head, context)] = predict_probabilities(
                        models[fold, head], clip.features, context)
                    guard()
            metrics = _metrics(arrays, events_from_labels(clip.labels), clip.pts_seconds)
            arrays['pts_seconds'] = clip.pts_seconds
            arrays['metadata'] = np.array(json.dumps(_identity(item, fingerprint), sort_keys=True))
            buffer = io.BytesIO()
            np.savez(buffer, **arrays)
            raw = buffer.getvalue()
            path = root / f"fold{fold}-clip{item['clip_id']}.npz"
            row = {'fold': fold, 'clip_id': item['clip_id'], 'path': str(path),
                   'sha256': hashlib.sha256(raw).hexdigest(), 'metrics': metrics}
            state['pending'] = row
            save()  # Known orphan identity precedes publication.
            guard()
            _publish(path, raw, pins=output_pins)
            outputs[str(path)] = row['sha256']
            guard()
            _validate_row(root, row, item, fingerprint, clip)
            state['clips'].append(row)
            state['pending'] = None
            state['in_flight'] = None
            save()
        full_verify()  # Full SHA verification before publishing bounded completion.
        attempt['ended_at'] = _now()
        if len(state['clips']) == len(items):
            report = _summaries(state['clips'])
            guard()
            report_path = root / 'report.json'
            report_raw = _bytes(report)
            report_sha = hashlib.sha256(report_raw).hexdigest()
            state['pending_report'] = {'path': str(report_path), 'sha256': report_sha}
            save()  # Persist the exact deterministic report identity before publication.
            if str(report_path) in outputs:
                if outputs[str(report_path)] != report_sha:
                    raise ValueError('Known report orphan differs from final replay')
            else:
                _publish(report_path, report_raw, pins=output_pins)
                outputs[str(report_path)] = report_sha
            state['report_sha256'] = report_sha
            state['pending_report'] = None
            guard()
            state.update(state='completed', completed_at=_now())
        else:
            state['state'] = 'partial'
        predecessor_raw = state_path.read_bytes()
        predecessor_sha = previous
        try:
            save()
            full_verify()
        except (OSError, ValueError, RuntimeError):
            # Retract only our own unchanged final successor. Never overwrite an
            # outside ledger edit, including a same-bytes write with new ctime.
            try:
                path_key = str(state_path)
                if (state_pins.hashes.get(path_key) == previous
                        and _stat_identity(state_path) == state_pins.identities.get(path_key)):
                    _publish(state_path, predecessor_raw, expected=previous, pins=state_pins)
                    previous = predecessor_sha
            except (OSError, ValueError):
                pass
            raise
        return state
