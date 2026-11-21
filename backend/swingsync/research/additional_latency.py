"""One-shot registered CPU timing for all completed additional-fold heads.

This computes no event accuracy, changes no model, and never downloads weights.
Six clip observations (four unique clips) are not a population latency guarantee.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from . import latency
from .additional_study import admit_completed_training, recheck_study
from .cache_job import DEFAULT_LOCK_PATH, _git_base_commit, _process_start
from .features import backbone_state_sha256, build_cache_identity, load_feature_cache
from .fold_protocol import DEFAULT_PROTOCOL, ROOT, _same, file_sha256
from .fold_training import _runtime, _tensor_hash
from .heldout import load_selected_head, predict_probabilities
from .timeline import inspect_clip_timeline

CONTEXTS = ('reference-64-reset', 'full-clip')
LATENCY_IDS = {2: [0, 1], 3: [6, 7], 4: [0, 1]}
WEIGHT_SHA256 = '7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f'
CLOSED_GATES = {'outer_test_evaluated': False, 'accuracy_metrics_computed': False,
                'training_performed': False, 'normal_model_available': False,
                'validated_for_serving': False, 'advice': []}


class LocalWeightsUnavailable(ValueError):
    """The exact previously acquired local reference weight is unavailable."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _serialized(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _stat_identity(value):
    # Reading bytes can change atime. These fields identify an unchanged file
    # without treating access itself as a publication change.
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _remove_unchanged_report(path, publication):
    """Remove only our unchanged publication; retain changed conflict evidence.

    Inodes may be reused after unlink, and size/mtime can be restored. Require
    both the complete recorded stat identity and the published byte digest.
    Cleanup must never obscure the original failure if another writer removes
    or changes the path while it is being inspected.
    """
    identity = publication.get('identity')
    if identity is None:
        return
    try:
        if (not path.is_symlink() and _stat_identity(path.stat()) == identity
                and file_sha256(path) == publication['sha256']
                and _stat_identity(path.stat()) == identity):
            path.unlink()
    except OSError:
        pass


def _write(path, value, *, exclusive=False, publication=None):
    raw = _serialized(value)
    expected = hashlib.sha256(raw).hexdigest()
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.part')
    publication_fd = None
    linked = False
    try:
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            if publication is not None:
                publication_fd = os.open(temporary, os.O_RDONLY)
            os.link(temporary, path)
            linked = True
        else:
            temporary.replace(path)
        if file_sha256(path) != expected:
            raise ValueError('Additional-fold timing publication bytes changed')
    finally:
        try:
            temporary.unlink(missing_ok=True)
            if publication_fd is not None and linked:
                publication.update(identity=_stat_identity(os.fstat(publication_fd)), sha256=expected)
        finally:
            if publication_fd is not None:
                os.close(publication_fd)
    return expected


def _admit(**kwargs):
    return admit_completed_training(**kwargs)


def _recheck_study(study):
    recheck_study(study)


def _sources():
    names = ('additional_latency.py', 'additional_study.py', 'full_fold_admission.py',
             'fold_training.py', 'latency.py', 'heldout.py', 'features.py', 'models.py', 'timeline.py')
    paths = [ROOT / 'backend/swingsync/research' / n for n in names]
    paths += [ROOT / 'scripts/benchmark_additional_folds.py', ROOT / 'uv.lock']
    return {str(p.resolve()): file_sha256(p) for p in paths}


def _registered(study):
    expected = {'latency_batch_size': 16, 'latency_warmup_frames': 16, 'latency_repetitions': 1,
                'latency_contexts': list(CONTEXTS), 'heads': ['frame', 'temporal'],
                'latency_partition': 'two-fixed-training-clips-per-fold'}
    evaluation = study.protocol.get('evaluation', {})
    if (not _same({k: evaluation.get(k) for k in expected}, expected)
            or set(study.folds) != {2, 3, 4} or set(study.training) != {2, 3, 4}):
        raise ValueError('Additional-fold timing differs from the fixed registered study')
    for fold in (2, 3, 4):
        ids = study.protocol['folds'][str(fold)]['latency_training_clip_ids']
        train_ids = study.folds[fold].manifest['partitions']['train']['clip_ids']
        if (not _same(ids, LATENCY_IDS[fold]) or ids != sorted(train_ids)[:2]
                or set(study.training[fold]) != {'frame', 'temporal'}):
            raise ValueError('Latency inputs must be the exact first two TRAIN clips per fold')


def _weight_path():
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'data/torch'))
    path = Path(torch.hub.get_dir()) / 'checkpoints/mobilenet_v2-7ebf99e0.pth'
    if path.is_symlink() or not path.is_file():
        raise LocalWeightsUnavailable('Required local ImageNet weight file is unavailable; no download attempted')
    if file_sha256(path) != WEIGHT_SHA256:
        raise LocalWeightsUnavailable('Local ImageNet weight bytes differ; no model constructed or download attempted')
    return path.resolve()


def _preflight(study, clips_dir):
    records = {r.clip_id: r for r in study.dataset.records}
    inputs, watched = {}, {}
    for fold in (2, 3, 4):
        admission = study.folds[fold]
        inputs[fold] = []
        for clip_id in LATENCY_IDS[fold]:
            record = records[clip_id]
            row = study.audit_report['clips'][str(clip_id)]
            source = clips_dir / f'{clip_id}.mp4'
            cache_path = Path(admission.cache_paths[clip_id]).resolve()
            if (source.is_symlink() or file_sha256(source) != row['source_video_sha256']
                    or cache_path.is_symlink() or file_sha256(cache_path) != row['cache_sha256']):
                raise ValueError('Latency source or training cache differs from authenticated audit')
            clip = load_feature_cache(cache_path)
            pts = clip.pts_seconds
            identity = clip.metadata['identity']
            required = {'clip_id': clip_id, 'source_id': record.source_id,
                        'annotation_sha256': study.dataset.sha256,
                        'expected_frame_count': record.expected_frame_count,
                        'timestamp_provenance': 'reencoded-clip-pts', **admission.feature_pipeline}
            if (clip.features.dtype != np.float32 or clip.features.shape != (record.expected_frame_count, 1280)
                    or pts.dtype != np.float64 or pts.shape != (record.expected_frame_count,)
                    or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0)
                    or hashlib.sha256(pts.tobytes()).hexdigest() != row['native_pts_sha256']
                    or any(not _same(identity.get(k), v) for k, v in required.items())):
                raise ValueError('Latency cache frame/PTS/pipeline identity differs')
            timeline = inspect_clip_timeline(source, record, origin='preprocessed')
            if (timeline.timestamp_provenance != 'reencoded-clip-pts'
                    or not np.array_equal(pts, np.asarray(timeline.pts_seconds, dtype=np.float64))):
                raise ValueError('Fresh native decoder timestamps differ from authenticated cached PTS')
            inputs[fold].append({'path': source, 'record': record, 'identity': identity,
                                 'pts': pts.copy(), 'cache_path': cache_path})
            watched[str(source.resolve())] = row['source_video_sha256']
            watched[str(cache_path)] = row['cache_sha256']
    return inputs, watched


def _check(study, sources, watched, runtime):
    _recheck_study(study)
    if any(Path(p).is_symlink() or file_sha256(p) != h for mapping in (sources, watched)
           for p, h in mapping.items()):
        raise ValueError('Additional-fold timing input or implementation changed')
    if _runtime() != runtime:
        raise ValueError('Additional-fold timing CPU runtime changed')


def _nonnegative(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def _validate_measurement(report, item):
    n, pts, record = len(item['pts']), item['pts'], item['record']
    if (type(report.get('frame_count')) is not int or report['frame_count'] != n
            or not _same(report.get('clip_id'), record.clip_id)
            or not _same(report.get('source_id'), record.source_id)
            or report.get('timestamp_provenance') != 'reencoded-clip-pts'
            or report.get('pts_sha256') != hashlib.sha256(pts.tobytes()).hexdigest()
            or report.get('first_pts_seconds') != float(pts[0])
            or report.get('last_pts_seconds') != float(pts[-1])
            or not _nonnegative(report.get('feature_extraction_wall_seconds'))
            or not _nonnegative(report.get('measured_clip_wall_seconds'))
            or set(report.get('heads', {})) != {'frame', 'temporal'}):
        raise ValueError('Timed clip report differs from authenticated TRAIN frames/PTS')
    batches = [16] * (n // 16) + ([n % 16] if n % 16 else [])
    if not _same(report.get('batch_frame_counts'), batches):
        raise ValueError('Timed backbone batches differ from registered batch16 and tail')
    for stage, count in (('decode', n), ('preprocessing', n), ('backbone', len(batches))):
        value = report.get(stage, {})
        if (type(value.get('sample_count')) is not int or value['sample_count'] != count
                or any(not _nonnegative(value.get(k)) for k in (
                    'total_seconds', 'mean_seconds', 'p50_seconds', 'p95_seconds'))):
            raise ValueError('Timed stage has invalid counts or units')
    for name in ('frame', 'temporal'):
        if set(report['heads'][name]) != set(CONTEXTS):
            raise ValueError('Timed head contexts differ from registration')
        for context in CONTEXTS:
            timing = report['heads'][name][context]
            if (not _nonnegative(timing.get('head_seconds'))
                    or timing.get('composed_video_pipeline_seconds')
                    != report['feature_extraction_wall_seconds'] + timing['head_seconds']):
                raise ValueError('Timed composed variant differs from shared feature wall plus head')
    _serialized(report)  # Every remaining numeric field must be JSON-finite too.


def run_additional_latency(*, annotations, manifests_dir, audit_run, smokes_dir, training_runs_dir,
                           training_receipt, expected_receipt_sha256, clips_dir, run_dir,
                           protocol_path=DEFAULT_PROTOCOL, lock_path=DEFAULT_LOCK_PATH,
                           clock=time.perf_counter):
    """Run registered TRAIN timing exactly once, after all six full jobs complete.

    Existing output directories are refused even if completed: this deliberately
    has no expensive automatic resume or extra repetition path.
    """
    root, clips_dir = Path(run_dir).resolve(), Path(clips_dir).resolve()
    lock_path = Path(lock_path).resolve()
    if (type(expected_receipt_sha256) is not str or len(expected_receipt_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_receipt_sha256)):
        raise ValueError('Provide the independently known completed training receipt SHA256')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another job owns the shared expensive-work lock') from exc
        if root.exists():
            raise ValueError('Latency run directory exists; inspect the one-shot evidence instead of rerunning')
        root.mkdir(parents=True, exist_ok=False)
        state_path, report_path = root / 'run.json', root / 'latency.json'
        state = {'schema_version': 1, 'job_id': uuid.uuid4().hex, 'state': 'initializing',
                 'created_at': _now(), 'pid': os.getpid(), 'process_start_identity': _process_start(os.getpid()),
                 'git_base_commit': _git_base_commit(), 'evidence_type': 'registered-training-only-CPU-latency-not-accuracy',
                 'training_receipt': str(Path(training_receipt).resolve()),
                 'expected_receipt_sha256': expected_receipt_sha256,
                 'completed_clip_observations': 0, 'report_sha256': None, **CLOSED_GATES}
        guard = {'sha256': _write(state_path, state, exclusive=True)}
        study = None
        published_report = {}
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            runtime, sources = _runtime(), _sources()
            study = _admit(annotations=annotations, manifests_dir=manifests_dir, audit_run=audit_run,
                            smokes_dir=smokes_dir, training_runs_dir=training_runs_dir,
                            training_receipt=training_receipt, expected_receipt_sha256=expected_receipt_sha256,
                            protocol_path=protocol_path)
            _registered(study)
            weight_path = _weight_path()
            inputs, watched = _preflight(study, clips_dir)
            watched[str(weight_path)] = WEIGHT_SHA256
            config = {'fold_training_clip_ids': {str(f): LATENCY_IDS[f] for f in (2, 3, 4)},
                      'batch_size': 16, 'warmup_frames_per_fold': 16, 'repetitions_per_fold_clip': 1,
                      'heads': ['frame', 'temporal'], 'contexts': list(CONTEXTS), 'device': 'cpu'}
            state.update(immutable={'study': study.immutable, 'config': config, 'runtime': runtime,
                                    'source_sha256': sources, 'input_sha256': watched},
                         hardware=latency._hardware(), devices=latency.detect_devices(), folds={})

            def save():
                expected = hashlib.sha256(_serialized(state)).hexdigest()
                for boundary in range(2):
                    _check(study, sources, watched, runtime)
                    if boundary == 0:
                        if (not state_path.is_file() or state_path.is_symlink()
                                or file_sha256(state_path) != guard['sha256']):
                            raise ValueError('Additional-fold timing predecessor ledger changed')
                        _write(state_path, state)
                        guard['sha256'] = expected
                if file_sha256(state_path) != expected:
                    raise ValueError('Additional-fold timing ledger changed after publication')

            # Full model/data/runtime provenance is durable before any model construction.
            save()
            initialized = clock()
            backbone, actual_weight_path = latency._load_local_backbone()
            if Path(actual_weight_path).resolve() != weight_path:
                raise ValueError('Backbone loader used a different local weight file')
            for fold in (2, 3, 4):
                heads = {name: load_selected_head(
                    study.training[fold][name]['checkpoint_path'],
                    study.training[fold][name]['checkpoint_sha256'], name,
                    study.training[fold][name]['config'].hidden_size) for name in ('frame', 'temporal')}
                head_hashes = {name: _tensor_hash(head) for name, head in heads.items()}
                for item in inputs[fold]:
                    actual = build_cache_identity(item['path'], item['record'], study.dataset.sha256,
                                                  backbone, device='cpu')
                    if not _same(actual, item['identity']):
                        raise ValueError('Actual timing backbone/transform/runtime differs from cached extraction')
                state.update(state='warming')
                state['folds'][str(fold)] = {'clip_ids': LATENCY_IDS[fold], 'clips': [],
                                             'selected_head_tensor_sha256': head_hashes}
                save()
                warm_started = clock()
                first = inputs[fold][0]
                features, pts, details = latency._extract_timed(first['path'], first['record'], backbone,
                                                               batch_size=16, limit=16, clock=clock)
                if (features.dtype != np.float32 or features.shape != (16, 1280)
                        or not np.isfinite(features).all() or pts.dtype != np.float64
                        or not np.array_equal(pts, first['pts'][:16]) or details.get('frame_count') != 16):
                    raise ValueError('Warmup differs from exactly sixteen authenticated TRAIN frames')
                for head in heads.values():
                    for context in CONTEXTS:
                        probabilities = predict_probabilities(head, features, context)
                        if probabilities.shape != (16, 9) or not np.isfinite(probabilities).all():
                            raise ValueError('Warmup head output is invalid')
                state['folds'][str(fold)].update(warmup_seconds=clock() - warm_started,
                                                warmup_frames=16)
                state.update(state='running')
                save()
                for item in inputs[fold]:
                    _check(study, sources, watched, runtime)
                    report = latency.measure_clip(item['path'], item['record'], backbone, heads,
                                                  expected_pts=item['pts'], batch_size=16,
                                                  contexts=CONTEXTS, clock=clock)
                    _validate_measurement(report, item)
                    if (backbone_state_sha256(backbone) != item['identity']['backbone_state_sha256']
                            or any(_tensor_hash(head) != head_hashes[name] for name, head in heads.items())):
                        raise ValueError('Frozen backbone or selected head changed during timing')
                    report['video_sha256'] = item['identity']['video_sha256']
                    state['folds'][str(fold)]['clips'].append(report)
                    state['completed_clip_observations'] += 1
                    save()
            result = {'schema_version': 1, 'job_id': state['job_id'], 'state': 'completed',
                      'immutable': state['immutable'], 'hardware': state['hardware'], 'devices': state['devices'],
                      'folds': state['folds'], 'completed_clip_observations': 6,
                      'elapsed_after_model_initialization_start_seconds': clock() - initialized,
                      'partition': 'train-only-timing', **CLOSED_GATES,
                      'timing_scope': 'CPU synchronous measurements; decode retrieval excludes RGB conversion; '
                                      'composed variant shares one extraction pass, whole-clip wall includes all4variants; '
                                      'metadata integrity checks and per-fold warmup excluded from stage timings',
                      'limitations': ['Two TRAIN clips per fold, one repetition; four unique clip IDs, not population bounds',
                                      'MPS/CUDA may be detected but only CPU measured; no minimum hardware claim',
                                      'Cached/native reencoded PTS verified, not original-source or browser time',
                                      'No event accuracy, uploaded-video reliability, causal coaching or serving admission']}
            for fold in (2, 3, 4):
                rows = result['folds'][str(fold)]['clips']
                result['folds'][str(fold)]['head_sequence_timings'] = {
                    name: {context: latency.timing_summary([r['heads'][name][context]['head_seconds'] for r in rows],
                                                            sample_unit='one whole-clip head call')
                           for context in CONTEXTS} for name in ('frame', 'temporal')}
                result['folds'][str(fold)]['feature_wall_seconds_per_frame'] = sum(
                    r['feature_extraction_wall_seconds'] for r in rows) / sum(r['frame_count'] for r in rows)
            _check(study, sources, watched, runtime)
            result_sha = hashlib.sha256(_serialized(result)).hexdigest()
            _write(report_path, result, exclusive=True, publication=published_report)
            _check(study, sources, watched, runtime)
            if file_sha256(report_path) != result_sha:
                raise ValueError('Additional-fold timing report changed after publication')
            state.update(state='completed', ended_at=_now(), report_sha256=result_sha)
            save()
            _check(study, sources, watched, runtime)
            if file_sha256(report_path) != result_sha or file_sha256(state_path) != guard['sha256']:
                raise ValueError('Completed additional-fold timing artifacts changed')
            return result
        except BaseException as exc:
            _remove_unchanged_report(report_path, published_report)
            failed = {**state, 'state': 'unavailable' if isinstance(exc, LocalWeightsUnavailable) else
                      'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                      'error': str(exc), 'ended_at': _now(), 'report_sha256': None, **CLOSED_GATES}
            if (state_path.is_file() and not state_path.is_symlink()
                    and file_sha256(state_path) == guard['sha256']):
                _write(state_path, failed)
            else:
                _write(root / ('run-conflict-' + uuid.uuid4().hex + '.json'), failed, exclusive=True)
            raise
