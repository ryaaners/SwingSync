"""Resumable local dataset feature preparation; never training or evaluation.

Selected media and immutable configuration are fingerprinted before extraction.
An unfinished job requires explicit resume; completed caches are revalidated,
not trusted merely because a prior process wrote a completed flag.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import inspect
import marshal
import os
from pathlib import Path
import subprocess
import time
import uuid

import torch

from .features import (
    build_cache_identity, cache_key, extract_clip_features, load_feature_cache, save_feature_cache,
)
from .golfdb import load_annotations, make_split_manifest
from .models import FrozenMobileNetV2, select_device

DEFAULT_LOCK_PATH = Path(__file__).resolve().parents[3] / 'artifacts/training/active-training.lock'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _publish_json(path, value):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.part')
    try:
        with temporary.open('x') as output:
            json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _process_start(pid):
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart='],
                                capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_base_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=Path(__file__).resolve().parents[3], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _check_previous_process(state):
    """The lock is authoritative; also avoid a still-live recorded owner if known."""
    if state.get('state') not in {'running', 'initializing'}:
        return
    pid = state.get('pid')
    if not isinstance(pid, int) or pid < 1:
        raise ValueError('Running job has invalid process identity; inspect it before resuming')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise ValueError('Cannot verify previous owner is stopped; inspect the recorded process') from exc
    old_start, current_start = state.get('process_start_identity'), _process_start(pid)
    if old_start is not None and current_start is not None and old_start != current_start:
        return  # PID was reused; the recorded owner ended.
    raise ValueError('Recorded job owner is still alive or unverifiable; do not duplicate it')


def _validate_partitions(partitions, allow_test):
    partitions = tuple(partitions)
    if (not partitions or len(set(partitions)) != len(partitions)
            or any(p not in {'train', 'validation', 'test'} for p in partitions)):
        raise ValueError('Choose unique nonempty train, validation or test partitions')
    if 'test' in partitions and not allow_test:
        raise ValueError('Outer test feature preparation requires explicit allow_test=True')
    return tuple(p for p in ('train', 'validation', 'test') if p in partitions)


def run_cache_job(annotations, manifest, clips_dir, job_dir, cache_dir=Path('artifacts/features'), *,
                  partitions=('train', 'validation'), allow_test=False, device='cpu', batch_size=16,
                  max_new_clips=None, resume=False, lock_path=DEFAULT_LOCK_PATH,
                  backbone=None, extractor=extract_clip_features):
    """Prepare selected full clips with a durable per-clip ledger and explicit resume.

    max_new_clips bounds fresh extraction, not valid existing-cache verification.
    A shared cache published before a crash is recovered without re-extraction.
    All prior completed entries are checked against actual bytes on every resume.
    Passing backbone/extractor supports offline software fixtures; defaults use
    explicit reference ImageNet V2 weights and the existing frozen extractor.
    """
    partitions = _validate_partitions(partitions, allow_test)
    if max_new_clips is not None and (
        isinstance(max_new_clips, bool) or not isinstance(max_new_clips, int) or max_new_clips < 0
    ):
        raise ValueError('max_new_clips must be a nonnegative integer or None')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 256:
        raise ValueError('batch_size must be between 1 and 256')
    if device not in {'cpu', 'mps', 'cuda'}:
        raise ValueError('Choose an explicit cpu, mps or cuda extraction device')
    clips_dir, job_dir, cache_dir = (Path(p).resolve() for p in (clips_dir, job_dir, cache_dir))
    annotations, manifest, lock_path = Path(annotations), Path(manifest), Path(lock_path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another training or feature job owns the shared expensive-work lock') from exc
        return _run_locked(annotations, manifest, clips_dir, job_dir, cache_dir, partitions, allow_test,
                           device, batch_size, max_new_clips, resume, lock_path, backbone, extractor)


def _run_locked(annotations, manifest_path, clips_dir, job_dir, cache_dir, partitions, allow_test,
                device, batch_size, max_new_clips, resume, lock_path, backbone, extractor):
    state_path = job_dir / 'job.json'
    existing = None
    if job_dir.exists():
        if not resume:
            raise ValueError('Job directory exists; inspect its state and explicitly resume it')
        try:
            existing = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError('Existing job has missing or corrupt durable state') from exc
        if not isinstance(existing, dict) or existing.get('schema_version') != 1:
            raise ValueError('Existing job has invalid state schema')
        if existing.get('state') not in {'initializing', 'running', 'partial', 'completed', 'failed', 'interrupted'}:
            raise ValueError('Existing job contains an unknown durable state')
        _check_previous_process(existing)
    elif resume:
        raise ValueError('Cannot resume a job directory that does not exist')

    dataset = load_annotations(annotations)
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        expected = make_split_manifest(dataset, manifest['outer_fold'], manifest['seed'],
                                       manifest['validation_fraction_sources'])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Malformed split manifest') from exc
    if manifest != expected:
        raise ValueError('Split manifest does not match audited source-grouped partitions')
    records = {r.clip_id: r for r in dataset.records}
    clip_ids = sorted(clip_id for part in partitions for clip_id in manifest['partitions'][part]['clip_ids'])
    if not clip_ids:
        raise ValueError('Requested partitions contain no clips')
    membership = {clip_id: part for part in partitions for clip_id in manifest['partitions'][part]['clip_ids']}
    inventory = [{'clip_id': i, 'source_id': records[i].source_id, 'partition': membership[i],
                  'video_sha256': _hash(clips_dir / f'{i}.mp4')} for i in clip_ids]
    config = {
        'annotations_sha256': dataset.sha256, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'partitions': list(partitions), 'allow_test': bool(allow_test), 'device': device,
        'batch_size': batch_size, 'torch_threads': 1, 'deterministic_algorithms': True,
        'clips_dir': str(clips_dir), 'cache_dir': str(cache_dir), 'lock_path': str(lock_path),
        'inventory': inventory, 'weights_request': 'IMAGENET1K_V2' if backbone is None else 'injected-backbone',
        'job_implementation_sha256': _hash(__file__),
        'extractor_implementation_sha256': hashlib.sha256(marshal.dumps(extractor.__code__)).hexdigest(),
        'extractor_module_sha256': _hash(inspect.getfile(extractor)),
        'dependency_lock_sha256': _hash(Path(__file__).resolve().parents[3] / 'uv.lock'),
    }
    # __file__ is backend/swingsync/research/cache_job.py; repository root is parents[3].
    config_sha = _json_hash(config)
    if existing is not None:
        if (existing.get('config_sha256') != config_sha or existing.get('config') != config
                or not isinstance(existing.get('attempts'), list) or not isinstance(existing.get('completed'), dict)):
            raise ValueError('Resume configuration, media inventory, implementation or dependency provenance changed')
    if existing is None:
        job_dir.mkdir(parents=True, exist_ok=False)
        state = {'schema_version': 1, 'job_id': uuid.uuid4().hex, 'created_at': _now(),
                 'config': config, 'config_sha256': config_sha, 'state': 'initializing',
                 'git_base_commit': _git_base_commit(),
                 'completed': {}, 'completed_clip_count': 0, 'target_clip_count': len(clip_ids),
                 'attempts': [], 'evidence_type': 'frozen-feature-preparation-not-model-training',
                 'outer_test_selected': 'test' in partitions, 'validated_for_serving': False}
    else:
        state = existing
    attempt = {'attempt': len(state['attempts']) + 1, 'pid': os.getpid(), 'started_at': _now(),
               'process_start_identity': _process_start(os.getpid()), 'max_new_clips': max_new_clips,
               'newly_extracted': 0, 'reused_cache': 0, 'state': 'initializing'}
    state['attempts'].append(attempt)
    state.update(state='initializing', pid=attempt['pid'], process_start_identity=attempt['process_start_identity'],
                 updated_at=_now(), error=None)
    _publish_json(state_path, state)  # Identity/config persist before any weight acquisition/extraction.
    started = time.monotonic()
    try:
        select_device(device)  # Explicit errors, never a silent backend fallback.
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault('TORCH_HOME', str(Path('data/torch').resolve()))
        backbone = FrozenMobileNetV2(weights='IMAGENET1K_V2') if backbone is None else backbone
        inventory_hashes = {entry['clip_id']: entry['video_sha256'] for entry in inventory}

        def checked_identity(clip_id):
            identity = build_cache_identity(clips_dir / f'{clip_id}.mp4', records[clip_id],
                                            dataset.sha256, backbone, device=device)
            if identity['video_sha256'] != inventory_hashes[clip_id]:
                raise ValueError(f'Queued media changed after inventory fingerprint for clip {clip_id}')
            return identity

        first_identity = checked_identity(clip_ids[0])
        model_identity = {k: first_identity[k] for k in (
            'backbone_class', 'backbone_state_sha256', 'backbone_provenance', 'transform', 'numpy', 'torch',
            'torchvision', 'pyav', 'ffmpeg_libraries', 'extraction_device', 'implementation_sha256', 'python')}
        if 'model_identity' in state and state['model_identity'] != model_identity:
            raise ValueError('Resume backbone, transform, dependency or pipeline implementation changed')
        fingerprint = _json_hash({'config_sha256': config_sha, 'model_identity': model_identity})
        if 'job_fingerprint' in state and state['job_fingerprint'] != fingerprint:
            raise ValueError('Stored job fingerprint is inconsistent with validated configuration')
        state['model_identity'] = model_identity
        state['job_fingerprint'] = fingerprint
        if (not isinstance(state.get('completed'), dict)
                or not set(state['completed']) <= {str(i) for i in clip_ids}
                or any(not isinstance(entry, dict) for entry in state['completed'].values())):
            raise ValueError('Completed ledger contains unexpected clip IDs')
        cache_dir.mkdir(parents=True, exist_ok=True)
        state.update(state='running', updated_at=_now())
        attempt['state'] = 'running'
        _publish_json(state_path, state)
        # Resume must verify every prior entry, including entries after a bounded-work boundary.
        for clip_id in clip_ids:
            saved = state['completed'].get(str(clip_id))
            if saved is not None:
                identity = checked_identity(clip_id)
                cache_path = cache_dir / f'{cache_key(identity)}.npz'
                clip = load_feature_cache(cache_path, expected_identity=identity)
                expected_entry = {'clip_id': clip_id, 'source_id': records[clip_id].source_id,
                                  'partition': membership[clip_id], 'cache_path': str(cache_path),
                                  'frames': len(clip.labels)}
                if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in expected_entry.items()):
                    raise ValueError(f'Completed cache ledger semantics changed for clip {clip_id}')
                if saved.get('cache_key') != cache_key(identity) or saved.get('cache_sha256') != _hash(cache_path):
                    raise ValueError(f'Completed cache checksum or provenance changed for clip {clip_id}')
        for clip_id in clip_ids:
            if str(clip_id) in state['completed']:
                continue
            identity = checked_identity(clip_id)
            cache_path = cache_dir / f'{cache_key(identity)}.npz'
            reused = cache_path.exists()
            if reused:
                clip = load_feature_cache(cache_path, expected_identity=identity)
            else:
                if max_new_clips is not None and attempt['newly_extracted'] >= max_new_clips:
                    break
                clip = extractor(clips_dir / f'{clip_id}.mp4', records[clip_id], annotation_sha256=dataset.sha256,
                                 backbone=backbone, batch_size=batch_size, device=device)
                if clip.metadata.get('identity') != identity:
                    raise ValueError('Extractor produced cache with unexpected provenance')
                save_feature_cache(clip, cache_path)
                attempt['newly_extracted'] += 1
            state['completed'][str(clip_id)] = {
                'clip_id': clip_id, 'source_id': records[clip_id].source_id, 'partition': membership[clip_id],
                'cache_key': cache_key(identity), 'cache_path': str(cache_path), 'cache_sha256': _hash(cache_path),
                'frames': len(clip.labels), 'completed_at': _now(), 'reused': reused,
            }
            attempt['reused_cache'] += int(reused)
            state.update(completed_clip_count=len(state['completed']), updated_at=_now())
            _publish_json(state_path, state)
        final = 'completed' if len(state['completed']) == len(clip_ids) else 'partial'
        state.update(state=final, completed_clip_count=len(state['completed']), target_clip_count=len(clip_ids),
                     updated_at=_now())
        attempt.update(state=final, ended_at=_now(), elapsed_seconds=time.monotonic() - started)
        _publish_json(state_path, state)
        return state
    except BaseException as exc:
        final = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        attempt.update(state=final, ended_at=_now(), elapsed_seconds=time.monotonic() - started, error=str(exc))
        state.update(state=final, updated_at=_now(), error=str(exc))
        _publish_json(state_path, state)
        raise
