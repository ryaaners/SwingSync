"""Resumable five-update CPU smoke for registered additional GolfDB folds.

This launcher deliberately cannot perform full training or outer evaluation.
Completion means the fixed training-only smoke budget was saved, never a trained
benchmark model. Existing historical implementations and registration stay frozen.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import uuid

import numpy as np
import torch

from .cache_job import DEFAULT_LOCK_PATH, _check_previous_process, _git_base_commit, _process_start
from .experiment import ExperimentConfig, HeadExperiment
from .fold_admission import admit_fold, recheck
from .fold_protocol import DEFAULT_PROTOCOL, ROOT, _same, file_sha256

SMOKE_STEPS = 5
SMOKE_FOLD = 2
CLOSED_GATES = {
    'full_training_completed': False,
    'outer_test_evaluated': False,
    'inner_validation_evaluated': False,
    'accuracy_metrics_computed': False,
    'normal_model_available': False,
    'validated_for_serving': False,
    'advice': [],
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _publish(path, value):
    raw = _bytes(value)
    expected = hashlib.sha256(raw).hexdigest()
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.part')
    try:
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        if file_sha256(path) != expected:
            raise ValueError('Smoke ledger publication bytes changed')
    finally:
        temporary.unlink(missing_ok=True)


def _read_state(raw):
    if not raw or len(raw) > 2 * 1024 * 1024:
        raise ValueError('Smoke ledger is empty or exceeds two MiB')
    def unique(pairs):
        value = {}
        for key, child in pairs:
            if key in value:
                raise ValueError('Duplicate smoke ledger JSON key')
            value[key] = child
        return value
    def nonfinite(_):
        raise ValueError('Nonfinite smoke ledger JSON value')
    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Malformed smoke ledger JSON') from exc
    if type(value) is not dict:
        raise ValueError('Smoke ledger must be a JSON object')
    return value


def _runtime():
    return {
        'python': platform.python_version(), 'python_executable': os.path.realpath(os.sys.executable),
        'torch': str(torch.__version__), 'torchvision': importlib.metadata.version('torchvision'),
        'numpy': np.__version__, 'pyav': importlib.metadata.version('av'),
        'machine': platform.machine(), 'platform': platform.platform(),
        'device': 'cpu', 'torch_num_threads': torch.get_num_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
    }


def _sources():
    paths = [Path(__file__), ROOT / 'scripts/train_additional_fold.py',
             ROOT / 'backend/swingsync/research/fold_admission.py',
             ROOT / 'backend/swingsync/research/fold_protocol.py', ROOT / 'uv.lock']
    return {str(p.resolve()): file_sha256(p) for p in paths}


def _recheck(admission, sources, runtime):
    recheck(admission)
    if any(file_sha256(path) != value for path, value in sources.items()):
        raise ValueError('Smoke implementation or dependency bytes changed')
    if _runtime() != runtime:
        raise ValueError('Smoke runtime changed')


def _save_state(path, state, admission, sources, runtime):
    expected = hashlib.sha256(_bytes(state)).hexdigest()
    _recheck(admission, sources, runtime)
    _checkpoint_rows(state, path.parent)
    _publish(path, state)
    _recheck(admission, sources, runtime)
    _checkpoint_rows(state, path.parent)
    if file_sha256(path) != expected:
        raise ValueError('Smoke ledger changed after publication')


def _summary_ok(summary, steps):
    return (type(summary) is dict
            and all(type(summary.get(k)) is int for k in (
                'optimizer_steps', 'completed_epochs', 'clips_processed_in_current_epoch'))
            and summary.get('optimizer_steps') == steps
            and summary.get('completed_epochs') == 0
            and summary.get('clips_processed_in_current_epoch') == steps * 4
            and summary.get('history') == [] and summary.get('best_epoch') is None
            and summary.get('best_validation_pce') is None
            and summary.get('state') == 'partial'
            and summary.get('outer_test_evaluated') is False
            and summary.get('validated_for_serving') is False)


def _checkpoint_rows(state, root):
    rows = state.get('checkpoints')
    if not isinstance(rows, list) or len(rows) > SMOKE_STEPS:
        raise ValueError('Invalid smoke checkpoint ledger')
    for step, row in enumerate(rows, 1):
        path = root / f'step-{step:07d}.pt'
        if (not isinstance(row, dict) or type(row.get('optimizer_steps')) is not int
                or row.get('optimizer_steps') != step
                or row.get('path') != str(path) or row.get('sha256') != file_sha256(path)):
            raise ValueError('Smoke checkpoint differs from durable ledger')
    return rows


def _driver_ok(driver, admission):
    if (driver.provenance['class_counts_train_only'] != admission.class_counts_train_only
            or any(driver.provenance['feature_pipeline'].get(k) != v
                   for k, v in admission.feature_pipeline.items())):
        raise ValueError('Smoke driver differs from admitted counts or frozen pipeline')


def _tensor_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_fold_smoke(*, annotations, manifests_dir, audit_run, model, run_dir,
                   fold=SMOKE_FOLD, protocol_path=DEFAULT_PROTOCOL, resume=False,
                   max_new_batches=None, lock_path=DEFAULT_LOCK_PATH, mode='smoke'):
    """Run at most five total training batches, saving every update for explicit resume.

    ``max_new_batches`` bounds only this invocation; completion always requires
    exactly five cumulative updates. No inner/outer examples are forwarded. The
    constructor authenticates all relevant train+inner arrays to count classes;
    class weights use complete training labels only. Completed resume validates
    bytes/provenance and returns without model construction or ledger mutation.
    """
    if mode != 'smoke':
        raise ValueError('Full training is unavailable until a separately reviewed full-run launcher')
    if type(fold) is not int or fold != SMOKE_FOLD:
        raise ValueError('The registered smoke uses additional fold 2 only')
    if model not in {'frame', 'temporal'}:
        raise ValueError('Choose frame or temporal')
    if type(resume) is not bool:
        raise ValueError('resume must be explicit boolean')
    if max_new_batches is not None and (type(max_new_batches) is not int
                                        or not 1 <= max_new_batches <= SMOKE_STEPS):
        raise ValueError('max_new_batches must be an integer in [1,5] or None')
    root = Path(run_dir).resolve()
    lock_path = Path(lock_path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another job owns the shared expensive-work lock') from exc
        return _run_locked(annotations, manifests_dir, audit_run, model, root, fold,
                           protocol_path, resume, max_new_batches, lock_path)


def _run_locked(annotations, manifests_dir, audit_run, model, root, fold,
                protocol_path, resume, max_new_batches, lock_path):
    state_path = root / 'run.json'
    existed = root.exists()
    if existed and not resume:
        raise ValueError('Run directory exists; inspect it and explicitly resume')
    if not existed and resume:
        raise ValueError('Run directory does not exist; cannot resume')
    previous_raw = state_path.read_bytes() if existed else None
    previous = _read_state(previous_raw) if existed else None
    if existed:
        _check_previous_process(previous)
    admission = admit_fold(annotations=annotations, manifests_dir=manifests_dir,
                           audit_run=audit_run, fold=fold, protocol_path=protocol_path)
    config_path = ROOT / f'experiments/{model}-v1.json'
    config = ExperimentConfig.from_dict(json.loads(config_path.read_text()))
    # A smoke may not cross an epoch boundary, since that invokes inner selection.
    if (config.batch_size != 4 or math.ceil(len(admission.manifest['partitions']['train']['clip_ids'])
                                          / config.batch_size) <= SMOKE_STEPS):
        raise ValueError('Five-update smoke must stay strictly within the first training epoch')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    runtime = _runtime()
    sources = _sources()
    sources[str(config_path.resolve())] = file_sha256(config_path)
    immutable = {
        'admission': admission.immutable, 'mode': 'smoke', 'fold': fold, 'model': model,
        'config': config.to_dict(), 'total_optimizer_batches': SMOKE_STEPS,
        'initialization': 'fresh-seeded-head-no-warm-start',
        'runtime': runtime, 'implementation_sha256': sources,
        'lock_path': str(lock_path), 'run_dir': str(root),
    }
    _recheck(admission, sources, runtime)
    if existed:
        state = previous
        if (type(state.get('schema_version')) is not int or state.get('schema_version') != 1
                or not _same(state.get('immutable'), immutable)
                or any(not _same(state.get(k), value) for k, value in CLOSED_GATES.items())
                or not isinstance(state.get('attempts'), list)):
            raise ValueError('Smoke resume provenance or closed gates changed')
        rows = _checkpoint_rows(state, root)
        if state.get('state') == 'completed':
            if (state.get('smoke_completed') is not True or len(rows) != SMOKE_STEPS
                    or not _summary_ok(state.get('summary'), SMOKE_STEPS)
                    or {str(p) for p in root.glob('step-*.pt')} != {r['path'] for r in rows}):
                raise ValueError('Completed smoke ledger is inconsistent')
            _recheck(admission, sources, runtime)
            _checkpoint_rows(state, root)
            if state_path.read_bytes() != previous_raw:
                raise ValueError('Completed smoke ledger changed while validating resume')
            return state
        if state.get('state') not in {'partial', 'failed', 'interrupted', 'running', 'initializing'}:
            raise ValueError('Unknown smoke lifecycle state')
        if state.get('smoke_completed') is not False:
            raise ValueError('Incomplete smoke has contradictory completion flag')
    else:
        root.mkdir(parents=True, exist_ok=False)
        state = {
            'schema_version': 1, 'job_id': uuid.uuid4().hex, 'created_at': _now(),
            'evidence_type': 'five-update-training-only-smoke-not-completed-benchmark-training',
            'immutable': immutable, 'git_base_commit': _git_base_commit(),
            'attempts': [], 'checkpoints': [], 'smoke_completed': False,
            'class_counts_train_only': admission.class_counts_train_only, **CLOSED_GATES,
        }
    attempt = {'pid': os.getpid(), 'process_start_identity': _process_start(os.getpid()),
               'started_at': _now()}
    state['attempts'].append(attempt)
    state.update(state='initializing', pid=attempt['pid'],
                 process_start_identity=attempt['process_start_identity'], error=None,
                 updated_at=_now())
    started = time.monotonic()
    # All immutable identities and process ownership persist before head construction.
    try:
        _save_state(state_path, state, admission, sources, runtime)
        driver = HeadExperiment(config, admission.dataset, admission.manifest, admission.cache_paths)
        _driver_ok(driver, admission)
        initial_hash = _tensor_hash(driver.model)
        if existed and state.get('initial_model_tensor_sha256') not in {None, initial_hash}:
            raise ValueError('Fresh smoke initialization changed')
        state['initial_model_tensor_sha256'] = initial_hash
        state['class_weights_train_only'] = driver.weights.tolist()
        if state['checkpoints']:
            driver.restore(state['checkpoints'][-1]['path'])
        if not _summary_ok(driver.summary(), len(state['checkpoints'])):
            raise ValueError('Restored smoke crossed its training-only budget')
        known = {Path(row['path']) for row in state['checkpoints']}
        orphans = [path for path in root.glob('step-*.pt') if path not in known]
        if len(orphans) > 1:
            raise ValueError('Multiple unreferenced smoke checkpoints require inspection')
        if orphans:
            next_step = len(state['checkpoints']) + 1
            if next_step > SMOKE_STEPS or orphans[0].name != f'step-{next_step:07d}.pt':
                raise ValueError('Unreferenced smoke checkpoint has invalid durable step')
            driver.restore(orphans[0])
            if not _summary_ok(driver.summary(), next_step):
                raise ValueError('Unreferenced smoke checkpoint exceeds the next saved update')
            state['checkpoints'].append({'path': str(orphans[0]), 'sha256': file_sha256(orphans[0]),
                                         'optimizer_steps': next_step, 'recovered': True})
            state.update(summary=driver.summary(), updated_at=_now())
            _save_state(state_path, state, admission, sources, runtime)
        state.update(state='running', summary=driver.summary(), updated_at=_now())
        _save_state(state_path, state, admission, sources, runtime)
        start_steps = driver.optimizer_steps
        while driver.optimizer_steps < SMOKE_STEPS:
            if max_new_batches is not None and driver.optimizer_steps - start_steps >= max_new_batches:
                break
            _recheck(admission, sources, runtime)
            driver.advance(max_batches=1)
            if not _summary_ok(driver.summary(), driver.optimizer_steps):
                raise ValueError('Smoke must not perform validation or exceed the first training epoch')
            checkpoint = root / f'step-{driver.optimizer_steps:07d}.pt'
            digest = driver.save(checkpoint)
            if file_sha256(checkpoint) != digest:
                raise ValueError('Published smoke checkpoint bytes changed')
            _recheck(admission, sources, runtime)
            state['checkpoints'].append({'path': str(checkpoint), 'sha256': digest,
                                         'optimizer_steps': driver.optimizer_steps})
            state.update(summary=driver.summary(), updated_at=_now())
            _save_state(state_path, state, admission, sources, runtime)
        complete = driver.optimizer_steps == SMOKE_STEPS
        state.update(state='completed' if complete else 'partial', smoke_completed=complete,
                     summary=driver.summary(), final_model_tensor_sha256=_tensor_hash(driver.model),
                     updated_at=_now())
        attempt.update(state=state['state'], ended_at=_now(), elapsed_seconds=time.monotonic() - started)
        _save_state(state_path, state, admission, sources, runtime)
        return state
    except BaseException as exc:
        state.update(state='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                     smoke_completed=False, error=str(exc), updated_at=_now(), **CLOSED_GATES)
        attempt.update(state=state['state'], ended_at=_now(), elapsed_seconds=time.monotonic() - started)
        # A failed dependency check must still leave an honest failure record.
        _publish(state_path, state)
        raise
