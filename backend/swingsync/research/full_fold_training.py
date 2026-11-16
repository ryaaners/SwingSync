"""Resumable fixed-budget additional-fold head training; never outer evaluation.

Historical code/configuration and completed smoke artifacts remain immutable.
An authenticated smoke admits fresh full training, never a smoke warm start.
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

import torch

from .cache_job import DEFAULT_LOCK_PATH, _check_previous_process, _git_base_commit, _process_start
from .experiment import ExperimentConfig, HeadExperiment
from .fold_protocol import DEFAULT_PROTOCOL, ROOT, _same, file_sha256
from .fold_training import _runtime, _tensor_hash
from .full_fold_admission import admit_full_fold, recheck_full
from .models import class_weights_from_counts

DEFAULT_CHECKPOINT_BATCHES = 500
CLOSED_GATES = {'outer_test_evaluated': False, 'normal_model_available': False,
                'validated_for_serving': False, 'advice': []}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _publish(path, state):
    raw = _bytes(state)
    expected = hashlib.sha256(raw).hexdigest()
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.part')
    try:
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        if file_sha256(path) != expected:
            raise ValueError('Full-training ledger publication bytes changed')
    finally:
        temporary.unlink(missing_ok=True)


def _read_state(raw):
    if not raw or len(raw) > 8 * 1024 * 1024:
        raise ValueError('Full-training ledger is empty or exceeds eight MiB')
    def pairs(items):
        value = {}
        for key, child in items:
            if key in value:
                raise ValueError('Duplicate full-training ledger key')
            value[key] = child
        return value
    def nonfinite(_):
        raise ValueError('Nonfinite full-training ledger JSON')
    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError('Nonfinite full-training ledger numeric overflow')
        return value
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Malformed full-training ledger') from exc
    if type(value) is not dict:
        raise ValueError('Full-training ledger must be an object')
    return value


def _sources(config_path):
    paths = [Path(__file__), ROOT / 'scripts/train_full_fold.py',
             ROOT / 'backend/swingsync/research/full_fold_admission.py',
             ROOT / 'backend/swingsync/research/fold_training.py', config_path, ROOT / 'uv.lock']
    return {str(p.resolve()): file_sha256(p) for p in paths}


def _load_config(model):
    return ExperimentConfig.from_dict(json.loads((ROOT / f'experiments/{model}-v1.json').read_text()))


def _summary(summary, config, train_count, counts):
    """Validate public progress without constructing a model or forwarding inputs."""
    if type(summary) is not dict or any(type(summary.get(k)) is not int for k in (
            'optimizer_steps', 'completed_epochs', 'clips_processed_in_current_epoch')):
        raise ValueError('Full-training progress has invalid typed counts')
    epoch, cursor, steps = (summary[k] for k in (
        'completed_epochs', 'clips_processed_in_current_epoch', 'optimizer_steps'))
    batches = math.ceil(train_count / config.batch_size)
    if (not 0 <= epoch <= config.epochs or not 0 <= cursor < train_count
            or cursor % config.batch_size or (epoch == config.epochs and cursor != 0)
            or steps != epoch * batches + math.ceil(cursor / config.batch_size)
            or summary.get('state') != ('completed' if epoch == config.epochs else 'partial')
            or not _same(summary.get('class_counts_train_only'), counts)
            or summary.get('outer_test_evaluated') is not False
            or summary.get('validated_for_serving') is not False):
        raise ValueError('Full-training progress differs from fixed budget or partition')
    history = summary.get('history')
    if type(history) is not list or len(history) != epoch:
        raise ValueError('Full-training history differs from completed epochs')
    scores = []
    for number, record in enumerate(history, 1):
        if type(record) is not dict or type(record.get('validation')) is not dict:
            raise ValueError('Full-training history must contain validation objects')
        validation = record['validation']
        score = validation.get('pce')
        loss = record.get('training_loss')
        if (type(record.get('epoch')) is not int or record['epoch'] != number
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                or type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
                or validation.get('partition') != 'inner-validation-only'
                or validation.get('temporal_context') != 'full-clip'
                or validation.get('outer_test_evaluated') is not False):
            raise ValueError('Full-training history is not inner-only fixed-context selection')
        scores.append(score)
    best = max(scores) if scores else None
    best_epoch = scores.index(best) + 1 if scores else None
    if not _same(summary.get('best_epoch'), best_epoch) or not _same(summary.get('best_validation_pce'), best):
        raise ValueError('Full-training selection differs from earliest inner-PCE maximum')
    return steps


def _summary_hash(summary):
    return hashlib.sha256(_bytes(summary)).hexdigest()


def _checkpoint_rows(state, root, config, train_count, counts, interval):
    rows = state.get('checkpoints')
    if type(rows) is not list:
        raise ValueError('Invalid full-training checkpoint ledger')
    prior = 0
    for row in rows:
        if type(row) is not dict or type(row.get('optimizer_steps')) is not int:
            raise ValueError('Invalid full-training checkpoint row')
        steps = row['optimizer_steps']
        path = root / f'step-{steps:07d}.pt'
        if (not prior < steps <= min(prior + interval,
                                   (prior // math.ceil(train_count / config.batch_size) + 1)
                                   * math.ceil(train_count / config.batch_size))
                or row.get('path') != str(path)
                or path.is_symlink() or file_sha256(path) != row.get('sha256')
                or type(row.get('summary_sha256')) is not str
                or len(row['summary_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['summary_sha256'])):
            raise ValueError('Full-training checkpoint differs from durable progress or bytes')
        prior = steps
    if rows and (_summary(state.get('summary'), config, train_count, counts) != prior
                 or _summary_hash(state['summary']) != rows[-1]['summary_sha256']):
        raise ValueError('Latest full-training summary differs from checkpoint ledger')
    return prior


def _pending(state, last_steps, total, interval):
    pending = state.get('pending_advance')
    if pending is not None and (type(pending) is not dict or set(pending) != {'from_steps', 'target_steps'}
            or type(pending['from_steps']) is not int or type(pending['target_steps']) is not int
            or pending['from_steps'] != last_steps
            or not last_steps < pending['target_steps'] <= min(total, last_steps + interval)):
        raise ValueError('Pending checkpoint reservation differs from durable progress')
    return pending


def _recheck(admission, sources, runtime):
    recheck_full(admission)
    if any(file_sha256(path) != digest for path, digest in sources.items()):
        raise ValueError('Full-training source or dependency changed')
    if _runtime() != runtime:
        raise ValueError('Full-training runtime changed')


def _guard_ledger(path, guard):
    observed = file_sha256(path) if path.exists() else None
    if observed != guard['sha256']:
        raise ValueError('Full-training predecessor ledger changed; external bytes must be preserved')


def _save_state(path, state, admission, sources, runtime, config, interval, guard):
    expected = hashlib.sha256(_bytes(state)).hexdigest()
    n = len(admission.manifest['partitions']['train']['clip_ids'])
    _guard_ledger(path, guard)
    for _ in range(2):
        _recheck(admission, sources, runtime)
        last = _checkpoint_rows(state, path.parent, config, n, admission.class_counts_train_only, interval)
        _pending(state, last, math.ceil(n / config.batch_size) * config.epochs, interval)
        if _ == 0:
            _guard_ledger(path, guard)
            _publish(path, state)
            guard['sha256'] = expected
    if file_sha256(path) != expected:
        raise ValueError('Full-training ledger changed after publication')


def run_full_fold(*, annotations, manifests_dir, audit_run, smokes_dir, model, fold, run_dir,
                  protocol_path=DEFAULT_PROTOCOL, resume=False, max_new_batches=None,
                  checkpoint_batches=DEFAULT_CHECKPOINT_BATCHES, lock_path=DEFAULT_LOCK_PATH):
    """Fit one fresh full head or explicitly continue that exact saved job.

    No warm-start parameter or outer evaluation exists. The operational checkpoint
    interval is immutable; per-invocation pause limits may differ on continuation.
    """
    if type(fold) is not int or fold not in (2, 3, 4) or model not in {'frame', 'temporal'}:
        raise ValueError('Choose registered fold 2/3/4 and frame/temporal head')
    if type(resume) is not bool:
        raise ValueError('resume must be an explicit boolean')
    for value, name, maximum in ((checkpoint_batches, 'checkpoint_batches', 1000),
                                  (max_new_batches, 'max_new_batches', 1_000_000)):
        if value is None and name == 'max_new_batches':
            continue
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(name + ' must be a positive bounded integer')
    root = Path(run_dir).resolve()
    lock_path = Path(lock_path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another job owns the shared expensive-work lock') from exc
        return _run_locked(annotations, manifests_dir, audit_run, smokes_dir, model, fold, root,
                           protocol_path, resume, max_new_batches, checkpoint_batches, lock_path)


def _run_locked(annotations, manifests_dir, audit_run, smokes_dir, model, fold, root,
                protocol_path, resume, max_new_batches, interval, lock_path):
    state_path = root / 'run.json'
    existed = root.exists()
    if existed != resume:
        raise ValueError('Existing runs require explicit resume; nonexistent runs cannot resume')
    if existed and list(root.glob('run-conflict-*.json')):
        raise ValueError('A preserved external ledger conflict requires inspection before continuation')
    previous_raw = state_path.read_bytes() if existed else None
    guard = {'sha256': hashlib.sha256(previous_raw).hexdigest() if existed else None}
    state = _read_state(previous_raw) if existed else None
    if existed:
        _check_previous_process(state)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    admission = admit_full_fold(annotations=annotations, manifests_dir=manifests_dir,
                                audit_run=audit_run, smokes_dir=smokes_dir, fold=fold,
                                protocol_path=protocol_path)
    config = _load_config(model)
    config_path = ROOT / f'experiments/{model}-v1.json'
    runtime, sources = _runtime(), _sources(config_path)
    n = len(admission.manifest['partitions']['train']['clip_ids'])
    total = math.ceil(n / config.batch_size) * config.epochs
    immutable = {'admission': admission.immutable, 'config': config.to_dict(), 'fold': fold,
                 'model': model, 'mode': 'full', 'total_optimizer_batches': total,
                 'initialization': 'fresh-seeded-head-never-smoke-or-prior-fold-warm-start',
                 'runtime': runtime, 'implementation_sha256': sources,
                 'checkpoint_batches': interval, 'run_dir': str(root), 'lock_path': str(lock_path)}
    _recheck(admission, sources, runtime)
    if existed:
        if (type(state.get('schema_version')) is not int or state['schema_version'] != 1
                or not _same(state.get('immutable'), immutable)
                or any(not _same(state.get(k), v) for k, v in CLOSED_GATES.items())
                or type(state.get('attempts')) is not list):
            raise ValueError('Full-training resume provenance or closed gates changed')
        expected_initial = admission.immutable['smoke_witnesses'][model]['initial_model_tensor_sha256']
        if (not _same(state.get('class_counts_train_only'), admission.class_counts_train_only)
                or (state.get('initial_model_tensor_sha256') != expected_initial
                    and (state.get('checkpoints') or state.get('initial_model_tensor_sha256') is not None))):
            raise ValueError('Full-training saved initialization or class counts changed')
        if state.get('class_weights_train_only') is not None:
            expected_weights = class_weights_from_counts(torch.tensor(admission.class_counts_train_only)).tolist()
            if not _same(state['class_weights_train_only'], expected_weights):
                raise ValueError('Full-training saved class weights changed')
        last = _checkpoint_rows(state, root, config, n, admission.class_counts_train_only, interval)
        pending = _pending(state, last, total, interval)
        if state.get('state') == 'completed':
            if (state.get('full_training_completed') is not True or pending is not None
                    or state.get('initial_model_tensor_sha256') != expected_initial
                    or last != total or _summary(state.get('summary'), config, n,
                                                  admission.class_counts_train_only) != total
                    or {str(p) for p in root.glob('step-*.pt')} != {r['path'] for r in state['checkpoints']}):
                raise ValueError('Completed full-training ledger is inconsistent')
            _recheck(admission, sources, runtime)
            _checkpoint_rows(state, root, config, n, admission.class_counts_train_only, interval)
            if state_path.read_bytes() != previous_raw:
                raise ValueError('Completed full-training ledger changed during resume')
            return state
        if (state.get('state') not in {'partial', 'failed', 'interrupted', 'running', 'initializing'}
                or state.get('full_training_completed') is not False):
            raise ValueError('Unknown or contradictory full-training lifecycle state')
    else:
        root.mkdir(parents=True, exist_ok=False)
        state = {'schema_version': 1, 'job_id': uuid.uuid4().hex, 'created_at': _now(),
                 'evidence_type': 'fixed-additional-fold-inner-selection-not-outer-benchmark',
                 'immutable': immutable, 'git_base_commit': _git_base_commit(), 'attempts': [],
                 'checkpoints': [], 'pending_advance': None, 'full_training_completed': False,
                 'class_counts_train_only': admission.class_counts_train_only, **CLOSED_GATES}
    attempt = {'pid': os.getpid(), 'process_start_identity': _process_start(os.getpid()), 'started_at': _now()}
    state['attempts'].append(attempt)
    state.update(state='initializing', pid=attempt['pid'], process_start_identity=attempt['process_start_identity'],
                 updated_at=_now(), error=None)
    started = time.monotonic()
    try:
        # Durable job/config/runtime/source identity precedes construction of any head.
        _save_state(state_path, state, admission, sources, runtime, config, interval, guard)
        driver = HeadExperiment(config, admission.dataset, admission.manifest, admission.cache_paths)
        if (driver.provenance['class_counts_train_only'] != admission.class_counts_train_only
                or any(driver.provenance['feature_pipeline'].get(k) != v
                       for k, v in admission.feature_pipeline.items())):
            raise ValueError('Full-training driver differs from admitted pipeline or train-only counts')
        initial = _tensor_hash(driver.model)
        expected_initial = admission.immutable['smoke_witnesses'][model]['initial_model_tensor_sha256']
        if initial != expected_initial:
            raise ValueError('Full head is not the witnessed fresh seeded initialization')
        if existed and state.get('initial_model_tensor_sha256') not in {None, initial}:
            raise ValueError('Fresh full-training initialization changed')
        state.update(initial_model_tensor_sha256=initial, class_weights_train_only=driver.weights.tolist())
        if state['checkpoints']:
            driver.restore(state['checkpoints'][-1]['path'])
            if not _same(driver.summary(), state['summary']):
                raise ValueError('Restored checkpoint differs from recorded summary')
        known = {row['path'] for row in state['checkpoints']}
        orphans = [p for p in root.glob('step-*.pt') if str(p) not in known]
        if len(orphans) > 1:
            raise ValueError('Multiple unreferenced checkpoints require inspection')
        if orphans:
            pending = _pending(state, driver.optimizer_steps, total, interval)
            if pending is None or orphans[0].name != f"step-{pending['target_steps']:07d}.pt":
                raise ValueError('Unreferenced checkpoint has no matching durable advance reservation')
            driver.restore(orphans[0])
            if driver.optimizer_steps != pending['target_steps']:
                raise ValueError('Unreferenced checkpoint progress differs from reservation')
            state['checkpoints'].append({'path': str(orphans[0]), 'sha256': file_sha256(orphans[0]),
                                         'optimizer_steps': driver.optimizer_steps,
                                         'summary_sha256': _summary_hash(driver.summary()), 'recovered': True})
        state.update(pending_advance=None, summary=driver.summary(), state='running', updated_at=_now())
        _save_state(state_path, state, admission, sources, runtime, config, interval, guard)
        start_steps = driver.optimizer_steps
        while driver.optimizer_steps < total:
            remaining = None if max_new_batches is None else max_new_batches - (driver.optimizer_steps - start_steps)
            if remaining is not None and remaining <= 0:
                break
            until_epoch_end = math.ceil((n - driver.cursor) / config.batch_size)
            budget = min(interval, total - driver.optimizer_steps, until_epoch_end)
            if remaining is not None:
                budget = min(budget, remaining)
            state.update(pending_advance={'from_steps': driver.optimizer_steps,
                                          'target_steps': driver.optimizer_steps + budget}, updated_at=_now())
            _save_state(state_path, state, admission, sources, runtime, config, interval, guard)
            driver.advance(max_batches=budget)
            _summary(driver.summary(), config, n, admission.class_counts_train_only)
            checkpoint = root / f'step-{driver.optimizer_steps:07d}.pt'
            digest = driver.save(checkpoint)
            if file_sha256(checkpoint) != digest:
                raise ValueError('Full-training checkpoint publication bytes changed')
            state['checkpoints'].append({'path': str(checkpoint), 'sha256': digest,
                                         'optimizer_steps': driver.optimizer_steps,
                                         'summary_sha256': _summary_hash(driver.summary())})
            state.update(summary=driver.summary(), pending_advance=None, updated_at=_now())
            _save_state(state_path, state, admission, sources, runtime, config, interval, guard)
        complete = driver.optimizer_steps == total
        state.update(state='completed' if complete else 'partial', full_training_completed=complete,
                     summary=driver.summary(), final_model_tensor_sha256=_tensor_hash(driver.model), updated_at=_now())
        attempt.update(state=state['state'], ended_at=_now(), elapsed_seconds=time.monotonic() - started)
        _save_state(state_path, state, admission, sources, runtime, config, interval, guard)
        return state
    except BaseException as exc:
        state.update(state='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                     full_training_completed=False, error=str(exc), updated_at=_now(), **CLOSED_GATES)
        attempt.update(state=state['state'], ended_at=_now(), elapsed_seconds=time.monotonic() - started)
        # Preserve outside edits instead of overwriting an unrecognized ledger.
        observed = file_sha256(state_path) if state_path.exists() else None
        if observed != guard['sha256']:
            marker = root / ('run-conflict-' + uuid.uuid4().hex + '.json')
            _publish(marker, {'state': 'external-ledger-conflict', 'job_id': state['job_id'],
                              'error': str(exc), 'observed_ledger_sha256': observed,
                              'last_owned_ledger_sha256': guard['sha256'],
                              'full_training_completed': False, **CLOSED_GATES})
        else:
            _publish(state_path, state)
        raise
