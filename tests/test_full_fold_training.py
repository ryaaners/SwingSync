"""Synthetic full-training wiring only; never GolfDB model performance evidence."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from test_fold_training import smoke as smoke_fixture  # noqa: E402
from backend.swingsync.research import full_fold_training as ft  # noqa: E402
from backend.swingsync.research.experiment import ExperimentConfig, HeadExperiment  # noqa: E402
from backend.swingsync.research.models import FrameClassifier, TemporalClassifier  # noqa: E402
from backend.swingsync.research.training import seed_everything  # noqa: E402

smoke = smoke_fixture


@pytest.fixture
def full(smoke, monkeypatch):
    options, admission = smoke
    options = dict(options, fold=2, smokes_dir=options['run_dir'].parent / 'synthetic-smokes')
    def config(model):
        return ExperimentConfig(model=model, epochs=2, hidden_size=4)
    initial = {}
    for model in ('frame', 'temporal'):
        seed_everything(config(model).seed)
        head = FrameClassifier() if model == 'frame' else TemporalClassifier(hidden_size=4)
        initial[model] = {'initial_model_tensor_sha256': ft._tensor_hash(head)}
    admission.immutable['smoke_witnesses'] = initial
    monkeypatch.setattr(ft, 'admit_full_fold', lambda **_: admission)
    def recheck(item):
        if any(ft.file_sha256(p) != h for p, h in item.watched.items()):
            raise ValueError('Synthetic admitted cache changed')
    monkeypatch.setattr(ft, 'recheck_full', recheck)
    monkeypatch.setattr(ft, '_load_config', config)
    monkeypatch.setattr(ft, '_git_base_commit', lambda: 'synthetic-no-git')
    monkeypatch.setattr(ft, '_process_start', lambda _: 'synthetic-process')
    return options, admission


def read(options):
    return json.loads((options['run_dir'] / 'run.json').read_text())


@pytest.mark.parametrize('model', ['frame', 'temporal'])
def test_complete_inner_only_training_fresh_init_and_epoch_checkpoints(full, monkeypatch, model):
    options, admission = full
    options['model'] = model
    constructor = ft.HeadExperiment
    def observe(config, dataset, manifest, paths):
        state = read(options)
        assert state['state'] == 'initializing'
        assert state['pid'] > 0 and state['process_start_identity']
        assert state['immutable']['runtime']['torch_num_threads'] == 1
        assert state['immutable']['implementation_sha256']
        assert not set(paths) & set(manifest['partitions']['test']['clip_ids'])
        return constructor(config, dataset, manifest, paths)
    monkeypatch.setattr(ft, 'HeadExperiment', observe)
    state = ft.run_full_fold(**options)
    n = len(admission.manifest['partitions']['train']['clip_ids'])
    batches = math.ceil(n / 4)
    assert state['state'] == 'completed' and state['full_training_completed'] is True
    assert state['summary']['completed_epochs'] == 2
    assert state['summary']['optimizer_steps'] == 2 * batches
    assert [r['optimizer_steps'] for r in state['checkpoints']] == [batches, 2 * batches]
    assert state['initial_model_tensor_sha256'] == admission.immutable['smoke_witnesses'][model]['initial_model_tensor_sha256']
    assert state['final_model_tensor_sha256'] != state['initial_model_tensor_sha256']
    assert state['summary']['best_epoch'] in (1, 2)
    assert len(state['summary']['history']) == 2
    assert all(h['validation']['partition'] == 'inner-validation-only' for h in state['summary']['history'])
    assert all(state[k] == v for k, v in ft.CLOSED_GATES.items())
    assert state['pending_advance'] is None
    final = torch.load(state['checkpoints'][-1]['path'], map_location='cpu', weights_only=True)
    selected_step = state['summary']['best_epoch'] * batches
    selected = next(r for r in state['checkpoints'] if r['optimizer_steps'] == selected_step)
    selected_payload = torch.load(selected['path'], map_location='cpu', weights_only=True)
    assert all(torch.equal(v, selected_payload['model'][k]) for k, v in final['best_model'].items())


@pytest.mark.parametrize('model', ['frame', 'temporal'])
def test_partial_resume_matches_continuation_and_completed_noop(full, monkeypatch, model):
    options, _ = full
    options['model'] = model
    whole = ft.run_full_fold(**options)
    options['run_dir'] = options['run_dir'].with_name('resumed')
    first = ft.run_full_fold(**options, max_new_batches=3)
    assert first['state'] == 'partial' and first['summary']['optimizer_steps'] == 3
    second = ft.run_full_fold(**options, resume=True)
    assert second['summary'] == whole['summary']
    assert second['final_model_tensor_sha256'] == whole['final_model_tensor_sha256']
    assert len(second['attempts']) == 2
    raw = (options['run_dir'] / 'run.json').read_bytes()
    monkeypatch.setattr(ft, 'HeadExperiment', lambda *_: pytest.fail('Completed resume constructed a model'))
    assert ft.run_full_fold(**options, resume=True) == second
    assert (options['run_dir'] / 'run.json').read_bytes() == raw


def test_fixed_twenty_epochs_exact_ties_select_earliest_state(full, monkeypatch):
    options, _ = full
    monkeypatch.setattr(ft, '_load_config', lambda model: ExperimentConfig(model=model, epochs=20, hidden_size=4))
    original = HeadExperiment._validate_inner
    def ties(driver):
        result = original(driver)
        result['pce'] = 0.25
        return result
    monkeypatch.setattr(HeadExperiment, '_validate_inner', ties)
    state = ft.run_full_fold(**options)
    assert state['summary']['completed_epochs'] == 20 and state['summary']['best_epoch'] == 1
    final = torch.load(state['checkpoints'][-1]['path'], weights_only=True)
    first = torch.load(state['checkpoints'][0]['path'], weights_only=True)
    assert all(torch.equal(v, first['model'][k]) for k, v in final['best_model'].items())
    assert any(not torch.equal(v, final['model'][k]) for k, v in final['best_model'].items())


def test_interrupted_checkpoint_recovers_exact_reserved_progress(full, monkeypatch):
    options, _ = full
    save = HeadExperiment.save
    def crash_after_publish(driver, path):
        save(driver, path)
        raise KeyboardInterrupt('synthetic publication gap')
    monkeypatch.setattr(HeadExperiment, 'save', crash_after_publish)
    with pytest.raises(KeyboardInterrupt):
        ft.run_full_fold(**options, max_new_batches=3)
    partial = read(options)
    assert partial['state'] == 'interrupted' and partial['checkpoints'] == []
    assert partial['pending_advance'] == {'from_steps': 0, 'target_steps': 3}
    monkeypatch.setattr(HeadExperiment, 'save', save)
    final = ft.run_full_fold(**options, resume=True)
    assert final['state'] == 'completed' and final['checkpoints'][0]['recovered'] is True
    assert final['checkpoints'][0]['optimizer_steps'] == 3


@pytest.mark.parametrize('problem', ['multiple', 'wrong_step', 'no_reservation'])
def test_ambiguous_or_unreserved_orphan_refused(full, problem):
    options, _ = full
    state = ft.run_full_fold(**options, max_new_batches=3)
    root = options['run_dir']
    source = Path(state['checkpoints'][-1]['path']).read_bytes()
    (root / 'step-0000004.pt').write_bytes(source)
    if problem == 'multiple':
        (root / 'step-0000005.pt').write_bytes(source)
    elif problem == 'wrong_step':
        state['pending_advance'] = {'from_steps': 3, 'target_steps': 5}
        (root / 'run.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='unreferenced|Unreferenced'):
        ft.run_full_fold(**options, resume=True)


@pytest.mark.parametrize('change', ['fold', 'model', 'resume', 'budget', 'interval', 'warm_start'])
def test_invalid_launch_before_directory(full, change):
    options, _ = full
    options.update({'fold': 1} if change == 'fold' else {'model': 'unknown'} if change == 'model'
                   else {'resume': True} if change == 'resume' else {'max_new_batches': True} if change == 'budget'
                   else {'checkpoint_batches': 0} if change == 'interval' else {'warm_start': 'smoke.pt'})
    with pytest.raises((ValueError, TypeError)):
        ft.run_full_fold(**options)
    assert not options['run_dir'].exists()


def test_shared_lock_prevents_admission(full, monkeypatch):
    options, _ = full
    monkeypatch.setattr(ft, 'admit_full_fold', lambda **_: pytest.fail('Admitted without lock'))
    with options['lock_path'].open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='shared expensive-work lock'):
            ft.run_full_fold(**options)


@pytest.mark.parametrize('problem', ['checkpoint', 'interval', 'source', 'counts', 'runtime', 'initial', 'weights'])
def test_resume_refuses_changed_provenance_without_further_updates(full, monkeypatch, problem):
    options, admission = full
    state = ft.run_full_fold(**options, max_new_batches=3)
    if problem == 'checkpoint':
        path = Path(state['checkpoints'][-1]['path'])
        path.write_bytes(path.read_bytes() + b'drift')
    elif problem == 'interval':
        options['checkpoint_batches'] = 499
    elif problem == 'source':
        old = ft._sources
        monkeypatch.setattr(ft, '_sources', lambda p: {**old(p), '/missing/source.py': '0' * 64})
    elif problem == 'counts':
        state['class_counts_train_only'][0] += 1
    elif problem == 'initial':
        state['initial_model_tensor_sha256'] = '0' * 64
    elif problem == 'weights':
        state['class_weights_train_only'][0] += 1
    else:
        old = ft._runtime
        monkeypatch.setattr(ft, '_runtime', lambda: {**old(), 'platform': 'changed'})
    if problem in {'counts', 'initial', 'weights'}:
        (options['run_dir'] / 'run.json').write_text(json.dumps(state))
    with pytest.raises((ValueError, OSError)):
        ft.run_full_fold(**options, resume=True)


@pytest.mark.parametrize('raw', [b'{"x":NaN}', b'{"x":1e999}', b'{"x":1,"x":2}', b'[]'])
def test_strict_ledger_rejects_nonfinite_overflow_duplicates_and_nonobject(raw):
    with pytest.raises(ValueError):
        ft._read_state(raw)


def test_predecessor_ledger_drift_is_preserved_and_blocks_continuation(full, monkeypatch):
    options, _ = full
    original = HeadExperiment.advance
    external = b'{"external-edit":"preserve exactly"}'
    def modify_then_advance(driver, **kwargs):
        result = original(driver, **kwargs)
        (options['run_dir'] / 'run.json').write_bytes(external)
        return result
    monkeypatch.setattr(HeadExperiment, 'advance', modify_then_advance)
    with pytest.raises(ValueError, match='predecessor ledger changed'):
        ft.run_full_fold(**options, max_new_batches=3)
    assert (options['run_dir'] / 'run.json').read_bytes() == external
    markers = list(options['run_dir'].glob('run-conflict-*.json'))
    assert len(markers) == 1 and json.loads(markers[0].read_text())['full_training_completed'] is False
    with pytest.raises(ValueError, match='requires inspection'):
        ft.run_full_fold(**options, resume=True)


def test_completed_resume_rechecks_checkpoint_after_final_dependency_boundary(full, monkeypatch):
    options, _ = full
    state = ft.run_full_fold(**options)
    recheck = ft._recheck
    calls = 0
    def drift(*args):
        nonlocal calls
        recheck(*args)
        calls += 1
        if calls == 2:
            path = Path(state['checkpoints'][-1]['path'])
            path.write_bytes(path.read_bytes() + b'late drift')
    monkeypatch.setattr(ft, '_recheck', drift)
    monkeypatch.setattr(ft, 'HeadExperiment', lambda *_: pytest.fail('Completed model constructed'))
    with pytest.raises(ValueError, match='checkpoint differs'):
        ft.run_full_fold(**options, resume=True)


@pytest.mark.parametrize('target', ['checkpoint', 'ledger', 'input'])
def test_postpublication_drift_fails_closed(full, monkeypatch, target):
    options, admission = full
    publish = ft._publish
    def mutate(path, value):
        publish(path, value)
        if value.get('state') == 'completed':
            if target == 'ledger':
                path.write_bytes(path.read_bytes() + b' ')
            elif target == 'checkpoint':
                cp = Path(value['checkpoints'][-1]['path'])
                cp.write_bytes(cp.read_bytes() + b' ')
            else:
                cp = next(iter(admission.cache_paths.values()))
                cp.write_bytes(cp.read_bytes() + b' ')
    monkeypatch.setattr(ft, '_publish', mutate)
    with pytest.raises(ValueError):
        ft.run_full_fold(**options)
    if target == 'ledger':
        assert list(options['run_dir'].glob('run-conflict-*.json'))
    else:
        assert read(options)['state'] == 'failed'


@pytest.mark.parametrize('mode', ['help', 'invalid-fold'])
def test_direct_entry_without_pythonpath_from_other_directory(tmp_path, mode):
    script = Path(__file__).resolve().parents[1] / 'scripts/train_full_fold.py'
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    args = ['--help'] if mode == 'help' else ['--annotations', 'missing', '--manifests-dir', 'missing',
            '--audit-run', 'missing', '--smokes-dir', 'missing', '--model', 'frame', '--fold', '1', '--run-dir', 'never']
    result = subprocess.run([sys.executable, str(script), *args], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == (0 if mode == 'help' else 2)
    assert 'ModuleNotFoundError' not in result.stderr
    assert not (tmp_path / 'never').exists()


@pytest.mark.parametrize('field', ['initial_model_tensor_sha256', 'class_counts_train_only', 'class_weights_train_only'])
def test_completed_resume_refuses_changed_root_initialization_or_counts(full, monkeypatch, field):
    options, _ = full
    state = ft.run_full_fold(**options)
    if field == 'initial_model_tensor_sha256':
        state[field] = '0' * 64
    else:
        state[field][0] += 1
    (options['run_dir'] / 'run.json').write_text(json.dumps(state))
    monkeypatch.setattr(ft, 'HeadExperiment', lambda *_: pytest.fail('Changed completed state constructed a head'))
    with pytest.raises(ValueError, match='initialization|class counts|class weights'):
        ft.run_full_fold(**options, resume=True)


def test_wrong_fresh_initialization_refused_before_any_update(full, monkeypatch):
    options, admission = full
    admission.immutable['smoke_witnesses']['frame']['initial_model_tensor_sha256'] = '0' * 64
    monkeypatch.setattr(HeadExperiment, 'advance', lambda *_a, **_k: pytest.fail('Wrong initialization updated'))
    with pytest.raises(ValueError, match='witnessed fresh seeded'):
        ft.run_full_fold(**options)
    assert read(options)['state'] == 'failed' and read(options)['checkpoints'] == []
