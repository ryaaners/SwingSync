"""Synthetic software-only tensor fixtures; no GolfDB fitting/performance evidence."""
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from backend.swingsync.research import fold_training as ft  # noqa: E402
from backend.swingsync.research.experiment import HeadExperiment  # noqa: E402
from backend.swingsync.research.features import FeatureClip, _array_hash, cache_key, save_feature_cache  # noqa: E402
from backend.swingsync.research.golfdb import GolfDBAnnotations, GolfDBRecord, make_split_manifest  # noqa: E402


@pytest.fixture
def smoke(tmp_path, monkeypatch):
    records = tuple(GolfDBRecord(i, f'source-{i // 2}', 'software-fixture', 'unknown', 'iron',
                                'face-on', False, (1, 1, 2, 3, 4, 5, 6, 7, 8, 10),
                                (0., 0., 1., 1.), i // 12 + 1) for i in range(48))
    dataset = GolfDBAnnotations(records, 'a' * 64)
    manifest = make_split_manifest(dataset, 2, 20261104, .2)
    pipeline = {
        'backbone_class': 'synthetic-software-fixture', 'backbone_state_sha256': 'b' * 64,
        'backbone_provenance': {'weights': 'random-tensors-never-extracted-features'},
        'transform': {'name': 'no-image-transform-synthetic-fixture'},
    }
    paths = {}
    counts = np.zeros(9, dtype=np.int64)
    for record in dataset.records:
        if record.clip_id not in manifest['partitions']['train']['clip_ids'] + manifest['partitions']['validation']['clip_ids']:
            continue
        n = record.expected_frame_count
        identity = {
            'schema_version': 1, 'annotation_sha256': dataset.sha256,
            'clip_id': record.clip_id, 'source_id': record.source_id, 'outer_split': record.outer_split,
            'original_events': list(record.original_events), 'expected_frame_count': n,
            'annotation_event_indices': list(record.event_indices),
            'historical_source_indices': list(range(n)), 'timestamp_provenance': 'reencoded-clip-pts',
            **pipeline,
        }
        arrays = {
            'features': np.random.default_rng(record.clip_id).normal(size=(n, 1280)).astype(np.float32),
            'labels': np.r_[np.arange(8), [8] * (n - 8)].astype(np.int64),
            'pts_seconds': np.arange(n, dtype=np.float64) ** 2 / 100,
            'historical_source_indices': np.arange(n, dtype=np.int64),
        }
        meta = {'identity': identity, 'cache_key': cache_key(identity),
                'array_sha256': {name: _array_hash(value) for name, value in arrays.items()}}
        path = tmp_path / f'{record.clip_id}.npz'
        save_feature_cache(FeatureClip(**arrays, metadata=meta), path)
        paths[record.clip_id] = path
        if record.clip_id in manifest['partitions']['train']['clip_ids']:
            counts += np.bincount(arrays['labels'], minlength=9)
    admission = SimpleNamespace(dataset=dataset, manifest=manifest, cache_paths=paths,
                                feature_pipeline=pipeline, class_counts_train_only=counts.tolist(),
                                immutable={'software_fixture': True, 'source': str(tmp_path)},
                                watched={str(p): ft.file_sha256(p) for p in paths.values()})
    monkeypatch.setattr(ft, 'admit_fold', lambda **_: admission)
    def check(item):
        if any(ft.file_sha256(p) != h for p, h in item.watched.items()):
            raise ValueError('Synthetic admitted input changed')
    monkeypatch.setattr(ft, 'recheck', check)
    monkeypatch.setattr(ft, '_git_base_commit', lambda: 'synthetic-no-git-operation')
    monkeypatch.setattr(ft, '_process_start', lambda _: 'synthetic-start-identity')
    args = dict(annotations=tmp_path / 'annotations.mat', manifests_dir=tmp_path / 'manifests',
                audit_run=tmp_path / 'audit-run.json', model='frame', run_dir=tmp_path / 'run',
                lock_path=tmp_path / 'shared.lock')
    return args, admission


def _read(args):
    return json.loads((args['run_dir'] / 'run.json').read_text())


@pytest.mark.parametrize('model', ['frame', 'temporal'])
def test_smoke_is_exactly_five_train_only_updates_with_saved_provenance(smoke, monkeypatch, model):
    args, admission = smoke
    args['model'] = model
    original = ft.HeadExperiment
    def observe(config, dataset, manifest, paths):
        state = _read(args)
        assert state['state'] == 'initializing'
        assert state['immutable']['config']['seed'] == 20261105
        assert state['immutable']['runtime']['torch_num_threads'] == 1
        assert state['immutable']['implementation_sha256']
        assert state['pid'] > 0 and state['process_start_identity']
        return original(config, dataset, manifest, paths)
    monkeypatch.setattr(ft, 'HeadExperiment', observe)
    monkeypatch.setattr(HeadExperiment, '_validate_inner', lambda _: pytest.fail('No inner forward in smoke'))
    state = ft.run_fold_smoke(**args)
    assert state['state'] == 'completed' and state['smoke_completed'] is True
    assert state['summary']['optimizer_steps'] == 5
    assert state['summary']['completed_epochs'] == 0 and state['summary']['history'] == []
    assert state['summary']['best_epoch'] is None and state['summary']['best_validation_pce'] is None
    assert state['class_counts_train_only'] == admission.class_counts_train_only
    assert len(state['checkpoints']) == 5
    assert state['initial_model_tensor_sha256'] != state['final_model_tensor_sha256']
    assert all(state[k] == value for k, value in ft.CLOSED_GATES.items())
    assert state['evidence_type'] == 'five-update-training-only-smoke-not-completed-benchmark-training'


@pytest.mark.parametrize('model', ['frame', 'temporal'])
def test_partial_resume_matches_continuous_and_completed_resume_is_byte_identical(smoke, monkeypatch, model):
    args, _ = smoke
    args['model'] = model
    one_shot = ft.run_fold_smoke(**args)
    args['run_dir'] = args['run_dir'].with_name('resumed')
    first = ft.run_fold_smoke(**args, max_new_batches=2)
    assert first['state'] == 'partial' and first['summary']['optimizer_steps'] == 2
    second = ft.run_fold_smoke(**args, resume=True)
    assert second['summary'] == one_shot['summary']
    assert second['final_model_tensor_sha256'] == one_shot['final_model_tensor_sha256']
    assert len(second['attempts']) == 2 and second['summary']['optimizer_steps'] == 5
    original = (args['run_dir'] / 'run.json').read_bytes()
    monkeypatch.setattr(ft, 'HeadExperiment', lambda *_: pytest.fail('Completed resume constructed a head'))
    assert ft.run_fold_smoke(**args, resume=True) == second
    assert (args['run_dir'] / 'run.json').read_bytes() == original


def test_orphan_checkpoint_is_recovered_without_repeating_saved_update(smoke):
    args, _ = smoke
    first = ft.run_fold_smoke(**args, max_new_batches=2)
    first['checkpoints'].pop()
    (args['run_dir'] / 'run.json').write_text(json.dumps(first))
    second = ft.run_fold_smoke(**args, resume=True)
    assert second['state'] == 'completed'
    assert second['checkpoints'][1]['recovered'] is True
    assert len(second['checkpoints']) == 5


@pytest.mark.parametrize('change', ['mode', 'fold', 'model', 'budget', 'budget_bool', 'resume'])
def test_invalid_launch_is_rejected_before_directory_creation(smoke, change):
    args, _ = smoke
    args.update({'mode': 'full'} if change == 'mode' else {'fold': 3} if change == 'fold'
                else {'model': 'baseline'} if change == 'model' else {'max_new_batches': 6} if change == 'budget'
                else {'max_new_batches': True} if change == 'budget_bool' else {'resume': True})
    with pytest.raises(ValueError):
        ft.run_fold_smoke(**args)
    assert not args['run_dir'].exists()


def test_shared_lock_blocks_admission_and_duplicate_work(smoke, monkeypatch):
    args, _ = smoke
    monkeypatch.setattr(ft, 'admit_fold', lambda **_: pytest.fail('admission ran without exclusive lock'))
    with args['lock_path'].open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='shared expensive-work lock'):
            ft.run_fold_smoke(**args)
    assert not args['run_dir'].exists()


def test_relaunch_needs_explicit_resume_and_changed_runtime_fails_before_constructor(smoke, monkeypatch):
    args, _ = smoke
    ft.run_fold_smoke(**args, max_new_batches=1)
    with pytest.raises(ValueError, match='explicitly resume'):
        ft.run_fold_smoke(**args)
    monkeypatch.setattr(ft, 'HeadExperiment', lambda *_: pytest.fail('Changed runtime constructed a head'))
    original = ft._runtime
    monkeypatch.setattr(ft, '_runtime', lambda: {**original(), 'platform': 'different-machine'})
    with pytest.raises(ValueError, match='provenance'):
        ft.run_fold_smoke(**args, resume=True)


@pytest.mark.parametrize('problem', ['checkpoint', 'gates', 'source', 'counts', 'cache'])
def test_mutated_resume_dependency_fails_closed(smoke, monkeypatch, problem):
    args, admission = smoke
    first = ft.run_fold_smoke(**args, max_new_batches=1)
    if problem == 'checkpoint':
        path = Path(first['checkpoints'][0]['path'])
        path.write_bytes(path.read_bytes() + b'corrupted')
    elif problem == 'gates':
        first['validated_for_serving'] = True
        (args['run_dir'] / 'run.json').write_text(json.dumps(first))
    elif problem == 'source':
        original = ft._sources
        monkeypatch.setattr(ft, '_sources', lambda: {**original(), '/missing/source': 'a' * 64})
    elif problem == 'counts':
        admission.class_counts_train_only[0] += 1
    else:
        path = next(iter(admission.cache_paths.values()))
        path.write_bytes(path.read_bytes() + b'corrupted')
    with pytest.raises((OSError, ValueError)):
        ft.run_fold_smoke(**args, resume=True)
    assert _read(args)['state'] != 'completed'


def test_too_small_training_partition_refuses_epoch_boundary_without_construction(smoke, monkeypatch):
    args, admission = smoke
    admission.manifest = deepcopy(admission.manifest)
    admission.manifest['partitions']['train']['clip_ids'] = admission.manifest['partitions']['train']['clip_ids'][:20]
    monkeypatch.setattr(ft, 'HeadExperiment', lambda *_: pytest.fail('Tiny fixture constructed a head'))
    with pytest.raises(ValueError, match='strictly within'):
        ft.run_fold_smoke(**args)
    assert not args['run_dir'].exists()


def test_interruption_after_physical_save_recovers_one_update(smoke, monkeypatch):
    args, _ = smoke
    original = HeadExperiment.save
    def crash_once(self, path):
        original(self, path)
        raise KeyboardInterrupt('software simulated interruption after checkpoint publish')
    monkeypatch.setattr(HeadExperiment, 'save', crash_once)
    with pytest.raises(KeyboardInterrupt):
        ft.run_fold_smoke(**args)
    interrupted = _read(args)
    assert interrupted['state'] == 'interrupted' and interrupted['smoke_completed'] is False
    assert len(interrupted['checkpoints']) == 0 and len(list(args['run_dir'].glob('step-*.pt'))) == 1
    monkeypatch.setattr(HeadExperiment, 'save', original)
    final = ft.run_fold_smoke(**args, resume=True)
    assert final['state'] == 'completed' and final['checkpoints'][0]['recovered'] is True


def test_changed_input_during_final_publication_does_not_leave_completed_state(smoke, monkeypatch):
    args, admission = smoke
    original = ft._publish
    def change_on_complete(path, value):
        original(path, value)
        if value['state'] == 'completed':
            cache = next(iter(admission.cache_paths.values()))
            cache.write_bytes(cache.read_bytes() + b'drift-after-completed-publish')
    monkeypatch.setattr(ft, '_publish', change_on_complete)
    with pytest.raises(ValueError, match='input changed'):
        ft.run_fold_smoke(**args)
    failed = _read(args)
    assert failed['state'] == 'failed' and failed['smoke_completed'] is False
    assert failed['validated_for_serving'] is False


def test_completed_resume_rejects_missing_checkpoint_without_model_construction(smoke, monkeypatch):
    args, _ = smoke
    state = ft.run_fold_smoke(**args)
    Path(state['checkpoints'][-1]['path']).unlink()
    monkeypatch.setattr(ft, 'HeadExperiment', lambda *_: pytest.fail('Constructed completed model'))
    with pytest.raises(OSError):
        ft.run_fold_smoke(**args, resume=True)


def test_checkpoint_drift_after_completed_publication_is_marked_failed(smoke, monkeypatch):
    args, _ = smoke
    original = ft._publish
    def change_checkpoint(path, value):
        original(path, value)
        if value['state'] == 'completed':
            checkpoint = Path(value['checkpoints'][-1]['path'])
            checkpoint.write_bytes(checkpoint.read_bytes() + b'changed after publish')
    monkeypatch.setattr(ft, '_publish', change_checkpoint)
    with pytest.raises(ValueError, match='checkpoint differs'):
        ft.run_fold_smoke(**args)
    assert _read(args)['state'] == 'failed' and _read(args)['smoke_completed'] is False


@pytest.mark.parametrize('suffix', [b', "state": "partial"}', b', "bogus": NaN}'])
def test_resume_refuses_duplicate_or_nonfinite_ledger_before_model(smoke, monkeypatch, suffix):
    args, _ = smoke
    ft.run_fold_smoke(**args, max_new_batches=1)
    state_path = args['run_dir'] / 'run.json'
    state_path.write_bytes(state_path.read_bytes().rstrip()[:-1] + suffix)
    monkeypatch.setattr(ft, 'HeadExperiment', lambda *_: pytest.fail('Malformed ledger constructed a head'))
    with pytest.raises(ValueError, match='Duplicate|Nonfinite'):
        ft.run_fold_smoke(**args, resume=True)


def test_ledger_mutation_after_publication_is_detected(smoke, monkeypatch):
    args, _ = smoke
    original = ft._publish
    def mutate(path, value):
        original(path, value)
        if value['state'] == 'completed':
            path.write_bytes(path.read_bytes() + b' ')
    monkeypatch.setattr(ft, '_publish', mutate)
    with pytest.raises(ValueError, match='ledger changed after publication'):
        ft.run_fold_smoke(**args)
    assert _read(args)['state'] == 'failed'


def test_completed_resume_rechecks_checkpoints_after_dependency_boundary(smoke, monkeypatch):
    args, _ = smoke
    state = ft.run_fold_smoke(**args)
    original = ft._recheck
    calls = 0
    def drift_after_input_check(*values):
        nonlocal calls
        original(*values)
        calls += 1
        # Initial preflight runs before _checkpoint_rows; drift after completed
        # branch's final input/source boundary must also be caught.
        if calls == 2:
            checkpoint = Path(state['checkpoints'][-1]['path'])
            checkpoint.write_bytes(checkpoint.read_bytes() + b'late resume drift')
    monkeypatch.setattr(ft, '_recheck', drift_after_input_check)
    monkeypatch.setattr(ft, 'HeadExperiment', lambda *_: pytest.fail('Completed resume constructed a head'))
    with pytest.raises(ValueError, match='checkpoint differs'):
        ft.run_fold_smoke(**args, resume=True)


@pytest.mark.parametrize('problem', ['extra_orphan', 'wrong_orphan_step', 'summary', 'model'])
def test_resume_refuses_inconsistent_checkpoint_or_model_identity(smoke, problem):
    args, _ = smoke
    state = ft.run_fold_smoke(**args, max_new_batches=2)
    root = args['run_dir']
    if problem == 'extra_orphan':
        source = Path(state['checkpoints'][-1]['path'])
        (root / 'step-0000003.pt').write_bytes(source.read_bytes())
        (root / 'step-0000004.pt').write_bytes(source.read_bytes())
    elif problem == 'wrong_orphan_step':
        (root / 'step-0000005.pt').write_bytes(Path(state['checkpoints'][-1]['path']).read_bytes())
    elif problem == 'summary':
        state = ft.run_fold_smoke(**args, resume=True)
        state['summary']['history'] = [{'synthetic-unwanted-validation': True}]
        (root / 'run.json').write_text(json.dumps(state))
    else:
        args['model'] = 'temporal'
    with pytest.raises(ValueError):
        ft.run_fold_smoke(**args, resume=True)


@pytest.mark.parametrize('mode', ['help', 'full-refused'])
def test_direct_script_bootstraps_repository_without_pythonpath(tmp_path, mode):
    # Pytest's pythonpath setting must not mask a broken documented CLI entry.
    script = Path(__file__).resolve().parents[1] / 'scripts/train_additional_fold.py'
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [sys.executable, str(script)]
    if mode == 'help':
        command += ['--help']
    else:
        command += ['--annotations', str(tmp_path / 'missing.mat'),
                    '--manifests-dir', str(tmp_path / 'missing-manifests'),
                    '--audit-run', str(tmp_path / 'missing-audit.json'),
                    '--model', 'frame', '--run-dir', str(tmp_path / 'never-created'),
                    '--mode', 'full']
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                            text=True, timeout=60, check=False)
    assert 'ModuleNotFoundError' not in result.stderr
    if mode == 'help':
        assert result.returncode == 0 and '--max-new-batches' in result.stdout
    else:
        assert result.returncode == 2
        assert 'Full training is unavailable' in result.stderr
        assert 'No such file' not in result.stderr
    assert not (tmp_path / 'never-created').exists()
