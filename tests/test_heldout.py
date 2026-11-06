"""Synthetic tensors/checkpoints test evaluation plumbing, never ML performance.

Pipeline metadata is an explicit mocked fixture contract. No ImageNet extraction,
real media decode or actual outer-test model result is produced by these tests.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from test_experiment import inputs as experiment_inputs  # noqa: E402,F401
from backend.swingsync.research import heldout  # noqa: E402
from backend.swingsync.research.experiment import ExperimentConfig, HeadExperiment  # noqa: E402
from backend.swingsync.research.features import (  # noqa: E402
    FeatureClip, _array_hash, cache_key, load_feature_cache, save_feature_cache,
)
from backend.swingsync.research.models import FrameClassifier, TemporalClassifier  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


@pytest.fixture
def admitted(tmp_path, monkeypatch, experiment_inputs):  # noqa: F811
    dataset, manifest, paths = experiment_inputs
    monkeypatch.setattr(heldout, 'load_annotations', lambda _: dataset)
    monkeypatch.setattr(heldout, '_git_base_commit', lambda: 'synthetic-software-revision')
    monkeypatch.setattr(heldout, 'DEFAULT_LOCK_PATH', tmp_path / 'shared.lock')
    pipeline = deepcopy(json.loads(Path('experiments/event-v1.json').read_text())['feature_pipeline'])
    for path in paths.values():
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        meta = json.loads(arrays['metadata'].item())
        meta['identity'].update(pipeline)
        meta['cache_key'] = cache_key(meta['identity'])
        arrays['metadata'] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
    manifest_path = tmp_path / 'manifest.json'
    dump(manifest_path, manifest)
    registration = deepcopy(json.loads(Path('experiments/event-v1.json').read_text()))
    registration.update(dataset_sha256=dataset.sha256, manifest_sha256=sha(manifest_path),
                        outer_test_clips=len(manifest['partitions']['test']['clip_ids']), config_sha256={})
    config_paths = {}
    for model in ('frame', 'temporal'):
        config = ExperimentConfig(model=model, epochs=2, batch_size=3, hidden_size=4)
        path = tmp_path / f'{model}-config.json'
        dump(path, config.to_dict())
        registration['config_sha256'][model] = sha(path)
        config_paths[model] = path
    registration_path = tmp_path / 'registration.json'
    dump(registration_path, registration)
    catalog = {'state': 'completed', 'outer_test_selected': False, 'model_identity': pipeline,
               'config': {'partitions': ['train', 'validation'], 'manifest_sha256': sha(manifest_path),
                          'annotations_sha256': dataset.sha256},
               'completed_clip_count': len(paths), 'target_clip_count': len(paths), 'completed': {}}
    for clip_id, path in paths.items():
        clip = load_feature_cache(path)
        catalog['completed'][str(clip_id)] = {'cache_sha256': sha(path), 'cache_key': clip.metadata['cache_key']}
    catalog_path = tmp_path / 'training-catalog.json'
    dump(catalog_path, catalog)
    contract = {'schema_version': 1, 'registration_path': str(registration_path),
                'registration_sha256': sha(registration_path), 'dataset_sha256': dataset.sha256,
                'manifest_sha256': sha(manifest_path), 'dependency_lock_sha256': sha('uv.lock'),
                'numpy_version': np.__version__, 'outer_fold': 1,
                'expected_test_clips': len(manifest['partitions']['test']['clip_ids']),
                'feature_pipeline': pipeline, 'heads': {}, 'contexts': list(heldout.CONTEXTS),
                'selection': 'best_model', 'timestamp_provenance': 'reencoded-clip-pts',
                'order_constrained_decoding': False, 'device': 'cpu', 'torch_num_threads': 1,
                'training_cache_catalog_path': str(catalog_path),
                'training_cache_catalog_sha256': sha(catalog_path)}
    for model in ('frame', 'temporal'):
        config = ExperimentConfig.from_dict(json.loads(config_paths[model].read_text()))
        driver = HeadExperiment(config, dataset, manifest, paths)
        driver.advance()
        checkpoint_path = tmp_path / f'{model}.pt'
        checkpoint_hash = driver.save(checkpoint_path)
        run_path = tmp_path / f'{model}-run.json'
        run = {'state': 'completed', 'config': config.to_dict(), 'summary': driver.summary(),
               'checkpoints': [{'path': str(checkpoint_path.resolve()), 'sha256': checkpoint_hash}],
               'immutable': {'config_sha256': sha(config_paths[model]),
                             'registration_sha256': sha(registration_path),
                             'cache_job_sha256': sha(catalog_path), 'manifest_sha256': sha(manifest_path),
                             'dependency_lock_sha256': sha('uv.lock'),
                             'cli_sha256': sha('scripts/train_event_head.py')},
               'outer_test_evaluated': False, 'validated_for_serving': False}
        dump(run_path, run)
        contract['heads'][model] = {'run_record_path': str(run_path), 'run_record_sha256': sha(run_path),
                                    'checkpoint_path': str(checkpoint_path), 'checkpoint_sha256': checkpoint_hash,
                                    'selected_epoch': driver.best_epoch, 'config_path': str(config_paths[model]),
                                    'config_sha256': sha(config_paths[model])}
    contract_path = tmp_path / 'contract.json'
    dump(contract_path, contract)
    clips_dir = tmp_path / 'media'
    clips_dir.mkdir()
    outer_rows, inventory, timelines = {}, [], {}
    records = {r.clip_id: r for r in dataset.records}
    for clip_id in manifest['partitions']['test']['clip_ids']:
        record = records[clip_id]
        n = record.expected_frame_count
        video = clips_dir / f'{clip_id}.mp4'
        video.write_bytes(f'SYNTHETIC SOFTWARE MEDIA STUB {clip_id}'.encode())
        identity = {'schema_version': 1, 'annotation_sha256': dataset.sha256, 'clip_id': clip_id,
                    'source_id': record.source_id, 'outer_split': record.outer_split,
                    'original_events': list(record.original_events), 'expected_frame_count': n,
                    'annotation_event_indices': list(record.event_indices),
                    'historical_source_indices': [record.source_frame_index(i) for i in range(n)],
                    'timestamp_provenance': 'reencoded-clip-pts', 'video_sha256': sha(video), **pipeline}
        arrays = {'features': np.random.default_rng(clip_id).normal(size=(n, 1280)).astype(np.float32),
                  'labels': np.r_[np.arange(8), [8] * (n - 8)].astype(np.int64),
                  'pts_seconds': np.arange(n, dtype=np.float64) ** 2 / 100,
                  'historical_source_indices': np.arange(n, dtype=np.int64)}
        meta = {'identity': identity, 'cache_key': cache_key(identity),
                'array_sha256': {name: _array_hash(value) for name, value in arrays.items()}}
        cache_path = tmp_path / f'outer-{clip_id}.npz'
        save_feature_cache(FeatureClip(**arrays, metadata=meta), cache_path)
        timelines[clip_id] = SimpleNamespace(pts_seconds=tuple(arrays['pts_seconds']))
        outer_rows[str(clip_id)] = {'clip_id': clip_id, 'source_id': record.source_id, 'partition': 'test',
                                    'cache_path': str(cache_path), 'cache_sha256': sha(cache_path),
                                    'cache_key': cache_key(identity), 'frames': n}
        inventory.append({'clip_id': clip_id, 'source_id': record.source_id, 'partition': 'test',
                          'video_sha256': sha(video)})
    monkeypatch.setattr(heldout, 'inspect_clip_timeline', lambda _, record, **kwargs: timelines[record.clip_id])
    config = {'partitions': ['test'], 'allow_test': True, 'weights_request': 'IMAGENET1K_V2',
              'annotations_sha256': dataset.sha256, 'manifest_sha256': sha(manifest_path),
              'dependency_lock_sha256': sha('uv.lock'), 'inventory': inventory, 'clips_dir': str(clips_dir)}
    job = {'state': 'completed', 'outer_test_selected': True, 'config': config,
           'config_sha256': heldout._json_sha(config), 'model_identity': pipeline,
           'completed_clip_count': len(outer_rows), 'target_clip_count': len(outer_rows), 'completed': outer_rows}
    job_path = tmp_path / 'outer-job.json'
    dump(job_path, job)
    return SimpleNamespace(contract=contract, contract_path=contract_path, dataset=dataset,
                           manifest=manifest, manifest_path=manifest_path, job=job, job_path=job_path,
                           root=tmp_path / 'evaluation', annotations=tmp_path / 'annotations.mat',
                           timelines=timelines)


def run(fixture, **kwargs):
    return heldout.run_heldout_evaluation(fixture.contract_path, fixture.annotations,
                                          fixture.manifest_path, fixture.job_path, fixture.root, **kwargs)


def test_contract_admission_constructs_only_selected_eval_heads(admitted):
    result = heldout.validate_contract(admitted.contract_path, admitted.annotations, admitted.manifest_path)
    assert set(result['heads']) == {'frame', 'temporal'}
    assert all(not head.training and not any(p.requires_grad for p in head.parameters())
               for head in result['heads'].values())
    assert result['immutable']['contract_sha256'] == sha(admitted.contract_path)


def test_bounded_resume_equals_continuous_and_skips_durable_inference(admitted, monkeypatch):
    first = run(admitted, max_new_clips=1)
    assert first['state'] == 'partial' and len(first['completed']) == 1 and first['summary'] is None
    with pytest.raises(ValueError, match='explicitly resume'):
        run(admitted)
    original = heldout.predict_probabilities
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(heldout, 'predict_probabilities', counted)
    complete = run(admitted, resume=True)
    assert complete['state'] == 'completed'
    assert len(calls) == (len(admitted.job['completed']) - 1) * 4
    assert complete['validated_for_serving'] is False
    for head in complete['summary'].values():
        for context, value in head.items():
            assert value['clip_count'] == len(admitted.job['completed'])
            assert value['partition'] == 'outer-test' and value['temporal_context'] == context
            assert value['order_constrained_decoding'] is False
    old_count = len(calls)
    assert run(admitted, resume=True) == complete
    assert len(calls) == old_count
    admitted.root = admitted.root.with_name('continuous')
    continuous = run(admitted)
    assert continuous['summary'] == complete['summary']
    assert continuous['completed'] == complete['completed']


def test_reference_chunk_boundaries_and_final_tail_reset_without_padding():
    class Recording(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lengths = []
        def forward(self, x, lengths):
            self.lengths.append((x.shape[1], lengths.tolist(), float(x[0, 0, 0])))
            logits = torch.zeros((1, x.shape[1], 9))
            # Output depends on start of THIS call, making hidden-context carry detectable.
            logits[..., 0] = x[0, 0, 0]
            return logits
    model = Recording()
    features = np.zeros((130, 1280), dtype=np.float32)
    features[:, 0] = np.arange(130)
    actual = heldout.predict_probabilities(model, features, 'reference-64-reset')
    assert model.lengths == [(64, [64], 0.), (64, [64], 64.), (2, [2], 128.)]
    expected = np.concatenate([torch.softmax(torch.tensor([[start] + [0.] * 8]), -1).numpy().repeat(n, axis=0)
                               for start, n in [(0., 64), (64., 64), (128., 2)]])
    np.testing.assert_array_equal(actual, expected)
    model.lengths.clear()
    full = heldout.predict_probabilities(model, features, 'full-clip')
    assert model.lengths == [(130, [130], 0.)]
    assert not np.array_equal(actual, full)


def test_frame_context_invariance_and_real_bilstm_chunk_reset():
    torch.set_num_threads(1)
    features = np.random.default_rng(7).normal(size=(65, 1280)).astype(np.float32)
    frame = FrameClassifier().eval()
    # Per-frame arithmetic can differ at low ULPs with BLAS batch shape, not context.
    np.testing.assert_allclose(heldout.predict_probabilities(frame, features, 'reference-64-reset'),
                               heldout.predict_probabilities(frame, features, 'full-clip'), rtol=1e-6, atol=1e-7)
    temporal = TemporalClassifier(hidden_size=4).eval()
    chunked = heldout.predict_probabilities(temporal, features, 'reference-64-reset')
    first = heldout.predict_probabilities(temporal, features[:64], 'full-clip')
    tail = heldout.predict_probabilities(temporal, features[64:], 'full-clip')
    np.testing.assert_array_equal(chunked, np.concatenate([first, tail]))
    assert not np.array_equal(chunked, heldout.predict_probabilities(temporal, features, 'full-clip'))


@pytest.mark.parametrize('problem', ['partial', 'train_partition', 'extra', 'pipeline', 'media', 'pts', 'cache'])
def test_incomplete_leaking_or_changed_outer_inventory_refused_before_run(admitted, problem):
    if problem == 'partial':
        admitted.job['state'] = 'partial'
    elif problem == 'train_partition':
        admitted.job['config']['partitions'] = ['train', 'test']
    elif problem == 'extra':
        admitted.job['completed']['999'] = next(iter(admitted.job['completed'].values()))
    elif problem == 'pipeline':
        admitted.job['model_identity']['transform']['center_crop'] = 160
    elif problem == 'media':
        video = Path(admitted.job['config']['clips_dir']) / '0.mp4'
        video.write_bytes(b'changed')
    elif problem == 'pts':
        admitted.timelines[0].pts_seconds = tuple(np.arange(9, dtype=float))
    else:
        path = Path(admitted.job['completed']['0']['cache_path'])
        path.write_bytes(path.read_bytes() + b'changed')
    dump(admitted.job_path, admitted.job)
    with pytest.raises(ValueError):
        run(admitted)
    assert not admitted.root.exists()


@pytest.mark.parametrize('problem', ['contract_runtime', 'checkpoint', 'run', 'selection', 'catalog', 'source_split'])
def test_changed_frozen_admission_is_refused(admitted, problem):
    if problem == 'contract_runtime':
        admitted.contract['numpy_version'] = 'unregistered'
    elif problem == 'checkpoint':
        path = Path(admitted.contract['heads']['frame']['checkpoint_path'])
        path.write_bytes(path.read_bytes() + b'changed')
    elif problem == 'run':
        path = Path(admitted.contract['heads']['frame']['run_record_path'])
        path.write_text(path.read_text() + ' ')
    elif problem == 'selection':
        admitted.contract['heads']['frame']['selected_epoch'] = 999
    elif problem == 'catalog':
        path = Path(admitted.contract['training_cache_catalog_path'])
        value = json.loads(path.read_text())
        value['completed']['0'] = next(iter(value['completed'].values()))
        dump(path, value)
        admitted.contract['training_cache_catalog_sha256'] = sha(path)
    else:
        value = deepcopy(admitted.manifest)
        value['partitions']['train']['clip_ids'][0] = value['partitions']['test']['clip_ids'][0]
        dump(admitted.manifest_path, value)
        admitted.contract['manifest_sha256'] = sha(admitted.manifest_path)
    dump(admitted.contract_path, admitted.contract)
    with pytest.raises(ValueError):
        run(admitted)
    assert not admitted.root.exists()


def test_resume_rejects_row_metric_tamper_and_changed_immutable_contract(admitted):
    state = run(admitted, max_new_clips=1)
    key = next(iter(state['completed']))
    state['completed'][key]['results']['frame']['full-clip']['pce'] = .999
    dump(admitted.root / 'run.json', state)
    with pytest.raises(ValueError, match='checksum'):
        run(admitted, resume=True)
    state['completed'][key]['sha256'] = heldout._row_hash(state['completed'][key])
    dump(admitted.root / 'run.json', state)
    with pytest.raises(ValueError, match='metrics differ'):
        run(admitted, resume=True)


def test_selected_weights_not_final_model_and_rng_is_not_restored(admitted):
    head = admitted.contract['heads']['frame']
    saved = torch.load(head['checkpoint_path'], map_location='cpu', weights_only=True)
    saved['model'] = {k: torch.zeros_like(v) for k, v in saved['model'].items()}
    path = admitted.root.with_name('distinct-final.pt')
    torch.save(saved, path)
    before = torch.get_rng_state().clone()
    model = heldout.load_selected_head(path, sha(path), 'frame', hidden_size=4)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(v, saved['best_model'][k]) for k, v in model.state_dict().items())
    assert any(not torch.equal(v, saved['model'][k]) for k, v in model.state_dict().items())


def test_shared_lock_exclusion(admitted):
    import fcntl
    with heldout.DEFAULT_LOCK_PATH.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='shared lock'):
            run(admitted)
    assert not admitted.root.exists()


def test_resume_adopts_final_clip_before_summary_crash_without_inference(admitted, monkeypatch):
    complete = run(admitted)
    crashed = deepcopy(complete)
    crashed.update(state='interrupted', summary=None, error='simulated final-summary interruption')
    crashed['attempts'][-1].update(state='interrupted')
    dump(admitted.root / 'run.json', crashed)
    def forbidden(*args, **kwargs):
        raise AssertionError('Durable final clips must not be rerun')
    monkeypatch.setattr(heldout, 'predict_probabilities', forbidden)
    recovered = run(admitted, resume=True)
    assert recovered['state'] == 'completed' and recovered['summary'] == complete['summary']
    assert recovered['completed'] == complete['completed']
    assert recovered['attempts'][0]['state'] == 'interrupted'
    assert recovered['attempts'][-1]['new_clips'] == 0
    assert 'no inference repeated' in recovered['attempts'][-1]['recovery']


@pytest.mark.parametrize('drift', ['evaluator', 'training_source', 'runtime'])
def test_live_source_and_runtime_drift_refuses_row_publication(admitted, monkeypatch, drift):
    original_predict = heldout.predict_probabilities
    original_sha = heldout._sha
    changed = False
    def fingerprint(path):
        target = Path(heldout.__file__) if drift == 'evaluator' else Path(heldout.__file__).with_name('models.py')
        return 'f' * 64 if changed and Path(path).resolve() == target.resolve() else original_sha(path)
    def predict(*args, **kwargs):
        nonlocal changed
        result = original_predict(*args, **kwargs)
        changed = True
        if drift == 'runtime':
            torch.set_num_threads(2)
        return result
    monkeypatch.setattr(heldout, '_sha', fingerprint)
    monkeypatch.setattr(heldout, 'predict_probabilities', predict)
    try:
        with pytest.raises(ValueError, match='changed during invocation'):
            run(admitted)
    finally:
        torch.set_num_threads(1)
    state = json.loads((admitted.root / 'run.json').read_text())
    assert state['state'] == 'failed' and state['completed'] == {} and state['summary'] is None
