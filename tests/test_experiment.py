"""Synthetic frozen caches establish tuning/resume mechanics, never ML performance."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from backend.swingsync.research.experiment import ExperimentConfig, HeadExperiment  # noqa: E402
from backend.swingsync.research.features import FeatureClip, _array_hash, cache_key, save_feature_cache  # noqa: E402
from backend.swingsync.research.golfdb import GolfDBAnnotations, GolfDBRecord, make_split_manifest  # noqa: E402


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def inputs(tmp_path):
    records = []
    for i in range(16):
        length = 9 + i % 3
        records.append(GolfDBRecord(i, f'source-{i // 2}', 'fixture', 'unknown', 'iron', 'face-on', False,
                                    (1, 1, 2, 3, 4, 5, 6, 7, 8, length), (0., 0., 1., 1.), i // 4 + 1))
    dataset = GolfDBAnnotations(tuple(records), 'a' * 64)
    manifest = make_split_manifest(dataset, outer_fold=1, seed=17, validation_fraction=.3)
    paths = {}
    selected_ids = set(manifest['partitions']['train']['clip_ids']) | set(
        manifest['partitions']['validation']['clip_ids'])
    for record in records:
        if record.clip_id not in selected_ids:
            continue
        n = record.expected_frame_count
        identity = {
            'schema_version': 1, 'annotation_sha256': dataset.sha256,
            'clip_id': record.clip_id, 'source_id': record.source_id, 'outer_split': record.outer_split,
            'original_events': list(record.original_events), 'expected_frame_count': n,
            'annotation_event_indices': list(record.event_indices),
            'historical_source_indices': list(range(n)), 'timestamp_provenance': 'reencoded-clip-pts',
            'backbone_state_sha256': 'b' * 64, 'backbone_class': 'synthetic-software-fixture',
            'backbone_provenance': {'weights': 'synthetic-features-not-a-trained-backbone'},
            'transform': {'name': 'synthetic-no-image-transform'},
        }
        arrays = {'features': np.random.default_rng(record.clip_id).normal(size=(n, 1280)).astype(np.float32),
                  'labels': np.r_[np.arange(8), [8] * (n - 8)].astype(np.int64),
                  'pts_seconds': np.arange(n, dtype=np.float64) ** 2 / 100,
                  'historical_source_indices': np.arange(n, dtype=np.int64)}
        meta = {'identity': identity, 'cache_key': cache_key(identity),
                'array_sha256': {name: _array_hash(value) for name, value in arrays.items()}}
        path = tmp_path / f'{record.clip_id}.npz'
        save_feature_cache(FeatureClip(**arrays, metadata=meta), path)
        paths[record.clip_id] = path
    return dataset, manifest, paths


def config(model='frame'):
    return ExperimentConfig(model=model, epochs=2, batch_size=3, hidden_size=4)


def test_config_exact_schema_and_restricted_tuning_protocol():
    value = ExperimentConfig().to_dict()
    assert ExperimentConfig.from_dict(value).to_dict() == value
    for bad in ({}, dict(value, extra='unregistered'), dict(value, context='reference-64'),
                dict(value, augmentation='random-frame-flip'), dict(value, device='mps'),
                dict(value, selection_metric='outer_test_pce'), dict(value, epochs=True),
                dict(value, learning_rate=float('nan'))):
        with pytest.raises(ValueError):
            ExperimentConfig.from_dict(bad)


def test_complete_source_partition_coverage_and_outer_test_forbidden(inputs):
    dataset, manifest, paths = inputs
    missing = dict(paths)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match='coverage'):
        HeadExperiment(config(), dataset, manifest, missing)
    outer_id = manifest['partitions']['test']['clip_ids'][0]
    with pytest.raises(ValueError, match='outer test'):
        HeadExperiment(config(), dataset, manifest, {**paths, outer_id: next(iter(paths.values()))})
    tampered = json.loads(json.dumps(manifest))
    tampered['partitions']['train']['clip_ids'][0] = outer_id
    with pytest.raises(ValueError, match='manifest differs'):
        HeadExperiment(config(), dataset, tampered, paths)


def test_counts_use_every_training_clip_not_validation_or_batches(inputs):
    dataset, manifest, paths = inputs
    experiment = HeadExperiment(config(), dataset, manifest, paths)
    train_ids = manifest['partitions']['train']['clip_ids']
    expected = np.zeros(9, dtype=np.int64)
    for record in dataset.records:
        if record.clip_id in train_ids:
            expected[:8] += 1
            expected[8] += record.expected_frame_count - 8
    np.testing.assert_array_equal(experiment.class_counts, expected)
    assert len(train_ids) > experiment.config.batch_size
    with pytest.raises(ValueError, match='No complete epoch'):
        experiment.selected_model_state()
    report = experiment.advance(max_batches=1)
    assert report['completed_epochs'] == 0 and report['history'] == []
    assert report['state'] == 'partial' and report['best_epoch'] is None
    assert report['outer_test_evaluated'] is False


@pytest.mark.parametrize('model', ['frame', 'temporal'])
@pytest.mark.parametrize('resume_batches', [1, 3])
def test_mid_epoch_resume_matches_continuous_epoch_order_optimizer_and_selection(tmp_path, inputs, model,
                                                                                 resume_batches):
    dataset, manifest, paths = inputs
    continuous = HeadExperiment(config(model), dataset, manifest, paths)
    continuous.advance(max_batches=resume_batches)
    checkpoint = tmp_path / f'{model}.pt'
    digest = continuous.save(checkpoint)
    assert len(digest) == 64
    full_report = continuous.advance()
    resumed = HeadExperiment(config(model), dataset, manifest, paths)
    partial = resumed.restore(checkpoint)
    assert partial['optimizer_steps'] == resume_batches
    assert partial['clips_processed_in_current_epoch'] == (3 if resume_batches == 1 else 0)
    assert resumed.advance() == full_report
    assert all(torch.equal(value, resumed.model.state_dict()[key])
               for key, value in continuous.model.state_dict().items())
    assert all(torch.equal(value, resumed.selected_model_state()[key])
               for key, value in continuous.selected_model_state().items())
    assert full_report['state'] == 'completed' and full_report['completed_epochs'] == 2
    assert [r['epoch'] for r in full_report['history']] == [1, 2]
    assert all(r['validation']['partition'] == 'inner-validation-only' for r in full_report['history'])
    scores = [r['validation']['pce'] for r in full_report['history']]
    assert full_report['best_epoch'] == scores.index(max(scores)) + 1
    original = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        resumed.save(checkpoint)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original
    other = HeadExperiment(replace(config(model), seed=33), dataset, manifest, paths)
    with pytest.raises(ValueError, match='provenance mismatch'):
        other.restore(checkpoint)


def test_changed_cache_cannot_silently_enter_training_after_preflight(inputs):
    dataset, manifest, paths = inputs
    experiment = HeadExperiment(config(), dataset, manifest, paths)
    first_id = experiment.order[0]
    paths[first_id].write_bytes(paths[first_id].read_bytes() + b'altered bytes')
    with pytest.raises(ValueError, match='cache changed'):
        experiment.advance(max_batches=1)


def test_mixed_backbone_pipeline_is_rejected(inputs):
    dataset, manifest, paths = inputs
    changed = paths[next(iter(paths))]
    with np.load(changed, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads(arrays['metadata'].item())
    meta['identity']['backbone_state_sha256'] = 'c' * 64
    meta['cache_key'] = cache_key(meta['identity'])
    arrays['metadata'] = np.array(json.dumps(meta))
    np.savez_compressed(changed, **arrays)
    with pytest.raises(ValueError, match='exact same frozen'):
        HeadExperiment(config(), dataset, manifest, paths)


def test_changed_thread_runtime_and_summary_mutation_cannot_change_contract(inputs):
    dataset, manifest, paths = inputs
    experiment = HeadExperiment(config(), dataset, manifest, paths)
    assert experiment.provenance['runtime']['torch_num_threads'] == 1
    assert experiment.provenance['runtime']['deterministic_algorithms'] is True
    summary = experiment.summary()
    summary['history'].append({'invalid': True})
    assert experiment.summary()['history'] == []
    torch.set_num_threads(2)
    try:
        with pytest.raises(ValueError, match='runtime differs'):
            experiment.advance(max_batches=1)
    finally:
        torch.set_num_threads(1)


def test_checkpoint_selection_metadata_tamper_is_rejected_before_state_restore(tmp_path, inputs):
    dataset, manifest, paths = inputs
    experiment = HeadExperiment(config(), dataset, manifest, paths)
    experiment.advance(max_batches=1)
    checkpoint = tmp_path / 'original.pt'
    experiment.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    saved['progress']['best_validation_pce'] = .99
    bad = tmp_path / 'bad.pt'
    torch.save(saved, bad)
    resumed = HeadExperiment(config(), dataset, manifest, paths)
    before = {k: v.clone() for k, v in resumed.model.state_dict().items()}
    with pytest.raises(ValueError, match='selection differs'):
        resumed.restore(bad)
    assert all(torch.equal(value, resumed.model.state_dict()[key]) for key, value in before.items())


def test_cache_event_semantics_must_match_actual_record_not_just_cache_metadata(inputs):
    dataset, manifest, paths = inputs
    path = paths[next(iter(paths))]
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads(arrays['metadata'].item())
    meta['identity']['annotation_event_indices'][:2] = [1, 0]
    arrays['labels'][:2] = [1, 0]
    meta['array_sha256']['labels'] = _array_hash(arrays['labels'])
    meta['cache_key'] = cache_key(meta['identity'])
    arrays['metadata'] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='provenance differs'):
        HeadExperiment(config(), dataset, manifest, paths)
