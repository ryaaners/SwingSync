"""Additional-fold registration contracts; synthetic schemas are not model results."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from backend.swingsync.research import fold_protocol as protocol
from backend.swingsync.research.golfdb import load_annotations, make_split_manifest
from test_golfdb import write_mat


def document():
    return json.loads(protocol.DEFAULT_PROTOCOL.read_text())


def test_registered_protocol_loads_without_models_or_dataset():
    value = protocol.validate_protocol()
    assert value['new_outer_folds'] == [2, 3, 4]
    assert value['training']['previous_head_warm_start'] is False
    assert value['launch_gates']['registration_alone_authorizes_training'] is False
    assert not any(value['admission_flags'].values())
    assert value['limitations']['fold1_results_already_observed'] is True
    assert value['folds']['3']['latency_training_clip_ids'] == [6, 7]
    assert protocol.file_sha256(protocol.DEFAULT_PROTOCOL) == protocol.PROTOCOL_SHA256


@pytest.mark.parametrize('edit', [
    lambda d: d.update(schema_version=True),
    lambda d: d.update(new_outer_folds=[1, 2, 3, 4]),
    lambda d: d.update(unknown='field'),
    lambda d: d['split'].update(seed=True),
    lambda d: d['split'].update(validation_fraction_sources=1),
    lambda d: d['training'].update(epochs=20.0),
    lambda d: d['training'].update(learning_rate=True),
    lambda d: d['training'].update(previous_head_warm_start=True),
    lambda d: d['training'].update(augmentation='random-independent-frames'),
    lambda d: d['training'].update(class_weight_rule='all-labels'),
    lambda d: d['training'].update(selection_metric='outer-test-pce'),
    lambda d: d['training'].update(full_training_warm_start_from_smoke=True),
    lambda d: d['evaluation'].update(contexts=['full-clip']),
    lambda d: d['evaluation'].update(order_constrained_decoding=True),
    lambda d: d['evaluation'].update(target_policy='semantic-event-labels-with-677-repair'),
    lambda d: d['evaluation'].update(partial_results_are_final=True),
    lambda d: d['evaluation'].update(confidence_calibration=True),
    lambda d: d['launch_gates'].update(require_pushed_registration_commit_before_fitting=False),
    lambda d: d['launch_gates'].update(require_independently_reviewed_cache_audit_and_known_sha256=False),
    lambda d: d['launch_gates'].update(max_expensive_jobs=True),
    lambda d: d['limitations'].update(study_is_wholly_prospective_four_fold=True),
    lambda d: d['limitations'].update(split_generalization_unit='unseen-golfer'),
    lambda d: d['admission_flags'].update(validated_for_serving=True),
    lambda d: d['admission_flags'].update(coaching_validated=0),
    lambda d: d['folds']['2']['partitions']['train'].update(clip_count=840),
    lambda d: d['folds']['3'].update(latency_training_clip_ids=[0, 1]),
    lambda d: d['original_configs']['frame'].update(sha256='0'*64),
    lambda d: d['cache_catalogs']['test'].update(sha256='0'*64),
])
def test_operational_changes_and_type_coercion_are_refused(edit):
    value = document()
    edit(value)
    with pytest.raises(ValueError, match='operational registration'):
        protocol.validate_document(value)


@pytest.mark.parametrize('raw,match', [
    (b'{"schema_version":1,"schema_version":1}', 'Duplicate'),
    (b'{"schema_version":NaN}', 'Nonfinite'),
    (b'\xff', 'Malformed'),
    (b'{' , 'Malformed'),
    (b'['*2000+b']'*2000, 'Malformed'),
    (b' '* (protocol.MAX_JSON_BYTES+1), 'one MiB'),
])
def test_untrusted_protocol_json_is_bounded_and_strict(tmp_path, raw, match):
    path = tmp_path/'bad.json'
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=match):
        protocol.validate_protocol(path)


def test_semantically_equal_reformatted_protocol_is_not_registered_bytes(tmp_path):
    path = tmp_path/'reformatted.json'
    path.write_text(json.dumps(document()))
    with pytest.raises(ValueError, match='registered SHA256'):
        protocol.validate_protocol(path)


@pytest.mark.parametrize('relative', ['uv.lock', *protocol.FROZEN_SOURCES,
                                     'experiments/frame-v1.json', 'experiments/temporal-v1.json',
                                     'experiments/event-v1.json', 'experiments/heldout-v1.json'])
def test_historical_code_config_and_lock_drift_is_refused(monkeypatch, relative):
    real_hash = protocol.file_sha256
    def observed_hash(path):
        return '0'*64 if path == protocol.ROOT/relative else real_hash(path)
    monkeypatch.setattr(protocol, 'file_sha256', observed_hash)
    with pytest.raises(ValueError, match='binding changed'):
        protocol.validate_protocol()


def test_protocol_final_publication_window_drift_is_refused(monkeypatch):
    real_hash = protocol.file_sha256
    monkeypatch.setattr(protocol, 'file_sha256', lambda path: '0'*64 if path == protocol.DEFAULT_PROTOCOL else real_hash(path))
    with pytest.raises(ValueError, match='changed during validation'):
        protocol.validate_protocol()


@pytest.fixture
def synthetic_manifest(tmp_path):
    dataset = load_annotations(write_mat(tmp_path/'synthetic.mat'))
    manifest = make_split_manifest(dataset, 2, protocol.SPLIT['seed'], protocol.SPLIT['validation_fraction_sources'])
    binding = {'partitions': {name: {'clip_count': len(p['clip_ids']), 'source_count': len(p['source_ids'])}
                             for name, p in manifest['partitions'].items()},
               'latency_training_clip_ids': manifest['partitions']['train']['clip_ids'][:2]}
    return dataset, manifest, binding


def test_manifest_helper_checks_whole_source_partitions(synthetic_manifest):
    dataset, manifest, binding = synthetic_manifest
    before = deepcopy(manifest)
    assert protocol.validate_manifest(dataset, manifest, 2, binding) == before
    assert manifest == before
    sources = [set(p['source_ids']) for p in manifest['partitions'].values()]
    assert not any(a & b for i, a in enumerate(sources) for b in sources[i+1:])


@pytest.mark.parametrize('edit', [
    lambda m: m['partitions']['train']['clip_ids'].reverse(),
    lambda m: m['partitions']['train']['clip_ids'].append(m['partitions']['test']['clip_ids'][0]),
    lambda m: m['partitions']['train']['source_ids'].append(m['partitions']['test']['source_ids'][0]),
    lambda m: m['partitions']['train']['clips'][0].update(source_id='wrong-source'),
    lambda m: m.update(seed=999),
    lambda m: m.update(outer_fold=1),
])
def test_rebound_manifest_ids_grouping_or_fold_cannot_change(synthetic_manifest, edit):
    dataset, manifest, binding = synthetic_manifest
    edit(manifest)
    with pytest.raises(ValueError, match='source-grouped annotation partitions'):
        protocol.validate_manifest(dataset, manifest, 2, binding)


def test_manifest_count_and_latency_selection_must_match_registration(synthetic_manifest):
    dataset, manifest, binding = synthetic_manifest
    changed = deepcopy(binding)
    changed['partitions']['train']['clip_count'] += 1
    with pytest.raises(ValueError, match='clip/source counts'):
        protocol.validate_manifest(dataset, manifest, 2, changed)
    binding['latency_training_clip_ids'] = manifest['partitions']['test']['clip_ids'][:2]
    with pytest.raises(ValueError, match='latency inputs'):
        protocol.validate_manifest(dataset, manifest, 2, binding)


def test_actual_manifest_verification_requires_both_inputs_and_exact_dataset(synthetic_manifest, tmp_path):
    dataset, _, _ = synthetic_manifest
    with pytest.raises(ValueError, match='both annotations'):
        protocol.validate_protocol(dataset=dataset)
    with pytest.raises(ValueError, match='both annotations'):
        protocol.validate_protocol(manifests_dir=tmp_path)
    with pytest.raises(ValueError, match='registered dataset'):
        protocol.validate_protocol(dataset=dataset, manifests_dir=tmp_path)
    # A guessed SHA cannot turn a sixteen-row fixture into the registered dataset.
    with pytest.raises(ValueError, match='registered dataset'):
        protocol.validate_protocol(dataset=replace(dataset, sha256=protocol.DATASET_SHA256), manifests_dir=tmp_path)
