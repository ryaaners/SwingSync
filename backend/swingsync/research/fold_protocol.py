"""Pure admission for a prospectively fixed additional-fold study, not a training launcher.

No checkpoints, feature arrays, predictions, videos or outer metrics are opened.
Remote Git registration, cache-audit admission and future job launch are separate gates.
"""
import hashlib
import json
from pathlib import Path

from .golfdb import make_split_manifest

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / 'experiments/additional-folds-v1.json'
MAX_JSON_BYTES = 1024 * 1024
PROTOCOL_SHA256 = 'b9c7e29b2afa63e6733d0a26c5e335f7ad31463143b97300a2ea3e3d91107a53'
DATASET_SHA256 = '36523fcef016d93334691ea058d643bcf0d3a4a8c70b733f3b4cc0b8e5e0cf0a'
SPLIT = {'algorithm': 'sha256-seed-source-rank-v1-ceil', 'seed': 20261104, 'validation_fraction_sources': 0.2}
FOLD_BINDINGS = {'2': {'latency_training_clip_ids': [0, 1],
       'manifest_path': 'artifacts/golfdb/preparation-002-all/fold-2.json',
       'manifest_sha256': '12a5b80d7156de65e7a6f3716d0f60670d36cc50b05fbb5adf91d59f848f4c5a',
       'partitions': {'test': {'clip_count': 350, 'source_count': 142},
                      'train': {'clip_count': 836, 'source_count': 350},
                      'validation': {'clip_count': 214, 'source_count': 88}}},
 '3': {'latency_training_clip_ids': [6, 7],
       'manifest_path': 'artifacts/golfdb/preparation-002-all/fold-3.json',
       'manifest_sha256': '5fe0225c912b0c18ab494458a7f1d9d1d71a29207f8bf87599d6138f577071d9',
       'partitions': {'test': {'clip_count': 350, 'source_count': 154},
                      'train': {'clip_count': 839, 'source_count': 340},
                      'validation': {'clip_count': 211, 'source_count': 86}}},
 '4': {'latency_training_clip_ids': [0, 1],
       'manifest_path': 'artifacts/golfdb/preparation-002-all/fold-4.json',
       'manifest_sha256': '3d20ab4b028e77d1cb96cb952fe3c944219ac4ab1f3bc1e697508e13ebe8434d',
       'partitions': {'test': {'clip_count': 350, 'source_count': 140},
                      'train': {'clip_count': 853, 'source_count': 352},
                      'validation': {'clip_count': 197, 'source_count': 88}}}}
ORIGINAL_CONFIG_BINDINGS = {'frame': {'path': 'experiments/frame-v1.json',
           'sha256': 'a57b0c196bcfa666d3e9bf6167cf054635783286b597c1b083e6a09cc9bb2860'},
 'temporal': {'path': 'experiments/temporal-v1.json',
              'sha256': 'd9d0a22955c3a4f8e22e83d5683465798cd1b2d8868284140cf9c6887c4cd5c1'}}
FEATURE_BINDING = {'path': 'experiments/event-v1.json',
 'sha256': '0af432a6f938422cb9995513e026160a47c2f0bd516c5d7245223d3579cc7c3e'}
HELDOUT_BINDING = {'path': 'experiments/heldout-v1.json',
 'sha256': '4b4339a469dd5fe09f97b57ae4b650ce378ce1c3a3863f86b46249ad8d9d0008'}
LOCK_SHA256 = '8ce1d6674a07bf122541f555b90ad81532029b7ce343af54a78cd057fc2b26fc'
FROZEN_SOURCES = {'backend/swingsync/research/cache_job.py': 'd1d92638c45b01410ef77744017242f403674688fb157815efa28aad460954d7',
 'backend/swingsync/research/evaluation.py': 'fe5c6f49282b4f6782181badad47948ca0b9545a6be7a0d878899d898e9a0f19',
 'backend/swingsync/research/experiment.py': 'cb16092361486cf911bff9dc4deb8aa0e72f3a62a713e63a950cf05aae28841d',
 'backend/swingsync/research/features.py': '0319843f83bf6453eefeadd431a88c6d42186b6a94e541d70672f272be53204d',
 'backend/swingsync/research/golfdb.py': '5d1cb1e60165e1c62c7df2ff6241b307c192d00a244fe7270f16ae4d38369b70',
 'backend/swingsync/research/heldout.py': '259346d903d25328b0d2c0c54db7406be13619d960d36cb840bfdc0049a57a09',
 'backend/swingsync/research/latency.py': '1ba9aac78f071fc77c2314f9f711a7cf5f56ab86be574a632a55f38b1f43b5c7',
 'backend/swingsync/research/models.py': '2101475e4d5f95fe1d5c1e54bc172038b08475638474be90900e410cff7899ba',
 'backend/swingsync/research/timeline.py': '0d573f013bc5df27481d18bf50eb321bca5bd4aae94968fd4a521019cfd1da4a',
 'backend/swingsync/research/training.py': 'e835d790866a75c1d39e36c7522809ccc98dcfc5950dced3ceba53e9c5b7ea0d'}
CATALOG_BINDINGS = {'test': {'path': 'artifacts/features/fold1-test-job-v1/job.json',
          'sha256': '159ec421d82ca82d66cfe070d4272e013c65604365cf2c31fc11a57af0d66cf4'},
 'train_validation': {'path': 'artifacts/features/fold1-job-v1/completed-catalog-v1.json',
                      'sha256': 'daf5f108512fc8468476722e7a751e1d95fe64e51e045a9bc95ab65df9918b9e'}}
TRAINING = {'augmentation': 'none',
 'batch_size': 4,
 'class_weight_rule': 'inverse-class-frequency-normalized-mean1-training-labels-only',
 'config_files_are_unchanged': True,
 'deterministic_algorithms': True,
 'device': 'cpu',
 'epochs': 20,
 'frozen_imagenet_backbone_only': True,
 'full_training_warm_start_from_smoke': False,
 'head_initialization': 'from-scratch-per-fold-and-model',
 'hidden_size': 64,
 'inner_validation_context': 'full-clip',
 'learning_rate': 0.001,
 'optimizer': 'Adam',
 'outer_test_selection_or_tuning': False,
 'previous_head_warm_start': False,
 'sample_unit': 'whole-clips',
 'seed': 20261105,
 'selection_metric': 'standard-inner-validation-pce',
 'selection_state': 'best_model',
 'selection_tie_break': 'earliest-epoch',
 'smoke_optimizer_batches_per_model': 5,
 'smoke_partition': 'training-only',
 'torch_num_threads': 1,
 'training_context': 'full-clip'}
EVALUATION = {'aggregation': 'publish-additional-fold-results-only-after-all-three-folds-complete;four-fold-summary-separately-labels-historical-fold1',
 'confidence_calibration': False,
 'contexts': ['reference-64-reset', 'full-clip'],
 'full_clip_reported_separately': True,
 'heads': ['frame', 'temporal'],
 'latency_batch_size': 16,
 'latency_contexts': ['reference-64-reset', 'full-clip'],
 'latency_partition': 'two-fixed-training-clips-per-fold',
 'latency_repetitions': 1,
 'latency_warmup_frames': 16,
 'order_constrained_decoding': False,
 'outer_result_driven_reselection_or_repeated_tuning': False,
 'partial_results_are_final': False,
 'primary_context': 'reference-64-reset',
 'required_metrics': ['overall-pce',
                      'per-event-pce',
                      'timing-error-frames',
                      'timing-error-native-pts-milliseconds',
                      'inference-latency-on-recorded-hardware'],
 'target_policy': 'chronological-occupied-frame-target-slots-including-677',
 'tie_policy': 'numpy-default-argsort-last-index-per-event',
 'tolerance_policy': 'max(numpy-round((chronological-target[5]-chronological-target[0])/30),1);inclusive-error<=tolerance'}
LAUNCH_GATES = {'completed_fold1_may_be_retrained_or_retuned': False,
 'max_expensive_jobs': 1,
 'registration_alone_authorizes_training': False,
 'require_exact_registration_sha256': True,
 'require_independently_reviewed_cache_audit_and_known_sha256': True,
 'require_pushed_registration_commit_before_fitting': True,
 'require_resumable_checkpoint_and_job_identity': True,
 'require_reviewed_training_launcher': True,
 'require_shared_expensive_work_lock': True,
 'require_train_only_smoke_before_full_run': True,
 'resume_requires_unchanged_inputs_code_runtime_config': True}
LIMITATIONS = {'additional_outer_clips_were_in_historical_fold1_train_or_inner': True,
 'fold1_results_already_observed': True,
 'no_new_results_in_this_registration': True,
 'same_pixel_cache_reuse_is_upload_domain_validation': False,
 'source_time_basis': 'reencoded-clip-pts-not-original-source-time',
 'split_generalization_unit': 'source-recording-not-verified-unseen-golfer',
 'study_is_wholly_prospective_four_fold': False}
ADMISSION_FLAGS = {'coaching_validated': False,
 'outcome_integration_permitted': False,
 'pose_measurement_reliability_validated': False,
 'registration_contains_evaluation_results': False,
 'registration_contains_training_results': False,
 'uploaded_video_compatibility_validated': False,
 'validated_for_serving': False}
SOURCE_REFERENCES = ['https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/eval.py',
 'https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/util.py']


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate protocol JSON key')
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError('Nonfinite protocol JSON value')


def _json(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise ValueError('Protocol JSON is empty or exceeds one MiB')
    try:
        return raw, json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Malformed protocol JSON') from exc


def _same(actual, expected):
    """Equality must not let bools masquerade as integers or coerce numbers."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(_same(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _expected_document():
    return {'schema_version': 1, 'study_id': 'additional-official-folds-v1', 'new_outer_folds': [2, 3, 4],
            'dataset_sha256': DATASET_SHA256, 'split': SPLIT, 'folds': FOLD_BINDINGS,
            'original_configs': ORIGINAL_CONFIG_BINDINGS, 'feature_registration': FEATURE_BINDING,
            'historical_heldout_registration': HELDOUT_BINDING, 'dependency_lock_sha256': LOCK_SHA256,
            'frozen_source_sha256': FROZEN_SOURCES, 'cache_catalogs': CATALOG_BINDINGS,
            'training': TRAINING, 'evaluation': EVALUATION, 'launch_gates': LAUNCH_GATES,
            'limitations': LIMITATIONS, 'admission_flags': ADMISSION_FLAGS,
            'source_references': SOURCE_REFERENCES}


def validate_document(document):
    """Strict operational schema check only; load via validate_protocol for byte admission."""
    if not _same(document, _expected_document()):
        raise ValueError('Additional-fold protocol differs from the fixed operational registration')
    return document


def _local_bindings():
    bindings = {'uv.lock': LOCK_SHA256, **FROZEN_SOURCES}
    for binding in (FEATURE_BINDING, HELDOUT_BINDING, *ORIGINAL_CONFIG_BINDINGS.values()):
        bindings[binding['path']] = binding['sha256']
    for relative, expected in bindings.items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError('Frozen registration/config/source/dependency binding changed: '+relative)
    for model, binding in ORIGINAL_CONFIG_BINDINGS.items():
        _, config = _json(ROOT / binding['path'])
        expected = {'schema_version': 1, 'seed': 20261105, 'model': model, 'epochs': 20, 'batch_size': 4,
                    'learning_rate': 0.001, 'hidden_size': 64, 'selection_metric': 'validation_pce',
                    'context': 'full-clip', 'augmentation': 'none', 'device': 'cpu'}
        if not _same(config, expected):
            raise ValueError('Historical head configuration changed')


def validate_manifest(dataset, manifest, fold, binding):
    """Regenerate complete IDs/source grouping; this helper does not authenticate dataset bytes."""
    expected = make_split_manifest(dataset, fold, SPLIT['seed'], SPLIT['validation_fraction_sources'])
    if not _same(manifest, expected):
        raise ValueError('Additional-fold manifest differs from source-grouped annotation partitions')
    counts = {name: {'clip_count': len(part['clip_ids']), 'source_count': len(part['source_ids'])}
              for name, part in expected['partitions'].items()}
    if not _same(counts, binding['partitions']):
        raise ValueError('Additional-fold clip/source counts differ from registration')
    if expected['partitions']['train']['clip_ids'][:2] != binding['latency_training_clip_ids']:
        raise ValueError('Fixed latency inputs must be the registered first two training clips')
    ids = [set(p['clip_ids']) for p in manifest['partitions'].values()]
    sources = [set(p['source_ids']) for p in manifest['partitions'].values()]
    if any(a & b for values in (ids, sources) for i, a in enumerate(values) for b in values[i+1:]):
        raise ValueError('Additional-fold partitions overlap')
    return manifest


def validate_protocol(path=DEFAULT_PROTOCOL, *, dataset=None, manifests_dir=None):
    """Authenticate pinned study bytes and frozen code/config; optionally check actual manifests.

    This does not read cache catalogs/arrays, load models, prove a Git push, or
    authorize a training job. Actual audit and reviewed future launcher are separate.
    """
    path = Path(path)
    raw, document = _json(path)
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('Additional-fold protocol bytes differ from the registered SHA256')
    validate_document(document)
    _local_bindings()
    if (dataset is None) != (manifests_dir is None):
        raise ValueError('Provide both annotations and manifest directory, or neither')
    if dataset is not None:
        if dataset.sha256 != DATASET_SHA256 or len(dataset.records) != 1400:
            raise ValueError('Additional-fold annotations differ from the registered dataset')
        outer_ids = set()
        for fold in document['new_outer_folds']:
            binding = document['folds'][str(fold)]
            manifest_path = Path(manifests_dir) / ('fold-'+str(fold)+'.json')
            manifest_raw, manifest = _json(manifest_path)
            if hashlib.sha256(manifest_raw).hexdigest() != binding['manifest_sha256']:
                raise ValueError('Additional-fold manifest bytes differ from registration')
            validate_manifest(dataset, manifest, fold, binding)
            test_ids = set(manifest['partitions']['test']['clip_ids'])
            if outer_ids & test_ids:
                raise ValueError('Additional official outer folds overlap')
            outer_ids |= test_ids
        expected_ids = {r.clip_id for r in dataset.records if r.outer_split in (2, 3, 4)}
        if outer_ids != expected_ids:
            raise ValueError('Additional official outer-fold coverage differs')
    if file_sha256(path) != PROTOCOL_SHA256:
        raise ValueError('Additional-fold protocol changed during validation')
    return document
