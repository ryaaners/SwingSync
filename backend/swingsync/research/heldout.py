"""Frozen, once-finalized outer evaluation; never fitting, selecting or serving.

Reference recurrence is nonoverlapping 64-frame calls from frame zero, including
an unpadded final tail. Each call starts a fresh hidden state. Post-softmax chunks
are concatenated before the unchanged full-clip GolfDB evaluator runs. Full-clip
recurrence is a separately disclosed comparison, never silently substituted.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import time

import numpy as np
import torch

from .cache_job import (
    DEFAULT_LOCK_PATH, _check_previous_process, _git_base_commit, _process_start, _publish_json,
)
from .evaluation import evaluate_clip, summarize_evaluations
from .experiment import ExperimentConfig
from .features import DEFAULT_TRANSFORM, load_feature_cache
from .golfdb import load_annotations, make_split_manifest
from .models import FrameClassifier, TemporalClassifier
from .timeline import inspect_clip_timeline

ROOT = Path(__file__).resolve().parents[3]
CONTEXTS = ('reference-64-reset', 'full-clip')


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_bound(path, expected_hash, description):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f'{description} byte fingerprint changed')
    return json.loads(raw)


def _selected_payload(checkpoint_path, expected_sha256, model_name, hidden_size):
    if model_name not in {'frame', 'temporal'}:
        raise ValueError('Unknown head name')
    if _sha(checkpoint_path) != expected_sha256:
        raise ValueError('Selected checkpoint byte fingerprint changed')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if _sha(checkpoint_path) != expected_sha256:
        raise ValueError('Selected checkpoint changed during loading')
    if saved.get('schema_version') != 1:
        raise ValueError('Unknown selected checkpoint schema')
    config = ExperimentConfig.from_dict(saved['provenance']['config'])
    progress = saved['progress']
    if (config.model != model_name or config.hidden_size != hidden_size
            or progress['epoch'] != config.epochs or progress['cursor'] != 0
            or len(progress['history']) != config.epochs):
        raise ValueError('Selected checkpoint training is incomplete or has a different head')
    scores = []
    for epoch, row in enumerate(progress['history'], 1):
        validation = row['validation']
        score = validation['pce']
        if (row['epoch'] != epoch or validation.get('partition') != 'inner-validation-only'
                or validation.get('outer_test_evaluated') is not False
                or isinstance(score, bool) or not isinstance(score, (float, int))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Invalid selected checkpoint inner-validation history')
        scores.append(score)
    selected = scores.index(max(scores)) + 1
    if progress['best_epoch'] != selected or progress['best_validation_pce'] != max(scores):
        raise ValueError('Selected checkpoint is not earliest maximum inner-validation PCE')
    # Constructing a template must not advance the caller's global training RNG.
    with torch.random.fork_rng(devices=[]):
        model = FrameClassifier() if model_name == 'frame' else TemporalClassifier(hidden_size=hidden_size)
    template = model.state_dict()
    weights = saved.get('best_model')
    if (not isinstance(weights, dict) or set(weights) != set(template)
            or any(not isinstance(v, torch.Tensor) or v.shape != template[k].shape
                   or v.dtype != template[k].dtype or not torch.isfinite(v).all()
                   for k, v in weights.items())):
        raise ValueError('Selected model tensors are missing, nonfinite or malformed')
    model.load_state_dict(weights, strict=True)
    model.eval()
    model.requires_grad_(False)
    return saved, model


def load_selected_head(checkpoint_path, expected_sha256, model_name, hidden_size=64):
    """Read CPU best_model only; no optimizer/RNG restore, fitting or last-epoch fallback.

    This verifies checkpoint completion/selection and tensor integrity. Call
    validate_contract for full registration/data/source admission before research
    inference. This helper alone does not enable application model availability.
    """
    return _selected_payload(checkpoint_path, expected_sha256, model_name, hidden_size)[1]


def predict_probabilities(model, features: np.ndarray, context: str) -> np.ndarray:
    """CPU post-softmax [T,9]; reset context per chunk and retain every final frame."""
    if context not in CONTEXTS:
        raise ValueError('Unknown temporal context')
    if (not isinstance(features, np.ndarray) or features.dtype != np.float32
            or features.ndim != 2 or features.shape[1] != 1280 or len(features) < 1
            or not np.isfinite(features).all()):
        raise ValueError('Expected finite float32 full-clip [T,1280] features')
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Finalized evaluator requires CPU heads')
    model.eval()
    step = 64 if context == 'reference-64-reset' else len(features)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), step):
            block = torch.from_numpy(features[start:start + step].copy())[None]
            logits = model(block, torch.tensor([block.shape[1]], dtype=torch.int64))
            if logits.shape != (1, block.shape[1], 9) or not torch.isfinite(logits).all():
                raise ValueError('Head returned malformed/nonfinite logits')
            outputs.append(torch.softmax(logits[0], dim=-1).cpu().numpy())
    return np.concatenate(outputs)


def validate_contract(contract_path, annotations, manifest):
    """Validate immutable finalized inputs without opening any outer-test cache.

    Returns JSON contract/manifest, annotations, selected CPU heads and immutable
    fingerprints. Useful for a separately scoped latency benchmark on train media.
    It never constructs an optimizer or reads test predictions to choose a model.
    """
    contract_raw = Path(contract_path).read_bytes()
    contract = json.loads(contract_raw)
    required = {'schema_version', 'registration_path', 'registration_sha256', 'dataset_sha256',
                'manifest_sha256', 'dependency_lock_sha256', 'numpy_version', 'outer_fold',
                'expected_test_clips', 'feature_pipeline', 'heads', 'contexts', 'selection',
                'timestamp_provenance', 'order_constrained_decoding', 'device', 'torch_num_threads',
                'training_cache_catalog_path', 'training_cache_catalog_sha256'}
    if not isinstance(contract, dict) or not required <= set(contract):
        raise ValueError('Missing finalized evaluation contract fields')
    if (type(contract['schema_version']) is not int or contract['schema_version'] != 1
            or contract['contexts'] != list(CONTEXTS) or contract['selection'] != 'best_model'
            or contract['timestamp_provenance'] != 'reencoded-clip-pts'
            or contract['order_constrained_decoding'] is not False or contract['device'] != 'cpu'
            or type(contract['torch_num_threads']) is not int or contract['torch_num_threads'] != 1
            or contract['numpy_version'] != np.__version__
            or type(contract['outer_fold']) is not int or not 1 <= contract['outer_fold'] <= 4
            or type(contract['expected_test_clips']) is not int or contract['expected_test_clips'] < 1):
        raise ValueError('Unsupported finalized evaluation protocol/runtime')
    registration = _read_bound(contract['registration_path'], contract['registration_sha256'], 'Registration')
    dataset = load_annotations(annotations)
    manifest_value = _read_bound(manifest, contract['manifest_sha256'], 'Manifest')
    regenerated = make_split_manifest(dataset, manifest_value['outer_fold'], manifest_value['seed'],
                                     manifest_value['validation_fraction_sources'])
    if (manifest_value != regenerated or dataset.sha256 != contract['dataset_sha256']
            or dataset.sha256 != registration['dataset_sha256']
            or contract['manifest_sha256'] != registration['manifest_sha256']
            or contract['outer_fold'] != manifest_value['outer_fold']
            or contract['outer_fold'] != registration['outer_fold']
            or contract['expected_test_clips'] != len(manifest_value['partitions']['test']['clip_ids'])
            or contract['expected_test_clips'] != registration['outer_test_clips']):
        raise ValueError('Contract data/split differs from audited source-grouped registration')
    lock_hash = _sha(ROOT / 'uv.lock')
    if lock_hash != contract['dependency_lock_sha256']:
        raise ValueError('Dependency lock differs from finalized contract')
    pipeline = contract['feature_pipeline']
    if (pipeline != registration['feature_pipeline'] or pipeline.get('numpy') != np.__version__
            or pipeline.get('torch') != str(torch.__version__)
            or pipeline.get('backbone_class') != 'backend.swingsync.research.models.FrozenMobileNetV2'
            or pipeline.get('backbone_provenance', {}).get('weights') != 'IMAGENET1K_V2'
            or pipeline.get('transform') != DEFAULT_TRANSFORM or pipeline.get('extraction_device') != 'cpu'):
        raise ValueError('Contract differs from registered frozen feature pipeline/runtime')
    train_ids = set(manifest_value['partitions']['train']['clip_ids'])
    val_ids = set(manifest_value['partitions']['validation']['clip_ids'])
    test_ids = set(manifest_value['partitions']['test']['clip_ids'])
    selected_ids = train_ids | val_ids
    catalog = _read_bound(contract['training_cache_catalog_path'], contract['training_cache_catalog_sha256'],
                          'Training cache catalog')
    if (catalog.get('state') != 'completed' or catalog.get('outer_test_selected') is not False
            or catalog.get('model_identity') != pipeline
            or catalog.get('config', {}).get('partitions') != ['train', 'validation']
            or catalog['config'].get('manifest_sha256') != contract['manifest_sha256']
            or catalog['config'].get('annotations_sha256') != dataset.sha256
            or catalog.get('completed_clip_count') != len(selected_ids)
            or catalog.get('target_clip_count') != len(selected_ids)
            or set(catalog.get('completed', {})) != {str(i) for i in selected_ids}):
        raise ValueError('Original training catalog coverage/pipeline is not complete train+inner-validation')
    expected_fingerprints = {i: {'sha256': row['cache_sha256'], 'cache_key': row['cache_key']}
                             for i, row in catalog['completed'].items()}
    if set(contract['heads']) != {'frame', 'temporal'}:
        raise ValueError('Both fixed comparison heads are required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    heads = {}
    for name in ('frame', 'temporal'):
        head = contract['heads'][name]
        if type(head.get('selected_epoch')) is not int or head['selected_epoch'] < 1:
            raise ValueError('Selected epoch must be a positive integer')
        config = _read_bound(head['config_path'], head['config_sha256'], 'Training configuration')
        config_value = ExperimentConfig.from_dict(config)
        if config_value.model != name or head['config_sha256'] != registration['config_sha256'][name]:
            raise ValueError('Training configuration differs from registered head')
        run = _read_bound(head['run_record_path'], head['run_record_sha256'], 'Completed training ledger')
        final = run['checkpoints'][-1]
        expected_immutable = {'config_sha256': head['config_sha256'],
                              'registration_sha256': contract['registration_sha256'],
                              'cache_job_sha256': contract['training_cache_catalog_sha256'],
                              'manifest_sha256': contract['manifest_sha256'],
                              'dependency_lock_sha256': lock_hash,
                              'cli_sha256': _sha(ROOT / 'scripts/train_event_head.py')}
        if (run.get('state') != 'completed' or run.get('config') != config
                or run.get('immutable') != expected_immutable
                or final.get('sha256') != head['checkpoint_sha256']
                or Path(final['path']).resolve() != Path(head['checkpoint_path']).resolve()
                or run.get('outer_test_evaluated') is not False
                or run.get('validated_for_serving') is not False):
            raise ValueError('Completed training ledger/provenance differs from finalized head')
        saved, model = _selected_payload(head['checkpoint_path'], head['checkpoint_sha256'],
                                          name, config_value.hidden_size)
        p, progress = saved['provenance'], saved['progress']
        expected_steps = config_value.epochs * math.ceil(len(train_ids) / config_value.batch_size)
        if (p.get('config') != config or p.get('dataset_sha256') != dataset.sha256
                or p.get('manifest_sha256') != _json_sha(manifest_value)
                or p.get('cache_fingerprints') != expected_fingerprints
                or set(map(int, p['cache_fingerprints'])) & test_ids
                or p.get('feature_pipeline') != {**pipeline, 'timestamp_provenance': 'reencoded-clip-pts'}
                or p.get('torch') != str(torch.__version__) or p.get('numpy') != np.__version__
                or p.get('outer_test_evaluated') is not False
                or p.get('runtime', {}).get('torch_num_threads') != 1
                or p.get('runtime', {}).get('deterministic_algorithms') is not True
                or progress['best_epoch'] != head['selected_epoch']
                or progress['optimizer_steps'] != expected_steps
                or run['summary'].get('completed_epochs') != config_value.epochs
                or run['summary'].get('optimizer_steps') != expected_steps
                or run['summary'].get('history') != progress['history']
                or run['summary'].get('best_epoch') != progress['best_epoch']
                or run['summary'].get('best_validation_pce') != progress['best_validation_pce']):
            raise ValueError('Selected checkpoint data/pipeline/progress differs from completed registration')
        implementations = p.get('implementation_sha256', {})
        expected_modules = {'experiment.py', 'models.py', 'training.py', 'evaluation.py',
                            'features.py', 'golfdb.py', 'timeline.py'}
        if set(implementations) != expected_modules or any(
                _sha(Path(__file__).parent / key) != value for key, value in implementations.items()):
            raise ValueError('Training/evaluation implementation bytes changed since selected checkpoint')
        heads[name] = model
    immutable = {'contract_sha256': hashlib.sha256(contract_raw).hexdigest(),
                 'manifest_sha256': contract['manifest_sha256'], 'dataset_sha256': dataset.sha256,
                 'dependency_lock_sha256': lock_hash, 'evaluator_sha256': _sha(__file__),
                 'evaluation_cli_sha256': _sha(ROOT / 'scripts/evaluate_event_heads.py'),
                 'training_implementation_sha256': implementations,
                 'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__,
                             'python': platform.python_version(), 'machine': platform.machine(),
                             'platform': platform.platform(), 'torch_num_threads': torch.get_num_threads(),
                             'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}}
    return {'contract': contract, 'dataset': dataset, 'manifest': manifest_value,
            'heads': heads, 'immutable': immutable}



def _assert_run_integrity(immutable, contract_path):
    """Refuse publication after live source/config/runtime drift, including recovery."""
    expected_files = {Path(__file__): immutable['evaluator_sha256'],
                      ROOT / 'scripts/evaluate_event_heads.py': immutable['evaluation_cli_sha256'],
                      ROOT / 'uv.lock': immutable['dependency_lock_sha256'],
                      Path(contract_path): immutable['contract_sha256']}
    expected_files.update({Path(__file__).parent / name: value
                           for name, value in immutable['training_implementation_sha256'].items()})
    if any(_sha(path) != expected for path, expected in expected_files.items()):
        raise ValueError('Evaluation source/config bytes changed during invocation; publication refused')
    runtime = immutable['runtime']
    if (torch.get_num_threads() != runtime['torch_num_threads']
            or torch.are_deterministic_algorithms_enabled() != runtime['deterministic_algorithms']
            or str(torch.__version__) != runtime['torch'] or np.__version__ != runtime['numpy']):
        raise ValueError('Evaluation CPU runtime changed during invocation; publication refused')


def _cache_preflight(validated, cache_job_path):
    contract, dataset, manifest = (validated[k] for k in ('contract', 'dataset', 'manifest'))
    job_raw = Path(cache_job_path).read_bytes()
    job = json.loads(job_raw)
    ids = sorted(manifest['partitions']['test']['clip_ids'])
    config = job.get('config', {})
    if (job.get('state') != 'completed' or job.get('outer_test_selected') is not True
            or config.get('partitions') != ['test'] or config.get('allow_test') is not True
            or config.get('weights_request') != 'IMAGENET1K_V2'
            or config.get('annotations_sha256') != dataset.sha256
            or config.get('manifest_sha256') != contract['manifest_sha256']
            or config.get('dependency_lock_sha256') != contract['dependency_lock_sha256']
            or job.get('config_sha256') != _json_sha(config)
            or job.get('model_identity') != contract['feature_pipeline']
            or job.get('completed_clip_count') != len(ids) or job.get('target_clip_count') != len(ids)
            or set(job.get('completed', {})) != {str(i) for i in ids}):
        raise ValueError('Outer evaluation requires complete exact test-only cache job with registered pipeline')
    inventory = config.get('inventory', [])
    if (len(inventory) != len(ids) or {r['clip_id'] for r in inventory} != set(ids)
            or any(r.get('partition') != 'test' for r in inventory)):
        raise ValueError('Outer cache media inventory coverage differs from test partition')
    records = {r.clip_id: r for r in dataset.records}
    inventory_by_id = {r['clip_id']: r for r in inventory}
    paths, fingerprints = {}, {}
    for clip_id in ids:
        row = job['completed'][str(clip_id)]
        path = Path(row['cache_path'])
        video = Path(config['clips_dir']) / f'{clip_id}.mp4'
        record = records[clip_id]
        inv = inventory_by_id[clip_id]
        if (row.get('clip_id') != clip_id or row.get('source_id') != record.source_id
                or row.get('partition') != 'test' or inv.get('source_id') != record.source_id
                or _sha(path) != row['cache_sha256'] or _sha(video) != inv['video_sha256']):
            raise ValueError('Outer cache/media ledger provenance changed')
        clip = load_feature_cache(path)
        identity = clip.metadata['identity']
        expected_identity = {'annotation_sha256': dataset.sha256, 'clip_id': clip_id,
                             'source_id': record.source_id, 'outer_split': record.outer_split,
                             'video_sha256': inv['video_sha256'],
                             'original_events': list(record.original_events),
                             'annotation_event_indices': list(record.event_indices),
                             'historical_source_indices': [record.source_frame_index(i)
                                                           for i in range(record.expected_frame_count)],
                             'timestamp_provenance': contract['timestamp_provenance'],
                             **contract['feature_pipeline']}
        timeline = inspect_clip_timeline(video, record, origin='preprocessed')
        if (any(identity.get(k) != v for k, v in expected_identity.items())
                or clip.features.shape != (record.expected_frame_count, 1280)
                or row.get('frames') != record.expected_frame_count
                or row.get('cache_key') != clip.metadata['cache_key']
                or not np.array_equal(clip.pts_seconds, timeline.pts_seconds)
                or _sha(path) != row['cache_sha256'] or _sha(video) != inv['video_sha256']):
            raise ValueError('Outer cache annotation/source/PTS/pipeline differs from audited media')
        paths[clip_id] = path
        fingerprints[str(clip_id)] = {'cache_sha256': row['cache_sha256'], 'cache_key': row['cache_key'],
                                     'video_sha256': inv['video_sha256']}
    return paths, fingerprints, hashlib.sha256(job_raw).hexdigest()


def _row_hash(row):
    return _json_sha({k: v for k, v in row.items() if k != 'sha256'})


def _validate_completed_row(row, clip_id, expected_fingerprint, record, clip):
    if (row.get('sha256') != _row_hash(row) or row.get('clip_id') != clip_id
            or row.get('source_id') != record.source_id or row.get('inputs') != expected_fingerprint
            or set(row.get('results', {})) != {'frame', 'temporal'}):
        raise ValueError('Persisted outer evaluation row differs from immutable inputs/checksum')
    for head in ('frame', 'temporal'):
        if set(row['results'][head]) != set(CONTEXTS):
            raise ValueError('Persisted outer evaluation contexts differ')
        for context in CONTEXTS:
            result = row['results'][head][context]
            predictions = result.get('predicted_frames', [])
            if (len(predictions) != 8 or any(type(i) is not int or not 0 <= i < len(clip.labels)
                                              for i in predictions)):
                raise ValueError('Persisted outer predictions are invalid')
            # Validate every stored metric from its predictions and immutable PTS;
            # no model rerun is needed for prior durably recorded clips.
            scores = np.zeros((len(clip.labels), 9))
            for event, frame in enumerate(predictions):
                scores[frame, event] = .1
            scores[:, 8] = 1 - scores[:, :8].sum(axis=1)
            expected = evaluate_clip(scores, record.event_indices,
                                     timestamps_seconds=clip.pts_seconds).to_dict()
            if json.loads(json.dumps(expected)) != result:
                raise ValueError('Persisted outer metrics differ from prediction/timestamp evidence')


def _summary(rows):
    result = {}
    for head in ('frame', 'temporal'):
        result[head] = {}
        for context in CONTEXTS:
            # Recreate the dataclass from hash-validated stored per-clip evidence.
            from .evaluation import ClipEvaluation
            clips = []
            for row in rows:
                value = dict(row['results'][head][context])
                value.pop('pce')
                value.pop('pce_percent')
                clips.append(ClipEvaluation(**value))
            metrics = summarize_evaluations(clips)
            metrics.update(partition='outer-test', temporal_context=context,
                           outer_test_evaluated=True, validated_for_serving=False)
            result[head][context] = metrics
    return result


def run_heldout_evaluation(contract_path, annotations, manifest, cache_job_path, run_dir, *,
                           resume=False, max_new_clips=None):
    """Persist all four fixed comparisons per full clip; no partial aggregate score.

    A whole test inventory must exist first. Explicit resume checks all prior
    inputs and result checksums/metric arithmetic, and skips completed inference.
    Per-clip writes are atomic in one ledger; an interrupted uncommitted head call
    may repeat, but frozen extraction and durably completed clip work do not.
    """
    if max_new_clips is not None and (type(max_new_clips) is not int or max_new_clips < 1):
        raise ValueError('max_new_clips must be a positive integer or None')
    DEFAULT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_LOCK_PATH.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another feature/training/evaluation job owns the shared lock') from exc
        root = Path(run_dir).resolve()
        state_path = root / 'run.json'
        existing = None
        if root.exists():
            if not resume:
                raise ValueError('Evaluation run exists; inspect and explicitly resume')
            existing = json.loads(state_path.read_text())
            _check_previous_process(existing)
        elif resume:
            raise ValueError('Evaluation run does not exist')
        validated = validate_contract(contract_path, annotations, manifest)
        paths, fingerprints, catalog_hash = _cache_preflight(validated, cache_job_path)
        immutable = {**validated['immutable'], 'test_cache_job_sha256': catalog_hash,
                     'test_cache_fingerprints': fingerprints}
        contract = validated['contract']
        ids = sorted(paths)
        records = {r.clip_id: r for r in validated['dataset'].records}
        if existing is None:
            root.mkdir(parents=True, exist_ok=False)
            state = {'schema_version': 1, 'state': 'initializing', 'created_at': _now(),
                     'immutable': immutable, 'git_base_commit': _git_base_commit(),
                     'completed': {}, 'attempts': [], 'summary': None,
                     'completed_clip_count': 0, 'target_clip_count': len(ids),
                     'outer_test_evaluated': False, 'validated_for_serving': False,
                     'evidence_type': 'project-trained-head-frozen-outer-test-comparison'}
        else:
            state = existing
            if (state.get('schema_version') != 1 or state.get('immutable') != immutable
                    or state.get('state') not in {'running', 'initializing', 'partial', 'completed',
                                                'interrupted', 'failed'}
                    or not isinstance(state.get('completed'), dict)
                    or not set(state['completed']) <= {str(i) for i in ids}):
                raise ValueError('Evaluation provenance/coverage changed; do not restart or retune')
        for key, row in state['completed'].items():
            i = int(key)
            clip = load_feature_cache(paths[i])
            _validate_completed_row(row, i, fingerprints[key], records[i], clip)
        _assert_run_integrity(immutable, contract_path)
        if len(state['completed']) == len(ids):
            summary = _summary([state['completed'][str(i)] for i in ids])
            if state.get('state') == 'completed':
                if state.get('summary') != summary:
                    raise ValueError('Completed evaluation summary differs from durable evidence')
                return state
            # A crash may occur after publishing the final clip but before the
            # aggregate/completed transition. All inputs/row metrics above have
            # already been revalidated; adopt them without repeating inference.
            recovered_at = _now()
            if state['attempts'] and state['attempts'][-1].get('state') is None:
                state['attempts'][-1].update(state='interrupted', ended_at=recovered_at,
                                             recovery_reason='Owner ended before final summary publication')
            state['attempts'].append({'state': 'completed', 'started_at': recovered_at,
                                      'ended_at': recovered_at, 'new_clips': 0,
                                      'pid': os.getpid(), 'process_start_identity': _process_start(os.getpid()),
                                      'recovery': 'Final durable clip adopted; no inference repeated'})
            _assert_run_integrity(immutable, contract_path)
            state.update(state='completed', summary=summary, updated_at=recovered_at,
                         outer_test_evaluated=True, error=None)
            _publish_json(state_path, state)
            return state
        if state.get('state') == 'completed':
            raise ValueError('Completed evaluation is missing durable clips')
        attempt = {'pid': os.getpid(), 'process_start_identity': _process_start(os.getpid()),
                   'started_at': _now(), 'max_new_clips': max_new_clips, 'new_clips': 0}
        state['attempts'].append(attempt)
        state.update(state='running', pid=attempt['pid'], process_start_identity=attempt['process_start_identity'],
                     summary=None, error=None, updated_at=_now())
        _publish_json(state_path, state)
        started = time.monotonic()
        try:
            for clip_id in ids:
                if str(clip_id) in state['completed']:
                    continue
                if max_new_clips is not None and attempt['new_clips'] >= max_new_clips:
                    break
                fingerprint = fingerprints[str(clip_id)]
                if _sha(paths[clip_id]) != fingerprint['cache_sha256']:
                    raise ValueError('Outer cache changed after preflight')
                clip = load_feature_cache(paths[clip_id])
                row = {'clip_id': clip_id, 'source_id': records[clip_id].source_id,
                       'inputs': fingerprint, 'results': {}}
                for head, model in validated['heads'].items():
                    row['results'][head] = {}
                    for context in contract['contexts']:
                        probabilities = predict_probabilities(model, clip.features, context)
                        row['results'][head][context] = evaluate_clip(
                            probabilities, records[clip_id].event_indices,
                            timestamps_seconds=clip.pts_seconds).to_dict()
                # Normalize tuples to JSON now, so first run and resumed rows are identical.
                row = json.loads(json.dumps(row, allow_nan=False))
                row['sha256'] = _row_hash(row)
                _validate_completed_row(row, clip_id, fingerprint, records[clip_id], clip)
                if _sha(paths[clip_id]) != fingerprint['cache_sha256']:
                    raise ValueError('Outer cache changed during inference')
                _assert_run_integrity(immutable, contract_path)
                state['completed'][str(clip_id)] = row
                state.update(completed_clip_count=len(state['completed']), target_clip_count=len(ids),
                             outer_test_evaluated=True, updated_at=_now())
                attempt['new_clips'] += 1
                _publish_json(state_path, state)
            complete = len(state['completed']) == len(ids)
            _assert_run_integrity(immutable, contract_path)
            state.update(state='completed' if complete else 'partial',
                         summary=_summary([state['completed'][str(i)] for i in ids]) if complete else None,
                         updated_at=_now())
            attempt.update(state=state['state'], ended_at=_now(), elapsed_seconds=time.monotonic() - started)
            _publish_json(state_path, state)
            return state
        except BaseException as exc:
            state.update(state='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                         error=str(exc), updated_at=_now())
            attempt.update(state=state['state'], ended_at=_now(), elapsed_seconds=time.monotonic() - started)
            _publish_json(state_path, state)
            raise
