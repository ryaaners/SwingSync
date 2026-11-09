"""Offline, standard-library-only packaging of a completed fixed event run.

Public API: ``build_event_report(run_path, expected_run_sha256, contract_path,
manifest_path, *, include_order_diagnostics=False)`` returns a deterministic JSON
value after validation. ``write_event_report(report, output_path)`` publishes it
atomically and exclusively. Neither function loads models, caches or media.
The caller must supply a separately obtained expected run byte fingerprint.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


HEADS = ('frame', 'temporal')
CONTEXTS = ('reference-64-reset', 'full-clip')
EVENT_NAMES = ('Address', 'Toe-up', 'Mid-backswing', 'Top', 'Mid-downswing', 'Impact',
               'Mid-follow-through', 'Finish')
PROTOCOL = 'golfdb-independent-events-v1'
TIE_POLICY = 'numpy-default-argsort-last-index'
TARGET_MAPPING = 'chronological occupied frames, matching reference np.where(labels < 8)'
ABSOLUTE_TOLERANCE = 1e-12


def _json_sha(value):
    # Deliberately identical to heldout._json_sha/_row_hash, without importing Torch.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _read(path, expected_hash=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None:
        _require(_hash(expected_hash) and digest == expected_hash, 'Input byte SHA256 differs')
    value = json.loads(raw, object_pairs_hook=_unique_object,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f'Nonfinite JSON constant: {token}')))
    _require(isinstance(value, dict), 'Expected a JSON object')
    return value, digest


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same(actual, expected, path='summary'):
    """Compare every recorded field, preserving the recorded floats on output."""
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), f'{path}: fields differ')
        for key, value in expected.items():
            _same(actual[key], value, f'{path}.{key}')
    elif type(expected) is float:
        _require(_number(actual) and math.isclose(actual, expected, rel_tol=0,
                                                abs_tol=ABSOLUTE_TOLERANCE),
                 f'{path}: aggregate/arithmetic differs')
    else:
        _require(type(actual) is type(expected) and actual == expected, f'{path}: value differs')


def _manifest_sources(manifest, contract):
    _require(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1
             and type(manifest.get('outer_fold')) is int and manifest['outer_fold'] == 1,
             'Only registered outer fold 1 is supported')
    _require(manifest.get('dataset_sha256') == contract['dataset_sha256'],
             'Manifest dataset differs')
    parts = manifest.get('partitions', {})
    _require(set(parts) == {'train', 'validation', 'test'}, 'Manifest partitions differ')
    all_ids, all_sources, test_sources = set(), set(), None
    for name in ('train', 'validation', 'test'):
        part = parts[name]
        ids, sources, clips = part['clip_ids'], part['source_ids'], part['clips']
        _require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids)
                 and len(set(ids)) == len(ids), 'Manifest clip IDs must be unique integers')
        _require(isinstance(sources, list) and sources
                 and all(isinstance(s, str) and s for s in sources)
                 and len(set(sources)) == len(sources), 'Manifest sources must be unique strings')
        _require(isinstance(clips, list) and len(clips) == len(ids)
                 and all(isinstance(row, dict) and set(row) == {'clip_id', 'source_id'}
                         and type(row['clip_id']) is int and isinstance(row['source_id'], str)
                         for row in clips), 'Manifest groups are malformed')
        mapping = {row['clip_id']: row['source_id'] for row in clips}
        _require(set(mapping) == set(ids) and set(mapping.values()) == set(sources),
                 'Manifest clip/source groups differ from partition lists')
        _require(not all_ids.intersection(ids) and not all_sources.intersection(sources),
                 'Manifest clip/source partitions overlap')
        all_ids.update(ids)
        all_sources.update(sources)
        if name == 'test':
            test_sources = mapping
    _require(len(test_sources) == contract['expected_test_clips'], 'Expected test coverage differs')
    return test_sources


def _result(value, numpy_version):
    required = {'annotation_event_frames', 'annotation_order_anomaly', 'event_frames',
                'predicted_frames', 'signed_error_frames', 'absolute_error_frames', 'tolerance_frames',
                'correct', 'signed_error_ms', 'absolute_error_ms', 'valid_frame_count', 'numpy_version',
                'protocol', 'tie_policy', 'pce', 'pce_percent'}
    _require(isinstance(value, dict) and set(value) == required, 'Per-clip result fields differ')
    count = value['valid_frame_count']
    _require(type(count) is int and count >= 8, 'Invalid full-clip frame count')
    for key in ('annotation_event_frames', 'event_frames', 'predicted_frames',
                'signed_error_frames', 'absolute_error_frames'):
        vector = value[key]
        _require(isinstance(vector, list) and len(vector) == 8
                 and all(type(i) is int for i in vector), f'{key}: expected eight integers')
    annotations, events, predictions = (value[k] for k in
                                       ('annotation_event_frames', 'event_frames', 'predicted_frames'))
    _require(len(set(annotations)) == 8 and all(0 <= i < count for i in annotations + predictions)
             and events == sorted(annotations), 'Annotation/chronological target/prediction mismatch')
    _require(type(value['annotation_order_anomaly']) is bool
             and value['annotation_order_anomaly'] == (annotations != events),
             'Annotation order anomaly flag differs')
    signed = [p - e for p, e in zip(predictions, events)]
    absolute = [abs(i) for i in signed]
    tolerance = max(round((events[5] - events[0]) / 30), 1)
    _require(value['signed_error_frames'] == signed and value['absolute_error_frames'] == absolute
             and type(value['tolerance_frames']) is int and value['tolerance_frames'] == tolerance,
             'Frame error/tolerance arithmetic differs')
    correct = value['correct']
    _require(isinstance(correct, list) and len(correct) == 8 and all(type(b) is bool for b in correct)
             and correct == [i <= tolerance for i in absolute], 'Correct-event flags differ')
    for key in ('signed_error_ms', 'absolute_error_ms'):
        vector = value[key]
        _require(isinstance(vector, list) and len(vector) == 8 and all(_number(i) for i in vector),
                 f'{key}: expected eight finite PTS errors; partial timing is forbidden')
    for frame, ms, absolute_ms in zip(signed, value['signed_error_ms'], value['absolute_error_ms']):
        _require(absolute_ms == abs(ms) and ((frame == 0 and ms == 0)
                 or (frame > 0 and ms > 0) or (frame < 0 and ms < 0)),
                 'PTS error sign/absolute consistency differs')
    _require(value['protocol'] == PROTOCOL and value['tie_policy'] == TIE_POLICY
             and value['numpy_version'] == numpy_version, 'Per-clip protocol/runtime differs')
    pce = sum(correct) / 8
    _same(value['pce'], pce, 'per-clip.pce')
    _same(value['pce_percent'], pce * 100, 'per-clip.pce_percent')


def _summary(results, context, numpy_version):
    count = len(results)
    per_event = {}
    for event, name in enumerate(EVENT_NAMES):
        pce = sum(r['correct'][event] for r in results) / count
        per_event[name] = {
            'pce': pce, 'pce_percent': pce * 100,
            'mae_frames': sum(r['absolute_error_frames'][event] for r in results) / count,
            'mae_ms': math.fsum(r['absolute_error_ms'][event] for r in results) / count,
        }
    pce = sum(sum(r['correct']) for r in results) / (count * 8)
    return {
        'protocol': PROTOCOL, 'tie_policy': TIE_POLICY, 'numpy_version': numpy_version,
        'clip_count': count, 'pce': pce, 'pce_percent': pce * 100,
        'mae_frames': sum(sum(r['absolute_error_frames']) for r in results) / (count * 8),
        'mae_ms': math.fsum(v for r in results for v in r['absolute_error_ms']) / (count * 8),
        'per_event': per_event, 'order_constrained_decoding': False,
        'annotation_order_anomaly_count': sum(r['annotation_order_anomaly'] for r in results),
        'target_mapping': TARGET_MAPPING, 'partition': 'outer-test', 'temporal_context': context,
        'outer_test_evaluated': True, 'validated_for_serving': False,
    }


def _diagnostics(results):
    predictions = [r['predicted_frames'] for r in results]
    strict = sum(all(a < b for a, b in zip(p, p[1:])) for p in predictions)
    return {'strictly_chronological_clip_count': strict,
            'not_strictly_chronological_clip_count': len(results) - strict,
            'duplicate_prediction_clip_count': sum(len(set(p)) < 8 for p in predictions),
            'duplicate_prediction_excess_event_count': sum(8 - len(set(p)) for p in predictions)}


def build_event_report(run_path, expected_run_sha256, contract_path, manifest_path, *,
                       include_order_diagnostics=False):
    """Validate a hash-bound completed run and return its compact reproducibility report.

    The expected hash is mandatory. All four fixed comparisons and exact contract
    test coverage must be present. Actual PTS arithmetic requires a separate cache
    audit; this offline check never reconstructs milliseconds from frame rate.
    """
    _require(_hash(expected_run_sha256), 'Explicit expected run SHA256 is required')
    run, run_sha = _read(run_path, expected_run_sha256)
    contract, contract_sha = _read(contract_path)
    manifest, manifest_sha = _read(manifest_path)
    _require(type(contract.get('schema_version')) is int and contract['schema_version'] == 1
             and type(contract.get('outer_fold')) is int and contract['outer_fold'] == 1
             and type(contract.get('expected_test_clips')) is int and contract['expected_test_clips'] > 0
             and contract.get('contexts') == list(CONTEXTS) and contract.get('selection') == 'best_model'
             and contract.get('timestamp_provenance') == 'reencoded-clip-pts'
             and contract.get('order_constrained_decoding') is False
             and contract.get('device') == 'cpu' and type(contract.get('torch_num_threads')) is int
             and contract['torch_num_threads'] == 1,
             'Unsupported fixed fold-1 contract')
    _require(set(contract.get('heads', {})) == set(HEADS), 'Fixed heads differ')
    for head in HEADS:
        identity = contract['heads'][head]
        _require(_hash(identity.get('checkpoint_sha256')) and identity.get('checkpoint_path')
                 and type(identity.get('selected_epoch')) is int and identity['selected_epoch'] > 0,
                 'Checkpoint identity/selection missing')
    immutable = run.get('immutable', {})
    _require(immutable.get('contract_sha256') == contract_sha
             and immutable.get('manifest_sha256') == manifest_sha == contract['manifest_sha256']
             and immutable.get('dataset_sha256') == contract['dataset_sha256']
             and immutable.get('dependency_lock_sha256') == contract['dependency_lock_sha256'],
             'Run/contract/manifest provenance binding differs')
    sources = _manifest_sources(manifest, contract)
    ids = sorted(sources)
    _require(type(run.get('schema_version')) is int and run['schema_version'] == 1
             and run.get('state') == 'completed' and run.get('outer_test_evaluated') is True
             and run.get('validated_for_serving') is False
             and type(run.get('completed_clip_count')) is int and run['completed_clip_count'] == len(ids)
             and type(run.get('target_clip_count')) is int and run['target_clip_count'] == len(ids),
             'Only completed exact-coverage evaluation can be reported')
    rows, fingerprints = run.get('completed', {}), immutable.get('test_cache_fingerprints', {})
    expected_keys = {str(i) for i in ids}
    _require(isinstance(rows, dict) and set(rows) == expected_keys
             and isinstance(fingerprints, dict) and set(fingerprints) == expected_keys,
             'Run rows/cache fingerprint coverage differs')
    for i in ids:
        row = rows[str(i)]
        fingerprint = fingerprints[str(i)]
        _require(isinstance(fingerprint, dict)
                 and set(fingerprint) == {'cache_sha256', 'cache_key', 'video_sha256'}
                 and all(_hash(v) for v in fingerprint.values()), 'Invalid immutable input fingerprint')
        _require(isinstance(row, dict) and row.get('sha256') == _json_sha(
            {k: v for k, v in row.items() if k != 'sha256'})
            and type(row.get('clip_id')) is int and row['clip_id'] == i
            and row.get('source_id') == sources[i] and row.get('inputs') == fingerprint
            and isinstance(row.get('results'), dict) and set(row['results']) == set(HEADS),
            'Row checksum/clip/source/input identity differs')
        for head in HEADS:
            _require(isinstance(row['results'][head], dict)
                     and set(row['results'][head]) == set(CONTEXTS), 'Fixed context coverage differs')
            for context in CONTEXTS:
                _result(row['results'][head][context], contract['numpy_version'])
        reference = row['results']['frame'][CONTEXTS[0]]
        for head in HEADS:
            for context in CONTEXTS:
                result = row['results'][head][context]
                for key in ('annotation_event_frames', 'event_frames', 'valid_frame_count'):
                    _require(result[key] == reference[key], 'Heads/contexts use different clip targets')
    calculated, diagnostics = {}, {}
    for head in HEADS:
        calculated[head], diagnostics[head] = {}, {}
        for context in CONTEXTS:
            results = [rows[str(i)]['results'][head][context] for i in ids]
            calculated[head][context] = _summary(results, context, contract['numpy_version'])
            diagnostics[head][context] = _diagnostics(results)
    _same(run.get('summary'), calculated)
    provenance = {k: v for k, v in immutable.items() if k != 'test_cache_fingerprints'}
    provenance.update(test_cache_fingerprints_sha256=_json_sha(fingerprints),
                      test_cache_fingerprints_count=len(fingerprints))
    report = {
        'schema_version': 1, 'evidence_type': run.get('evidence_type'),
        'scope': 'Fixed official GolfDB outer fold 1 only; project-trained selected heads',
        'input_sha256': {'run': run_sha, 'contract': contract_sha, 'manifest': manifest_sha},
        'contract': contract, 'run_immutable_provenance': provenance,
        'run_record': {k: run[k] for k in ('git_base_commit', 'created_at', 'updated_at', 'attempts')
                       if k in run},
        'coverage': {'test_clip_count': len(ids), 'test_clip_ids': ids,
                     'test_frame_count': sum(rows[str(i)]['results']['frame'][CONTEXTS[0]]
                                             ['valid_frame_count'] for i in ids),
                     'test_source_count': len(set(sources.values())),
                     'test_source_ids': sorted(set(sources.values())),
                     'source_disjoint_partitions_verified': True,
                     'completed_rows_sha256': _json_sha(rows),
                     'annotation_order_anomaly_clip_ids': [i for i in ids if
                         rows[str(i)]['results']['frame'][CONTEXTS[0]]['annotation_order_anomaly']]},
        'primary_metrics': {'name': 'official independent-event PCE',
                            'unit': 'fraction; pce_percent also retained',
                            'protocol': PROTOCOL, 'tie_policy': TIE_POLICY,
                            'comparison': 'temporal minus frame, separately in each fixed context',
                            'temporal_context_definitions': {
                                'reference-64-reset': 'Nonoverlapping 64-frame calls from frame zero; '
                                'fresh hidden state each call; final tail unpadded; concatenate '
                                'post-softmax chunks before full-clip evaluation',
                                'full-clip': 'Separately disclosed single full-clip recurrence with '
                                'fresh hidden state for each clip',
                            },
                            'tolerance_frames': 'max(round((chronological_target[5] - '
                            'chronological_target[0]) / 30), 1); nearest integer, ties to even',
                            'target_mapping': TARGET_MAPPING, 'event_slot_names': list(EVENT_NAMES),
                            'summary_check_absolute_tolerance': ABSOLUTE_TOLERANCE},
        # Retain the evaluator's precise recorded floats after recomputation.
        'summary': run['summary'],
        'head_comparison': {context: {
            'pce_delta_temporal_minus_frame': run['summary']['temporal'][context]['pce']
                - run['summary']['frame'][context]['pce'],
            'percentage_point_delta_temporal_minus_frame': run['summary']['temporal'][context]['pce_percent']
                - run['summary']['frame'][context]['pce_percent'],
        } for context in CONTEXTS},
        'validated_for_serving': False,
        'verification_limits': [
            'Offline artifact arithmetic and fingerprint checks only; no model, checkpoint, media or cache loading.',
            'Recorded PTS millisecond errors checked for finite values, sign and absolute consistency only; '
            'actual PTS differences require an independent cache/media audit.',
            'Predictions are independent event peaks; no order-constrained decoding or test-driven selection.',
            'Chronological occupied-frame targets preserve annotation anomalies; event slots need not equal '
            'semantic annotation labels for anomalous clips.',
            'Reencoded clip PTS do not establish original source timestamps or browser synchronization.',
            'One outer fold does not establish four-fold performance, unseen-golfer generalization, '
            'uploaded-video serving, pose reliability, coaching efficacy or advice validation.',
            'Hash binding verifies supplied artifacts, not the truth of their externally recorded provenance.',
        ],
    }
    if include_order_diagnostics:
        report['posthoc_order_diagnostics'] = {
            'scope': 'Descriptive saved-prediction counts only; no decoding, improvement or model selection',
            'strict_order_definition': 'All seven adjacent predicted event-frame differences are positive',
            'duplicate_excess_definition': 'Sum of 8 minus unique predicted-frame count per clip',
            'counts': diagnostics,
        }
    return report


def write_event_report(report, output_path):
    """Atomically publish JSON via a same-directory hard link; never replace a file.

    The entire temporary file is flushed before link creation. Existing paths,
    including symlinks, fail exclusively. Failed writes clean up the temporary.
    """
    output = Path(output_path)
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f'.{output.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
