"""Synthetic saved evidence tests packaging only, never model performance."""
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import sys

import pytest

from backend.swingsync.research.event_report import (
    CONTEXTS, EVENT_NAMES, PROTOCOL, TARGET_MAPPING, TIE_POLICY,
    build_event_report, write_event_report,
)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip_result(predictions, anomaly=False):
    events = list(range(0, 16, 2))
    annotation = events.copy()
    if anomaly:
        annotation[0], annotation[1] = annotation[1], annotation[0]
    signed = [p - e for p, e in zip(predictions, events)]
    correct = [abs(e) <= 1 for e in signed]
    return {'annotation_event_frames': annotation, 'annotation_order_anomaly': anomaly,
            'event_frames': events, 'predicted_frames': predictions,
            'signed_error_frames': signed, 'absolute_error_frames': [abs(e) for e in signed],
            'tolerance_frames': 1, 'correct': correct, 'signed_error_ms': [e * 13.25 for e in signed],
            'absolute_error_ms': [abs(e) * 13.25 for e in signed], 'valid_frame_count': 24,
            'numpy_version': 'synthetic-version', 'protocol': PROTOCOL, 'tie_policy': TIE_POLICY,
            'pce': sum(correct) / 8, 'pce_percent': sum(correct) / 8 * 100}


def expected_summary(head, context):
    # Independently specified expected arithmetic for the two synthetic rows.
    perfect = head == 'temporal'
    pce = 1.0 if perfect else 13 / 16
    per_event = {name: {'pce': 1.0, 'pce_percent': 100.0, 'mae_frames': 0.0, 'mae_ms': 0.0}
                 for name in EVENT_NAMES}
    if not perfect:
        per_event['Mid-follow-through'] = {
            'pce': .5, 'pce_percent': 50.0, 'mae_frames': 1.0, 'mae_ms': 13.25}
        per_event['Finish'] = {'pce': 0.0, 'pce_percent': 0.0, 'mae_frames': 2.0, 'mae_ms': 26.5}
    return {'protocol': PROTOCOL, 'tie_policy': TIE_POLICY, 'numpy_version': 'synthetic-version',
            'clip_count': 2, 'pce': pce, 'pce_percent': pce * 100,
            'mae_frames': 0.0 if perfect else .375, 'mae_ms': 0.0 if perfect else 4.96875,
            'per_event': per_event, 'order_constrained_decoding': False,
            'annotation_order_anomaly_count': 1, 'target_mapping': TARGET_MAPPING,
            'partition': 'outer-test', 'temporal_context': context,
            'outer_test_evaluated': True, 'validated_for_serving': False}


@pytest.fixture
def artifacts(tmp_path):
    def partition(ids, source):
        return {'clip_ids': ids, 'source_ids': [source],
                'clips': [{'clip_id': i, 'source_id': source} for i in ids]}
    manifest = {'schema_version': 1, 'outer_fold': 1, 'dataset_sha256': 'd' * 64,
                'partitions': {'train': partition([1], 'train-source'),
                               'validation': partition([2], 'validation-source'),
                               'test': partition([10, 11], 'test-source')}}
    manifest_path, contract_path, run_path = (tmp_path / f'{name}.json'
                                              for name in ('manifest', 'contract', 'run'))
    manifest_sha = dump(manifest_path, manifest)
    contract = {'schema_version': 1, 'outer_fold': 1, 'expected_test_clips': 2,
                'contexts': list(CONTEXTS), 'selection': 'best_model',
                'timestamp_provenance': 'reencoded-clip-pts', 'order_constrained_decoding': False,
                'device': 'cpu', 'torch_num_threads': 1, 'dataset_sha256': 'd' * 64,
                'dependency_lock_sha256': 'e' * 64, 'manifest_sha256': manifest_sha,
                'numpy_version': 'synthetic-version',
                'evaluation_policy': 'Synthetic fixture; no model results',
                'heads': {head: {'checkpoint_path': f'synthetic/{head}.pt',
                                  'checkpoint_sha256': c * 64, 'selected_epoch': epoch}
                          for head, c, epoch in (('frame', 'a', 19), ('temporal', 'b', 20))}}
    contract_sha = dump(contract_path, contract)
    fingerprints = {str(i): {'cache_sha256': '1' * 64, 'cache_key': '2' * 64, 'video_sha256': '3' * 64}
                    for i in (10, 11)}
    run = {'schema_version': 1, 'state': 'completed', 'completed_clip_count': 2, 'target_clip_count': 2,
           'outer_test_evaluated': True, 'validated_for_serving': False,
           'evidence_type': 'synthetic-software-test-only', 'git_base_commit': 'synthetic',
           'immutable': {'contract_sha256': contract_sha, 'manifest_sha256': manifest_sha,
                         'dataset_sha256': 'd' * 64, 'dependency_lock_sha256': 'e' * 64,
                         'test_cache_job_sha256': 'f' * 64, 'test_cache_fingerprints': fingerprints},
           'completed': {},
           'summary': {head: {c: expected_summary(head, c) for c in CONTEXTS}
                       for head in ('frame', 'temporal')}}
    for i in (10, 11):
        frame_predictions = ([0, 2, 4, 6, 8, 10, 14, 12] if i == 10
                             else [0, 2, 4, 6, 8, 10, 12, 12])
        row = {'clip_id': i, 'source_id': 'test-source', 'inputs': fingerprints[str(i)],
               'results': {head: {c: clip_result(frame_predictions if head == 'frame'
                                                else list(range(0, 16, 2)), i == 11)
                                  for c in CONTEXTS} for head in ('frame', 'temporal')}}
        row['sha256'] = canonical_sha(row)
        run['completed'][str(i)] = row
    dump(run_path, run)
    return run_path, contract_path, manifest_path, run, contract, manifest


def build(artifacts, *, diagnostics=False):
    run_path, contract_path, manifest_path = artifacts[:3]
    return build_event_report(run_path, hashlib.sha256(run_path.read_bytes()).hexdigest(),
                              contract_path, manifest_path, include_order_diagnostics=diagnostics)


def resave(artifacts, *, rows=False, bindings=False):
    run_path, contract_path, manifest_path, run, contract, manifest = artifacts
    if rows:
        for row in run['completed'].values():
            row['sha256'] = canonical_sha({k: v for k, v in row.items() if k != 'sha256'})
    if bindings:
        contract['manifest_sha256'] = dump(manifest_path, manifest)
        run['immutable']['manifest_sha256'] = contract['manifest_sha256']
        run['immutable']['contract_sha256'] = dump(contract_path, contract)
    dump(run_path, run)


def test_complete_report_preserves_metrics_and_provenance(artifacts):
    report = build(artifacts)
    assert report == build(artifacts)
    assert report['summary'] == artifacts[3]['summary']
    assert report['contract'] == artifacts[4]
    assert report['coverage']['test_clip_count'] == 2
    assert report['coverage']['test_frame_count'] == 48
    assert report['coverage']['test_source_count'] == 1
    assert report['coverage']['annotation_order_anomaly_clip_ids'] == [11]
    assert report['head_comparison']['full-clip']['pce_delta_temporal_minus_frame'] == .1875
    assert report['head_comparison']['full-clip']['percentage_point_delta_temporal_minus_frame'] == 18.75
    assert report['validated_for_serving'] is False
    assert 'posthoc_order_diagnostics' not in report
    assert 'test_cache_fingerprints' not in report['run_immutable_provenance']
    assert report['run_immutable_provenance']['test_cache_fingerprints_sha256'] == canonical_sha(
        artifacts[3]['immutable']['test_cache_fingerprints'])


def test_optional_order_diagnostics_count_duplicates_and_reversals(artifacts):
    diagnostic = build(artifacts, diagnostics=True)['posthoc_order_diagnostics']
    assert 'no decoding, improvement or model selection' in diagnostic['scope']
    frame = diagnostic['counts']['frame']['reference-64-reset']
    assert frame == {'strictly_chronological_clip_count': 0, 'not_strictly_chronological_clip_count': 2,
                     'duplicate_prediction_clip_count': 1, 'duplicate_prediction_excess_event_count': 1}
    assert diagnostic['counts']['temporal']['full-clip']['strictly_chronological_clip_count'] == 2


@pytest.mark.parametrize('state', ['running', 'partial', 'failed', 'interrupted'])
def test_partial_or_failed_runs_never_publish_aggregates(artifacts, state):
    artifacts[3]['state'] = state
    resave(artifacts)
    with pytest.raises(ValueError, match='completed exact-coverage'):
        build(artifacts)


@pytest.mark.parametrize('mutation', ['missing-row', 'extra-row', 'wrong-id', 'missing-head',
                                      'missing-context', 'fingerprint', 'source'])
def test_exact_coverage_and_identity_required(artifacts, mutation):
    run = artifacts[3]
    row = run['completed']['10']
    if mutation == 'missing-row':
        del run['completed']['11']
    elif mutation == 'extra-row':
        run['completed']['12'] = deepcopy(row)
    elif mutation == 'wrong-id':
        row['clip_id'] = 12
    elif mutation == 'missing-head':
        del row['results']['temporal']
    elif mutation == 'missing-context':
        del row['results']['frame']['full-clip']
    elif mutation == 'fingerprint':
        row['inputs'] = {**row['inputs'], 'cache_sha256': '0' * 64}
    else:
        row['source_id'] = 'train-source'
    resave(artifacts, rows=True)
    with pytest.raises(ValueError):
        build(artifacts)


def test_expected_run_sha_is_explicit_and_byte_bound(artifacts):
    run_path, contract_path, manifest_path = artifacts[:3]
    for bad in (None, '', '0' * 64):
        with pytest.raises(ValueError):
            build_event_report(run_path, bad, contract_path, manifest_path)
    expected = hashlib.sha256(run_path.read_bytes()).hexdigest()
    run_path.write_bytes(run_path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='byte SHA256'):
        build_event_report(run_path, expected, contract_path, manifest_path)


def test_row_checksum_detects_tampering_even_if_run_hash_updated(artifacts):
    artifacts[3]['completed']['10']['results']['frame']['full-clip']['predicted_frames'][0] = 1
    resave(artifacts)
    with pytest.raises(ValueError, match='Row checksum'):
        build(artifacts)


@pytest.mark.parametrize('mutation', ['contract', 'manifest', 'dataset', 'lock'])
def test_provenance_binding_rejects_changes(artifacts, mutation):
    if mutation in ('contract', 'manifest'):
        path = artifacts[1 if mutation == 'contract' else 2]
        path.write_bytes(path.read_bytes() + b' ')
    else:
        key = 'dataset_sha256' if mutation == 'dataset' else 'dependency_lock_sha256'
        artifacts[3]['immutable'][key] = '0' * 64
        resave(artifacts)
    with pytest.raises(ValueError, match='provenance binding'):
        build(artifacts)


@pytest.mark.parametrize('mutation', ['source-overlap', 'id-overlap', 'wrong-groups', 'duplicate-id',
                                      'wrong-count', 'outer-fold'])
def test_source_disjoint_manifest_and_groups_required(artifacts, mutation):
    contract, manifest = artifacts[4:]
    test = manifest['partitions']['test']
    if mutation == 'source-overlap':
        test['source_ids'] = ['train-source']
        for clip in test['clips']:
            clip['source_id'] = 'train-source'
    elif mutation == 'id-overlap':
        test['clip_ids'][0] = 1
        test['clips'][0]['clip_id'] = 1
    elif mutation == 'wrong-groups':
        test['clips'][0]['source_id'] = 'unlisted-source'
    elif mutation == 'duplicate-id':
        test['clip_ids'][1] = 10
    elif mutation == 'wrong-count':
        contract['expected_test_clips'] = 350
    else:
        contract['outer_fold'] = manifest['outer_fold'] = 2
    resave(artifacts, bindings=True)
    with pytest.raises(ValueError):
        build(artifacts)


@pytest.mark.parametrize(('field', 'replacement'), [
    ('correct', [1] * 8), ('correct', [True] * 8), ('signed_error_frames', [0] * 8),
    ('absolute_error_frames', [0] * 8), ('predicted_frames', [24] * 8),
    ('tolerance_frames', True), ('tolerance_frames', 2), ('event_frames', list(reversed(range(8)))),
    ('annotation_order_anomaly', True), ('signed_error_ms', None), ('absolute_error_ms', None),
    ('absolute_error_ms', [0] * 8), ('signed_error_ms', [False] * 8),
    ('signed_error_ms', [0] * 8), ('pce', 1.0), ('pce_percent', 100.0),
    ('numpy_version', 'other'), ('protocol', 'other'), ('tie_policy', 'other'),
])
def test_row_arithmetic_and_protocol_recomputed_after_valid_rehash(artifacts, field, replacement):
    artifacts[3]['completed']['10']['results']['frame']['full-clip'][field] = replacement
    resave(artifacts, rows=True)
    with pytest.raises(ValueError):
        build(artifacts)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_values_rejected_even_with_bound_bytes(artifacts, bad):
    run = artifacts[3]
    run['completed']['10']['results']['frame']['full-clip']['signed_error_ms'][0] = bad
    # Encode deliberately invalid JSON numeric constants to exercise the read boundary.
    artifacts[0].write_text(json.dumps(run, allow_nan=True))
    with pytest.raises(ValueError, match='Nonfinite'):
        build(artifacts)


def test_numeric_overflow_to_infinity_is_rejected(artifacts):
    artifacts[0].write_text(artifacts[0].read_text().replace('13.25', '1e999'))
    with pytest.raises(ValueError):
        build(artifacts)


def test_heads_and_contexts_must_use_same_targets(artifacts):
    result = artifacts[3]['completed']['10']['results']['temporal']['full-clip']
    # Individually self-consistent result from a different full clip is forbidden.
    result['valid_frame_count'] = 25
    resave(artifacts, rows=True)
    with pytest.raises(ValueError, match='different clip targets'):
        build(artifacts)


def test_frame_tolerance_uses_reference_ties_to_even(artifacts):
    row = artifacts[3]['completed']['10']
    # (75 - 0) / 30 = 2.5, whose reference np.round result is 2, not 3.
    for head in ('frame', 'temporal'):
        for context in CONTEXTS:
            result = row['results'][head][context]
            events = [0, 15, 30, 45, 60, 75, 90, 105]
            result.update(annotation_event_frames=events, annotation_order_anomaly=False,
                          event_frames=events, predicted_frames=events,
                          signed_error_frames=[0] * 8, absolute_error_frames=[0] * 8,
                          signed_error_ms=[0.0] * 8, absolute_error_ms=[0.0] * 8,
                          correct=[True] * 8, pce=1.0, pce_percent=100.0,
                          valid_frame_count=106, tolerance_frames=3)
    resave(artifacts, rows=True)
    with pytest.raises(ValueError, match='tolerance arithmetic'):
        build(artifacts)
    for head in ('frame', 'temporal'):
        for context in CONTEXTS:
            row['results'][head][context]['tolerance_frames'] = 2
    resave(artifacts, rows=True)
    # Now rows pass; the still-original aggregate must be rejected instead.
    with pytest.raises(ValueError, match='summary'):
        build(artifacts)


@pytest.mark.parametrize('key', ['pce', 'pce_percent', 'mae_frames', 'mae_ms',
                                'clip_count', 'annotation_order_anomaly_count'])
def test_summary_recomputation_detects_mutation(artifacts, key):
    artifacts[3]['summary']['frame']['full-clip'][key] += .001
    resave(artifacts)
    with pytest.raises(ValueError, match='summary'):
        build(artifacts)


def test_per_event_summary_and_unexpected_fields_are_checked(artifacts):
    summary = artifacts[3]['summary']['frame']['full-clip']
    summary['per_event']['Finish']['mae_ms'] += 1e-8
    resave(artifacts)
    with pytest.raises(ValueError, match='Finish.mae_ms'):
        build(artifacts)
    summary['per_event']['Finish']['mae_ms'] = 26.5
    summary['unknown_metric'] = 1
    resave(artifacts)
    with pytest.raises(ValueError, match='fields differ'):
        build(artifacts)


def test_recorded_precise_floats_retained_with_tight_tolerance(artifacts):
    summary = artifacts[3]['summary']['frame']['full-clip']
    summary['mae_ms'] += 1e-13
    resave(artifacts)
    assert build(artifacts)['summary']['frame']['full-clip']['mae_ms'] == summary['mae_ms']


def test_duplicate_json_keys_are_refused(artifacts):
    text = artifacts[0].read_text()
    artifacts[0].write_text(text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        build(artifacts)


def test_exclusive_atomic_output_and_symlink_protection(artifacts, tmp_path):
    report = build(artifacts)
    output = tmp_path / 'report.json'
    write_event_report(report, output)
    content = output.read_bytes()
    assert json.loads(content) == report
    with pytest.raises(FileExistsError):
        write_event_report({'replacement': True}, output)
    assert output.read_bytes() == content
    link = tmp_path / 'link.json'
    link.symlink_to(output)
    with pytest.raises(FileExistsError):
        write_event_report(report, link)
    assert output.read_bytes() == content
    assert list(tmp_path.glob('.report.json.*')) == []
    assert list(tmp_path.glob('.link.json.*')) == []


def test_failed_publication_leaves_no_output_or_temp(artifacts, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError('synthetic link failure')
    monkeypatch.setattr(os, 'link', fail)
    with pytest.raises(OSError, match='synthetic link failure'):
        write_event_report(build(artifacts), tmp_path / 'out.json')
    assert not (tmp_path / 'out.json').exists()
    assert not list(tmp_path.glob('.out.json.*'))


def test_import_has_no_ml_or_media_dependencies():
    result = subprocess.run([sys.executable, '-c',
                             'import sys; import backend.swingsync.research.event_report; '
                             'assert not set(("torch", "numpy", "av")) & set(sys.modules)'],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_cli_packages_once_and_refuses_overwrite(artifacts, tmp_path):
    run_path, contract_path, manifest_path = artifacts[:3]
    output = tmp_path / 'cli-report.json'
    args = [sys.executable, '-m', 'scripts.report_event_evaluation', '--run', str(run_path),
            '--run-sha256', hashlib.sha256(run_path.read_bytes()).hexdigest(),
            '--contract', str(contract_path), '--manifest', str(manifest_path),
            '--output', str(output), '--include-order-diagnostics']
    first = subprocess.run(args, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)['test_clips'] == 2
    assert 'posthoc_order_diagnostics' in json.loads(output.read_text())
    original = output.read_bytes()
    second = subprocess.run(args, capture_output=True, text=True, check=False)
    assert second.returncode == 2
    assert 'Event report refused' in second.stderr
    assert output.read_bytes() == original
