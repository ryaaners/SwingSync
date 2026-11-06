"""Synthetic timing/streaming checks; samples are never real hardware claims."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from test_features import TinyFrozen, fixture_video  # noqa: E402,F401
from backend.swingsync.research.features import extract_clip_features  # noqa: E402
from backend.swingsync.research import latency  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Clock:
    """One synthetic second per clock call, not a benchmark."""
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return self.value


def test_stats_quantiles_state_units_and_reject_invalid_samples():
    result = latency.timing_summary([1, 3, 5], sample_unit='synthetic frame')
    assert result['sample_count'] == 3 and result['total_seconds'] == 9
    assert result['p50_seconds'] == 3 and result['p95_seconds'] == pytest.approx(4.8)
    assert result['sample_unit'] == 'synthetic frame'
    for bad in ([], [np.nan], [-1], [[1, 2]], [np.inf]):
        with pytest.raises(ValueError):
            latency.timing_summary(bad, sample_unit='synthetic')


def test_timed_extraction_matches_registered_extractor_features_pts_and_batches(fixture_video):  # noqa: F811
    path, record = fixture_video
    expected = extract_clip_features(path, record, annotation_sha256='a' * 64,
                                     backbone=TinyFrozen(), batch_size=4)
    model = TinyFrozen()
    features, pts, result = latency._extract_timed(path, record, model, batch_size=4, clock=Clock())
    np.testing.assert_array_equal(features, expected.features)
    np.testing.assert_array_equal(pts, expected.pts_seconds)
    assert model.batch_lengths == result['batch_frame_counts'] == [4, 4, 1]
    assert result['frame_count'] == 9
    assert result['decode']['sample_count'] == result['preprocessing']['sample_count'] == 9
    assert result['backbone']['sample_count'] == 3
    assert result['decode']['mean_seconds'] == result['preprocessing']['mean_seconds'] == 1
    assert result['backbone']['mean_seconds'] == 1
    assert result['decoder_eof_drain_seconds'] == result['container_open_seconds'] == 1


def test_fixed_warmup_only_decodes_requested_training_prefix(fixture_video):  # noqa: F811
    path, record = fixture_video
    model = TinyFrozen()
    features, pts, result = latency._extract_timed(path, record, model, batch_size=4, limit=4)
    assert len(features) == len(pts) == 4
    assert model.batch_lengths == [4]
    assert result['decoder_eof_drain_seconds'] == 0
    assert pts.tolist() == pytest.approx([0, .042, .088, .138])


def test_timing_rejects_missing_or_extra_frames(fixture_video):  # noqa: F811
    path, record = fixture_video
    for frames in (8, 10):
        changed = replace(record, original_events=(*record.original_events[:-1], frames))
        with pytest.raises(ValueError, match='frame count'):
            latency._extract_timed(path, changed, TinyFrozen(), batch_size=4)


def test_measure_clip_preserves_pts_scopes_and_all_fixed_variants(monkeypatch, fixture_video):  # noqa: F811
    path, record = fixture_video
    calls = []
    def prediction(model, features, context):
        calls.append((model, len(features), context))
        return np.ones((len(features), 9)) / 9
    monkeypatch.setitem(sys.modules, 'backend.swingsync.research.heldout',
                        SimpleNamespace(predict_probabilities=prediction))
    expected = [.0, .042, .088, .138, .192, .250, .312, .378, .448]
    result = latency.measure_clip(path, record, TinyFrozen(), {'frame': 'frame', 'temporal': 'temporal'},
                                  expected_pts=expected, batch_size=4, clock=Clock())
    assert calls == [(name, 9, context) for name in ('frame', 'temporal') for context in latency.CONTEXTS]
    assert result['timestamp_provenance'] == 'reencoded-clip-pts'
    assert result['first_pts_seconds'] == 0 and result['last_pts_seconds'] == .448
    for head in result['heads'].values():
        for timing in head.values():
            assert timing['head_seconds'] == 1
            assert timing['composed_video_pipeline_seconds'] == result['feature_extraction_wall_seconds'] + 1
            assert 'not a separate replay' in timing['composition']
    assert 'all four' in result['measured_clip_wall_scope']
    with pytest.raises(ValueError, match='PTS differ'):
        latency.measure_clip(path, record, TinyFrozen(), {'frame': None, 'temporal': None},
                             expected_pts=np.arange(9), batch_size=4)


def test_atomic_report_never_replaces_existing_output(tmp_path):
    path = tmp_path / 'report.json'
    latency._exclusive_json(path, {'scope': 'synthetic software fixture'})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        latency._exclusive_json(path, {'scope': 'overwrite'})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('.latency-*'))


def test_missing_local_weights_fail_without_constructor_or_download(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path))
    monkeypatch.setattr(latency, 'FrozenMobileNetV2', lambda **_: pytest.fail('must not construct/download'))
    with pytest.raises(ValueError, match='unavailable'):
        latency._load_local_backbone()


def test_benchmark_fixed_training_config_and_failure_state(monkeypatch, tmp_path):
    contract = {'latency': deepcopy(latency.LATENCY_CONFIG)}
    contract['latency']['clip_ids'] = [2, 3]
    admitted = {'contract': contract, 'dataset': SimpleNamespace(records=()),
                'manifest': {'partitions': {'train': {'clip_ids': [0, 1]}}}, 'heads': {}}
    monkeypatch.setitem(sys.modules, 'backend.swingsync.research.heldout', SimpleNamespace(
        validate_contract=lambda *_: admitted, predict_probabilities=lambda *_: None))
    monkeypatch.setattr(latency, 'DEFAULT_LOCK_PATH', tmp_path / 'lock')
    monkeypatch.setattr(latency, '_git_base_commit', lambda: 'software-fixture')
    monkeypatch.setattr(latency, '_process_start', lambda _: 'software-fixture')
    output = tmp_path / 'timing.json'
    with pytest.raises(ValueError, match='fixed train-only'):
        latency.run_latency_benchmark('contract', 'annotations', 'manifest', 'clips', output)
    assert not output.exists()
    state = json.loads(Path(str(output) + '.run.json').read_text())
    assert state['state'] == 'failed'
    assert 'latency.py' in state['timing_implementation_sha256']
    with pytest.raises(ValueError, match='already exists'):
        latency.run_latency_benchmark('contract', 'annotations', 'manifest', 'clips', output)


@pytest.mark.parametrize('config', [{'batch_size': 0}, {'batch_size': True}, {'limit': 0}, {'limit': False}])
def test_invalid_bounded_extraction_config_rejected(fixture_video, config):  # noqa: F811
    path, record = fixture_video
    with pytest.raises(ValueError):
        latency._extract_timed(path, record, TinyFrozen(), **config)


@pytest.fixture
def admitted_benchmark(monkeypatch, tmp_path, fixture_video):  # noqa: F811
    """Synthetic admission/timings exercise publication, never model performance."""
    source, record = fixture_video
    clips = tmp_path / 'clips'
    clips.mkdir()
    for clip_id in (0, 1):
        (clips / f'{clip_id}.mp4').write_bytes(source.read_bytes())
    backbone = TinyFrozen()
    weight_path = tmp_path / 'synthetic-reference-bytes'
    weight_path.write_bytes(b'not actual ImageNet; injected only for software publication tests')
    records = (record, replace(record, clip_id=1, source_id='other-software-source'))
    contract = {'latency': {**deepcopy(latency.LATENCY_CONFIG), 'scope': 'synthetic timing fixture'},
                'feature_pipeline': {}}
    admission = {'contract': contract, 'dataset': SimpleNamespace(records=records, sha256='a' * 64),
                 'manifest': {'partitions': {'train': {'clip_ids': [0, 1]}}},
                 'heads': {'frame': object(), 'temporal': object()}, 'immutable': {'synthetic': True}}
    drift = {'kind': None, 'active': False}
    def integrity(*_):
        if drift['active'] and drift['kind'] == 'helper-or-contract':
            raise ValueError('Evaluation source/config bytes changed during invocation; publication refused')
    monkeypatch.setitem(sys.modules, 'backend.swingsync.research.heldout', SimpleNamespace(
        validate_contract=lambda *_: admission, predict_probabilities=lambda *_: np.ones((16, 9)) / 9,
        _assert_run_integrity=integrity))
    monkeypatch.setattr(latency, 'DEFAULT_LOCK_PATH', tmp_path / 'lock')
    monkeypatch.setattr(latency, '_git_base_commit', lambda: 'synthetic-revision')
    monkeypatch.setattr(latency, '_process_start', lambda _: 'synthetic-start')
    monkeypatch.setattr(latency, '_hardware', lambda: {'processor': 'synthetic'})
    monkeypatch.setattr(latency, 'detect_devices', lambda: {'cpu': True})
    monkeypatch.setattr(latency, '_load_local_backbone', lambda: (backbone, weight_path))
    monkeypatch.setattr(latency, 'build_cache_identity', lambda path, *_args, **_kwargs: {
        'video_sha256': latency._hash(path), 'backbone_state_sha256': latency.backbone_state_sha256(backbone)})
    monkeypatch.setattr(latency, 'inspect_clip_timeline', lambda *_args, **_kwargs:
                        SimpleNamespace(pts_seconds=tuple(range(9))))
    monkeypatch.setattr(latency, '_extract_timed', lambda *_args, **_kwargs:
                        (np.zeros((16, 1280), dtype=np.float32), np.arange(16), {'frame_count': 16}))
    def measure(*_args, **_kwargs):
        if drift['kind'] is not None:
            drift['active'] = True
        return {'frame_count': 9, 'feature_extraction_wall_seconds': 1.,
                'heads': {name: {context: {'head_seconds': .01} for context in latency.CONTEXTS}
                          for name in ('frame', 'temporal')}}
    monkeypatch.setattr(latency, 'measure_clip', measure)
    original_hash = latency._hash
    def hash_drift(path):
        if drift['active'] and drift['kind'] == 'own-module' and Path(path).resolve() == Path(latency.__file__).resolve():
            return '0' * 64
        return original_hash(path)
    monkeypatch.setattr(latency, '_hash', hash_drift)
    return tmp_path, clips, drift


@pytest.mark.parametrize('kind', ['own-module', 'helper-or-contract'])
def test_benchmark_refuses_drift_before_publication_and_records_failed_state(admitted_benchmark, kind):
    root, clips, drift = admitted_benchmark
    drift['kind'] = kind
    output = root / 'timing.json'
    with pytest.raises(ValueError, match='changed during invocation'):
        latency.run_latency_benchmark('contract', 'annotations', 'manifest', clips, output)
    assert not output.exists()
    state = json.loads(Path(str(output) + '.run.json').read_text())
    assert state['state'] == 'failed'
    assert 'publication refused' in state['error']
    assert state['output'] == str(output)


def test_benchmark_retracts_own_output_if_drift_detected_after_publication(monkeypatch, admitted_benchmark):
    root, clips, drift = admitted_benchmark
    output = root / 'timing.json'
    original_publish = latency._exclusive_json
    def publish_then_drift(path, result):
        original_publish(path, result)
        if Path(path) == output:
            drift.update(kind='own-module', active=True)
    monkeypatch.setattr(latency, '_exclusive_json', publish_then_drift)
    with pytest.raises(ValueError, match='Timing source bytes changed'):
        latency.run_latency_benchmark('contract', 'annotations', 'manifest', clips, output)
    assert not output.exists()
    assert json.loads(Path(str(output) + '.run.json').read_text())['state'] == 'failed'


def test_synthetic_publication_succeeds_with_stable_sources_and_scope_annotation(admitted_benchmark):
    root, clips, _ = admitted_benchmark
    output = root / 'timing.json'
    result = latency.run_latency_benchmark('contract', 'annotations', 'manifest', clips, output)
    assert result['state'] == 'completed'
    assert result['frame_count'] == 18
    assert result['accuracy_metrics_computed'] is result['outer_test_evaluated'] is False
    assert result['runtime']['hardware']['processor'] == 'synthetic'
    state = json.loads(Path(str(output) + '.run.json').read_text())
    assert state['state'] == 'completed' and state['output_sha256'] == latency._hash(output)
