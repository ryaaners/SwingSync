"""Generic input software fixtures; no actual model accuracy or selected forwards."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.swingsync.research import event_input as runner


def frames(n=3):
    return [{'frame_index': i, 'pts': 5000+i*40, 'time_base_numerator': 1,
             'time_base_denominator': 1000, 'pts_seconds': 5.+i*.04} for i in range(n)]


def probabilities(n=3):
    result = np.zeros((n, 9), dtype=np.float32)
    result[:, 8] = .92
    result[:, :8] = .01
    return result


def metadata():
    return dict(camera_view='down-the-line', handedness='right', club='driver', rights='software fixture only',
                timestamp_basis='reencoded', unmirrored=True, single_swing_confirmed=True)


def test_independent_peak_ties_preserve_duplicates_and_background():
    values = probabilities()
    result = runner.proposal_rows(values, frames())
    expected = np.argsort(values[:, :8], axis=0)[-1]
    assert [e['frame_index'] for e in result['events']] == expected.tolist()
    assert not result['strict_order'] and result['duplicate_peak_count'] == 7
    for i, event in enumerate(result['events']):
        assert event['class_index'] == i and event['name'] == runner.EVENT_NAMES[i]
        assert event['pts_seconds'] == frames()[event['frame_index']]['pts_seconds']
        assert event['background_probability'] > event['score']


def test_nonchronological_peaks_are_never_repaired():
    array = np.full((8, 9), .01, dtype=np.float32)
    for event in range(8):
        array[7-event, event] = .92
    result = runner.proposal_rows(array, frames(8))
    assert [e['frame_index'] for e in result['events']] == list(reversed(range(8)))
    assert result['duplicate_peak_count'] == 0 and not result['strict_order']


@pytest.mark.parametrize('bad', [np.ones((3, 9), dtype=np.float32), np.zeros((3, 8), dtype=np.float32),
    probabilities().astype(np.float64), np.full((3, 9), np.nan, dtype=np.float32),
    np.full((3, 9), -1., dtype=np.float32), probabilities(1)])
def test_invalid_probability_arrays_fail_closed(bad):
    with pytest.raises(ValueError, match='probabilities'):
        runner.proposal_rows(bad, frames())


def test_atomic_json_no_clobber_or_nan_partial(tmp_path):
    path = tmp_path/'result.json'
    runner._publish_json(path, {'software': True})
    initial = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner._publish_json(path, {'changed': True})
    with pytest.raises(ValueError):
        runner._publish_json(tmp_path/'bad.json', {'bad': float('nan')})
    assert path.read_bytes() == initial
    assert sorted(p.name for p in tmp_path.iterdir()) == ['result.json']


def test_npz_no_clobber_exact_dtype_and_values(tmp_path):
    path = tmp_path/'arrays.npz'
    values = {'features': np.arange(12, dtype=np.float32).reshape(3, 4)}
    runner._publish_arrays(path, values)
    initial = path.read_bytes()
    with np.load(path, allow_pickle=False) as arrays:
        assert np.array_equal(arrays['features'], values['features'])
    with pytest.raises(FileExistsError):
        runner._publish_arrays(path, values)
    assert path.read_bytes() == initial and not list(tmp_path.glob('*.part'))


@pytest.fixture
def job(tmp_path, monkeypatch):
    from backend.swingsync.research import event_input_contract
    monkeypatch.setattr(runner, 'LOCK', tmp_path/'active.lock')
    source = tmp_path/'software-only.mp4'
    source.write_bytes(b'software-only fake video; never passed to a model')
    other = {}
    for name in ('contract', 'annotations', 'manifest', 'protocol', 'weights'):
        p = tmp_path/(name+'.json')
        p.write_text('{"fixture":"software only"}')
        other[name] = p
    monkeypatch.setattr(event_input_contract, 'validate_protocol', lambda _: {
        'fixture': 'software only', 'model_contract': {'sha256': runner.sha256(other['contract'])},
        'dependency_lock_sha256': runner.sha256(runner.ROOT/'uv.lock')})
    inspect = {'frames': frames(), 'width': 160, 'height': 160, 'frame_count': 3}
    monkeypatch.setattr(runner, '_source_identity', lambda _: {'fixture': 'software only'})
    monkeypatch.setattr(runner, 'inspect_bounded', lambda _: copy.deepcopy(inspect))
    admission = {'immutable': {'fixture': 'software only'}, 'contract': {'heads': {
        'frame': {'fixture': 'software only'}, 'temporal': {'fixture': 'software only'}}}}
    monkeypatch.setattr(runner, '_prepare_models', lambda *args: (admission, None, other['weights'],
                                                                {'fixture': 'software only'}))
    monkeypatch.setattr(runner, 'extract_generic_features', lambda *args: np.zeros((3, 1280), dtype=np.float32))
    monkeypatch.setattr(runner, '_predict_heads', lambda *args: {'frame': probabilities(),
                                                              'temporal': probabilities()})
    monkeypatch.setattr(runner, '_assert_integrity', lambda *args, **kwargs: None)
    return dict(video=source, output=tmp_path/'run', metadata=metadata(),
                **{key: other[key] for key in ('contract', 'annotations', 'manifest', 'protocol')})


def test_synthetic_job_publishes_full_frame_arrays_and_honest_gates(job):
    result = runner.run_event_input(**job)
    path = job['output']
    ledger = json.loads((path/'run.json').read_text())
    assert ledger['state'] == result['state'] == 'completed'
    assert runner.sha256(path/'result.json') == ledger['result_sha256']
    assert runner.sha256(path/'arrays.npz') == ledger['arrays_sha256']
    assert result['primary_head'] == 'temporal' and result['temporal_context'] == 'reference-64-reset'
    assert result['frames'] == frames() and result['advice'] == []
    assert result['normal_model_available'] is False and result['validated_for_serving'] is False
    assert result['accuracy_metrics_computed'] is False and result['physical_timing_validated'] is False
    with np.load(path/'arrays.npz', allow_pickle=False) as arrays:
        assert set(arrays.files) == set(runner.ARRAY_NAMES)
        assert arrays['features'].shape == (3, 1280)
        assert arrays['pts_seconds'].dtype == np.float64
        assert arrays['pts_seconds'].tolist() == [5., 5.04, 5.08]


def test_existing_output_refuses_before_models(job, monkeypatch):
    job['output'].mkdir()
    monkeypatch.setattr(runner, '_prepare_models', lambda *args: pytest.fail('No model construction'))
    with pytest.raises(ValueError, match='Output exists'):
        runner.run_event_input(**job)


def test_shared_lock_refuses_without_attempt(job):
    import fcntl
    with runner.LOCK.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='shared lock'):
            runner.run_event_input(**job)
    assert not job['output'].exists()


def test_missing_checkpoint_or_runtime_is_unavailable_no_fake_proposals(job, monkeypatch):
    def unavailable(*args):
        raise ValueError('Required local model is unavailable')
    monkeypatch.setattr(runner, '_prepare_models', unavailable)
    with pytest.raises(ValueError, match='unavailable'):
        runner.run_event_input(**job)
    saved = json.loads((job['output']/'run.json').read_text())
    assert saved['state'] == 'failed' and saved['normal_model_available'] is False
    assert not (job['output']/'arrays.npz').exists() and not (job['output']/'result.json').exists()


def test_preflight_clocks_rejected_before_model_construction(job, monkeypatch):
    def bad(*args):
        raise ValueError('Non-increasing timestamps')
    monkeypatch.setattr(runner, 'inspect_bounded', bad)
    monkeypatch.setattr(runner, '_prepare_models', lambda *args: pytest.fail('No model construction'))
    with pytest.raises(ValueError, match='timestamps'):
        runner.run_event_input(**job)
    assert json.loads((job['output']/'run.json').read_text())['state'] == 'failed'


@pytest.mark.parametrize('phase', ['after_inference', 'after_result', 'after_completed_ledger'])
def test_drift_at_each_publication_boundary_removes_completed_outputs(job, monkeypatch, phase):
    calls = []
    def drift(*args, **kwargs):
        calls.append(1)
        if len(calls) == {'after_inference': 1, 'after_result': 2, 'after_completed_ledger': 3}[phase]:
            raise ValueError('source drift')
    monkeypatch.setattr(runner, '_assert_integrity', drift)
    with pytest.raises(ValueError, match='source drift'):
        runner.run_event_input(**job)
    assert json.loads((job['output']/'run.json').read_text())['state'] == 'failed'
    assert not (job['output']/'result.json').exists() and not (job['output']/'arrays.npz').exists()


def test_failed_completed_ledger_publish_removes_unbound_result(job, monkeypatch):
    original = runner._publish_json
    def fail(path, value, **kwargs):
        if path.name == 'run.json' and value['state'] == 'completed':
            raise OSError('synthetic publication failure')
        return original(path, value, **kwargs)
    monkeypatch.setattr(runner, '_publish_json', fail)
    with pytest.raises(OSError, match='synthetic publication failure'):
        runner.run_event_input(**job)
    assert json.loads((job['output']/'run.json').read_text())['state'] == 'failed'
    assert not (job['output']/'result.json').exists() and not (job['output']/'arrays.npz').exists()


def test_native_preflight_uses_separate_process_and_timeout(monkeypatch):
    calls = []
    def child(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps({'frames': frames()}))
    monkeypatch.setattr(runner.subprocess, 'run', child)
    assert runner.inspect_bounded(Path('synthetic.mp4'))['frames'] == frames()
    assert calls[0][0][:3] == [runner.sys.executable, '-m', 'backend.swingsync.research.event_input']
    assert calls[0][1]['timeout'] == 45 and calls[0][1]['check'] is True


def test_protocol_refusal_precedes_attempt_and_model_load(job, monkeypatch):
    from backend.swingsync.research import event_input_contract
    def bad_protocol(_):
        raise ValueError('Unsupported protocol')
    monkeypatch.setattr(event_input_contract, 'validate_protocol', bad_protocol)
    monkeypatch.setattr(runner, '_prepare_models', lambda *args: pytest.fail('No model construction'))
    with pytest.raises(ValueError, match='Unsupported protocol'):
        runner.run_event_input(**job)
    assert not job['output'].exists()


def test_generic_tensor_extraction_keeps_all_frames_and_final_batch(tmp_path):
    torch = pytest.importorskip('torch')
    import av
    from fractions import Fraction
    from torchvision.models import MobileNet_V2_Weights
    from backend.swingsync.research.event_input_contract import inspect_input
    video = tmp_path/'fixture.mp4'
    with av.open(str(video), 'w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=25)
        stream.width, stream.height, stream.pix_fmt = 80, 48, 'yuv420p'
        stream.codec_context.sample_aspect_ratio = Fraction(1, 1)
        stream.sample_aspect_ratio = Fraction(1, 1)
        for i in range(17):
            rgb = np.full((48, 80, 3), (i*11) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    inspection = inspect_input(video)

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []
        def forward(self, images):
            self.batch_sizes.append(len(images))
            return images[:, 0, 0, 0][:, None].expand(-1, 1280).clone()

    backbone = FakeBackbone()
    actual = runner.extract_generic_features(video, inspection, backbone)
    expected = []
    transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
    with av.open(str(video), format='mov') as container:
        for frame in container.decode(video=0):
            expected.append(float(transform(torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1))[0, 0, 0]))
    assert actual.shape == (17, 1280) and actual.dtype == np.float32
    assert backbone.batch_sizes == [16, 1]
    assert np.array_equal(actual[:, 0], np.array(expected, dtype=np.float32))
    assert len(inspection['frames']) == 17
    broken = copy.deepcopy(inspection)
    broken['frames'][0]['pts'] += 1
    with pytest.raises(ValueError, match='PTS/raster'):
        runner.extract_generic_features(video, broken, FakeBackbone())


def test_model_contract_must_match_protocol_before_attempt(job, monkeypatch):
    from backend.swingsync.research import event_input_contract
    monkeypatch.setattr(event_input_contract, 'validate_protocol', lambda _: {
        'model_contract': {'sha256': '0'*64},
        'dependency_lock_sha256': runner.sha256(runner.ROOT/'uv.lock')})
    monkeypatch.setattr(runner, '_prepare_models', lambda *args: pytest.fail('No model construction'))
    with pytest.raises(ValueError, match='model contract/root lock'):
        runner.run_event_input(**job)
    assert not job['output'].exists()
