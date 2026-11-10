"""Synthetic input contracts prove software admission, never golf accuracy."""
from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from backend.swingsync.research import event_input_contract as contract


def metadata(**changes):
    return dict(camera_view='face-on', handedness='right', club='driver', unmirrored=True,
                single_swing_confirmed=True, rights='Synthetic fixture, no accuracy evidence',
                timestamp_basis='source') | changes


def stream(**changes):
    return SimpleNamespace(metadata={}, sample_aspect_ratio=Fraction(1),
                           codec_context=SimpleNamespace(width=640, height=360),
                           duration=None, time_base=Fraction(1, 1000), **changes)


def frame(pts=5000, **changes):
    return SimpleNamespace(**(dict(pts=pts, time_base=Fraction(1, 1000), duration=1,
                                  width=640, height=360, rotation=0, side_data=[]) | changes))


def fake_input(tmp_path, monkeypatch, frames, *, video_stream=None, stream_count=1):
    p = tmp_path / 'fixture.mp4'
    p.write_bytes(b'Explicitly synthetic decoder fixture')
    active_stream = stream() if video_stream is None else video_stream

    class Container:
        streams = SimpleNamespace(video=[active_stream] * stream_count)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, **kwargs):
            yield from frames

    def open_video(path, *, format, options):
        assert Path(path) == p and format == 'mov' and options == {'protocol_whitelist': 'file'}
        return Container()

    monkeypatch.setitem(sys.modules, 'av', SimpleNamespace(open=open_video))
    return p


def test_explicit_metadata_is_declaration_not_reliability():
    m = contract.validate_metadata(**metadata())
    assert m['club'] == 'driver' and m['timestamp_basis'] == 'source'
    assert m['mirrored'] is False and m['input_distribution_validated'] is False
    assert not m['metadata_empirically_validated'] and not m['physical_timing_validated']


@pytest.mark.parametrize('changes', [
    {'camera_view': 'unknown'}, {'handedness': 'unknown'}, {'unmirrored': False}, {'unmirrored': 1},
    {'single_swing_confirmed': False}, {'single_swing_confirmed': 1}, {'club': ''}, {'club': 'x'*81},
    {'rights': ''}, {'rights': 'x'*2001}, {'timestamp_basis': 'fps'},
])
def test_ambiguous_metadata_refused(changes):
    with pytest.raises(ValueError):
        contract.validate_metadata(**metadata(**changes))


@pytest.mark.parametrize('w,h,resized,crop', [
    (160, 160, (232, 232), (4, 4)), (640, 360, (412, 232), (94, 4)),
    (360, 640, (232, 412), (4, 94)), (640, 359, (413, 232), (94, 4)),
    (359, 640, (232, 413), (4, 94)),
])
def test_pinned_resize_center_crop_geometry(w, h, resized, crop):
    g = contract.transform_geometry(w, h)
    assert (g['resized_width'], g['resized_height']) == resized
    assert (g['center_crop_left'], g['center_crop_top']) == crop
    assert g['pre_backbone_roi_crop'] is None and not g['crop_subject_coverage_validated']


@pytest.mark.parametrize('width,height', [(True, 100), (0, 100), (100, -1), (1.5, 100)])
def test_geometry_requires_integer_dimensions(width, height):
    with pytest.raises(ValueError):
        contract.transform_geometry(width, height)


def test_native_vfr_nonzero_origin_preserved_without_ms_quantization(tmp_path, monkeypatch):
    p = fake_input(tmp_path, monkeypatch, [frame(5000), frame(5001, time_base=Fraction(1, 1000)),
                                         frame(50014, time_base=Fraction(1, 10000))])
    result = contract.inspect_input(p)
    assert [r['pts_seconds'] for r in result['frames']] == [5., 5.001, 5.0014]
    assert [r['pts'] for r in result['frames']] == [5000, 5001, 50014]
    assert [r['time_base_denominator'] for r in result['frames']] == [1000, 1000, 10000]
    assert [r['frame_index'] for r in result['frames']] == [0, 1, 2]
    assert result['duration_seconds'] == pytest.approx(.0015)
    assert 'api_timestamp_ms' not in result['frames'][0]
    assert result['crop'] is None and result['resize'] is None


@pytest.mark.parametrize('frames', [[], [frame()], [frame(0), frame(0)], [frame(1), frame(0)],
    [frame(0), frame(None)], [frame(0), frame(1, time_base=None)],
    [frame(0), frame(True)], [frame(0), frame(2**63)],
    [frame(2**62), frame(2**62+1)], [frame(0), frame(30001)],
    [frame(0), frame(30000, duration=1)], [frame(0), frame(34, duration=-1)],
    [frame(0), frame(34, width=641)]])
def test_bad_frame_clocks_durations_and_coverage_fail(tmp_path, monkeypatch, frames):
    p = fake_input(tmp_path, monkeypatch, frames)
    with pytest.raises(ValueError):
        contract.inspect_input(p)


def test_unknown_final_duration_not_invented_from_fps(tmp_path, monkeypatch):
    p = fake_input(tmp_path, monkeypatch, [frame(5000), frame(5034, duration=0)])
    result = contract.inspect_input(p)
    assert result['duration_seconds'] == .034
    assert result['duration_method'] == 'native-PTS-span-only'
    assert result['last_frame_duration_seconds'] is None


def test_frame_budget_and_stream_multiplicity(tmp_path, monkeypatch):
    p = fake_input(tmp_path, monkeypatch, [frame(i) for i in range(3601)])
    with pytest.raises(ValueError, match='3600'):
        contract.inspect_input(p)
    p = fake_input(tmp_path, monkeypatch, [frame(0), frame(1)], stream_count=2)
    with pytest.raises(ValueError, match='Exactly one'):
        contract.inspect_input(p)


def test_geometry_refuses_ambiguous_transforms_and_sizes():
    s, f = stream(), frame()
    contract._check_geometry(s, f)
    for rotation in ('90', 'unknown', 'nan'):
        s.metadata = {'rotate': rotation}
        with pytest.raises(ValueError):
            contract._check_geometry(s, f)
    s.metadata = {}
    for sar in (None, Fraction(2)):
        s.sample_aspect_ratio = sar
        with pytest.raises(ValueError, match='square pixel'):
            contract._check_geometry(s, f)
    s.sample_aspect_ratio = Fraction(1)
    for changes in ({'rotation': 90}, {'side_data': [SimpleNamespace(type='DISPLAY_MATRIX')]}, {'width': 3841}):
        with pytest.raises(ValueError):
            contract._check_geometry(s, frame(**changes))
    s.codec_context.width = 0
    with pytest.raises(ValueError, match='Coded dimensions'):
        contract._check_geometry(s)


def test_declared_duration_size_and_extension_refused(tmp_path, monkeypatch):
    s = stream()
    s.duration = 30001
    p = fake_input(tmp_path, monkeypatch, [frame(0), frame(1)], video_stream=s)
    with pytest.raises(ValueError, match='Declared'):
        contract.inspect_input(p)
    p.write_bytes(b'')
    with pytest.raises(ValueError, match='empty'):
        contract.inspect_input(p)
    with pytest.raises(ValueError, match='extension'):
        contract.inspect_input(p.with_suffix('.txt'))


def test_real_tiny_video_exact_decoded_timestamps(tmp_path):
    import av
    import numpy as np
    p = tmp_path / 'native.mp4'
    with av.open(str(p), mode='w') as output:
        s = output.add_stream('mpeg4', rate=30)
        s.width = s.height = 16
        s.pix_fmt = 'yuv420p'
        s.sample_aspect_ratio = Fraction(1)
        s.time_base = Fraction(1, 1000)
        for pts in (5000, 5034, 5080):
            f = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format='rgb24')
            f.pts, f.time_base = pts, Fraction(1, 1000)
            for packet in s.encode(f):
                output.mux(packet)
        for packet in s.encode():
            output.mux(packet)
    with av.open(str(p)) as inp:
        expected = [(f.pts, f.time_base.numerator, f.time_base.denominator,
                     float(f.pts*f.time_base)) for f in inp.decode(video=0)]
    got = contract.inspect_input(p)
    assert [(f['pts'], f['time_base_numerator'], f['time_base_denominator'], f['pts_seconds'])
            for f in got['frames']] == expected
    assert got['frame_count'] == 3 and got['width'] == 16


def test_registered_protocol_admits_fixed_operational_policy():
    path = Path(__file__).resolve().parents[1] / 'experiments/event-input-v1.json'
    protocol = contract.validate_protocol(path)
    assert protocol['inference']['context'] == 'reference-64-reset'
    assert [f['clip_id'] for f in protocol['parity']['fixtures']] == [0, 1]
    assert all(value is False for value in protocol['gates'].values())


@pytest.mark.parametrize('section,field,value', [
    ('inference', 'batch_size', 8), ('inference', 'torch_num_threads', True),
    ('inference', 'context', 'full-clip'), ('inference', 'selection', 'model'),
    ('inference', 'order_constrained_decoding', True), ('input_policy', 'max_frames', 9999),
    ('input_policy', 'pre_backbone_roi_crop', [0, 0, 1, 1]),
    ('gates', 'normal_app_model_available', True),
])
def test_rehashed_policy_changes_still_fail_closed(tmp_path, monkeypatch, section, field, value):
    import hashlib
    import json
    original = Path(__file__).resolve().parents[1] / 'experiments/event-input-v1.json'
    document = json.loads(original.read_text())
    document[section][field] = value
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='byte fingerprint'):
        contract.validate_protocol(path)
    monkeypatch.setattr(contract, 'INPUT_PROTOCOL_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        contract.validate_protocol(path)
