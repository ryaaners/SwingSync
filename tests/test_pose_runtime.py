"""Synthetic software contracts, never landmark/identity accuracy evidence."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from backend.swingsync.pose_runtime.contracts import (
    as_pose_sequence, pose_observation, validate_context, video_clock,
)
from backend.swingsync.pose_runtime import extraction as ex


def point(**updates):
    return SimpleNamespace(**({'x': .2, 'y': .3, 'z': -.1, 'visibility': .9, 'presence': .8} | updates))


def context(**updates):
    return dict(camera_view='face-on', handedness='right', unmirrored=True,
                single_golfer_confirmed=True, rights='Synthetic fixture, no accuracy claim',
                timestamp_basis='reencoded') | updates


def test_rational_vfr_clock_nonzero_origin_half_up_and_immutable():
    ticks = [(5000, 1, 1000), (10001, 1, 2000), (5002, 1, 1000), (5034, 1, 1000)]
    clocks = video_clock(ticks)
    assert [f.pts_seconds for f in clocks] == [5, 5.0005, 5.002, 5.034]
    assert [f.api_timestamp_ms for f in clocks] == [0, 1, 2, 34]
    assert [f.frame_index for f in clocks] == [0, 1, 2, 3]
    assert clocks[1].pts == 10001 and clocks[1].time_base_denominator == 2000
    ticks[0] = (0, 1, 1)
    assert clocks[0].pts_seconds == 5
    with pytest.raises(FrozenInstanceError):
        clocks[0].api_timestamp_ms = 1


@pytest.mark.parametrize('values', [[], [(0, 1, 1), (0, 1, 1)], [(1, 1, 1), (0, 1, 1)],
    [(0, 1, 10000), (4, 1, 10000), (11, 1, 10000)], [(True, 1, 1)], [(0, 0, 1)],
    [(0, 1, 0)], [(0., 1, 1)], [(2**63, 1, 1)], [(0, 1, 1), (2**62, 1, 1)],
    [(2**62, 1, 1000), (2**62 + 1, 1, 1000)]])
def test_bad_clocks_fail_without_jitter(values):
    with pytest.raises(ValueError):
        video_clock(values)


def test_negative_native_origin_retained():
    clocks = video_clock([(-3, 1, 100), (-2, 1, 100)])
    assert clocks[0].pts_seconds == -.03 and clocks[1].api_timestamp_ms == 10


@pytest.mark.parametrize('changes', [{'camera_view': 'unknown'}, {'handedness': 'unknown'},
    {'unmirrored': False}, {'unmirrored': 1}, {'single_golfer_confirmed': False},
    {'rights': ''}, {'timestamp_basis': 'fps-guess'}])
def test_ambiguous_context_refused(changes):
    with pytest.raises(ValueError):
        validate_context(**context(**changes))


def test_missing_multiple_and_no_silent_first_pose():
    assert pose_observation([])['status'] == 'no_pose'
    r = pose_observation([[point() for _ in range(33)], [point() for _ in range(33)]])
    assert r['status'] == 'multiple_poses' and r['pose_count'] == 2
    assert all(v is None for p in r['landmarks'] for v in p)
    assert len(r['raw_poses']) == 2


@pytest.mark.parametrize('update', [{'visibility': None}, {'presence': None}, {'presence': float('nan')},
    {'visibility': 1.1}, {'x': float('inf')}, {'presence': True}])
def test_malformed_scores_or_coordinates_never_become_measurement(update):
    poses = [[point() for _ in range(33)]]
    poses[0][13] = point(**update)
    r = pose_observation(poses)
    assert r['status'] == 'malformed_pose'
    assert r['landmarks'] == [[None] * 4 for _ in range(33)]
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('floor', [None, True, float('nan'), -1, 2])
def test_presence_filter_config_validated(floor):
    with pytest.raises(ValueError):
        pose_observation([], presence_floor=floor)


def test_presence_filter_retains_raw_visibility_and_anatomical_ids():
    poses = [[point(x=i / 20, presence=.4 if i == 13 else .8) for i in range(33)]]
    r = pose_observation(poses)
    assert r['status'] == 'single_pose'
    assert r['presence'][13] == .4 and r['raw_poses'][0][13]['visibility'] == .9
    assert r['landmarks'][13] == [None] * 4
    assert r['landmarks'][12][0] == .6 and r['landmarks'][32][0] == 1.6  # no clamping/swapping
    assert r['presence_filtered_indices'] == [13]
    poses[0][12].x = 99
    assert r['landmarks'][12][0] == .6


def test_wrong_landmark_count_explicit_malformed():
    assert pose_observation([[point()]])['status'] == 'malformed_pose'


def test_pose_sequence_preserves_native_times_and_owned_readonly_arrays():
    ctx = validate_context(**context())
    frames = [dict(pts_seconds=5.0005, **pose_observation([[point() for _ in range(33)]])),
              dict(pts_seconds=5.034, **pose_observation([]))]
    result = {'schema_version': 1, 'state': 'completed', 'context': ctx,
              'frames': frames, 'video': {'width': 640, 'height': 360}}
    sequence = as_pose_sequence(result)
    assert sequence.timestamps_seconds.tolist() == [5.0005, 5.034]
    assert sequence.landmarks.shape == (2, 33, 4)
    assert np.isnan(sequence.landmarks[1]).all()
    assert not sequence.landmarks.flags.writeable and not sequence.timestamps_seconds.flags.writeable
    frames[0]['landmarks'][0][0] = 99
    assert sequence.landmarks[0, 0, 0] == .2
    result['state'] = 'running'
    with pytest.raises(ValueError):
        as_pose_sequence(result)


def test_atomic_publication_no_overwrite_or_part_on_json_failure(tmp_path):
    dest = tmp_path / 'result.json'
    ex.publish_json(dest, {'complete': 1})
    old = dest.read_bytes()
    with pytest.raises(FileExistsError):
        ex.publish_json(dest, {'complete': 2})
    assert dest.read_bytes() == old
    with pytest.raises(ValueError):
        ex.publish_json(tmp_path / 'bad.json', {'bad': float('nan')})
    assert sorted(p.name for p in tmp_path.iterdir()) == ['result.json']


def model_fixture(tmp_path, monkeypatch):
    model = tmp_path / 'model.task'
    with zipfile.ZipFile(model, 'w') as z:
        z.writestr('pose_detector.tflite', b'not-a-model')
        z.writestr('pose_landmarks_detector.tflite', b'not-a-model-either')
    import hashlib
    manifest = {'schema_version': 1, 'url': ex.MODEL_URL, 'size_bytes': model.stat().st_size,
                'sha256': ex.file_sha256(model), 'license_evidence': {'fixture': 'software only'}, 'members': []}
    with zipfile.ZipFile(model) as z:
        for info in z.infolist():
            manifest['members'].append({'name': info.filename, 'size_bytes': info.file_size,
                                        'sha256': hashlib.sha256(z.read(info)).hexdigest()})
    monkeypatch.setattr(ex, 'LICENSE_EVIDENCE', manifest['license_evidence'])
    monkeypatch.setattr(ex, 'MODEL_SHA256', manifest['sha256'])
    monkeypatch.setattr(ex, 'MODEL_SIZE_BYTES', manifest['size_bytes'])
    monkeypatch.setattr(ex, 'MODEL_MEMBERS', json.loads(json.dumps(manifest['members'])))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return model, path


def test_model_exact_bundle_member_and_manifest_binding(tmp_path, monkeypatch):
    model, manifest = model_fixture(tmp_path, monkeypatch)
    identity = ex.validate_model(model, manifest)
    assert identity['manifest_sha256'] == ex.file_sha256(manifest)
    doc = json.loads(manifest.read_text())
    doc['members'][0]['sha256'] = '0' * 64
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='member'):
        ex.validate_model(model, manifest)
    doc['members'] = ex.MODEL_MEMBERS
    manifest.write_text(json.dumps(doc))
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='byte'):
        ex.validate_model(model, manifest)


def test_missing_model_and_unregistered_url_unavailable(tmp_path, monkeypatch):
    model, manifest = model_fixture(tmp_path, monkeypatch)
    doc = json.loads(manifest.read_text())
    doc['url'] = 'https://example.org/latest'
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='Unregistered'):
        ex.validate_model(model, manifest)
    with pytest.raises(FileNotFoundError):
        ex.validate_model(model, tmp_path / 'missing')


def test_geometry_rejects_transform_ambiguity():
    stream = SimpleNamespace(metadata={}, sample_aspect_ratio=Fraction(1))
    frame = SimpleNamespace(rotation=0, side_data=[], width=640, height=360)
    ex._check_geometry(stream, frame)
    for metadata in ({'rotate': '90'}, {'rotate': 'unknown'}):
        stream.metadata = metadata
        with pytest.raises(ValueError):
            ex._check_geometry(stream, frame)
    stream.metadata = {}
    stream.sample_aspect_ratio = None
    with pytest.raises(ValueError):
        ex._check_geometry(stream, frame)
    stream.sample_aspect_ratio = Fraction(2)
    with pytest.raises(ValueError):
        ex._check_geometry(stream, frame)
    stream.sample_aspect_ratio = Fraction(1)
    frame.rotation = 90
    with pytest.raises(ValueError):
        ex._check_geometry(stream, frame)
    frame.rotation = 0
    frame.side_data = [SimpleNamespace(type='DISPLAYMATRIX')]
    with pytest.raises(ValueError):
        ex._check_geometry(stream, frame)


def test_existing_output_no_duplicate_job_before_native_import(tmp_path):
    with pytest.raises(ValueError, match='already exists'):
        ex.extract_pose(video='missing.mp4', model='missing.task', model_manifest='missing.json',
                        output=tmp_path, **context())


def test_global_lock_busy_refuses_without_creating_attempt(tmp_path, monkeypatch):
    import fcntl
    lock_path = tmp_path / 'active.lock'
    monkeypatch.setattr(ex, 'LOCK', lock_path)
    with lock_path.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='shared lock'):
            ex.extract_pose(video='missing.mp4', model='missing.task', model_manifest='missing.json',
                            output=tmp_path / 'attempt', **context())
    assert not (tmp_path / 'attempt').exists()


def test_native_preflight_failure_records_failed_identity_and_no_result(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(ex, 'LOCK', tmp_path / 'active.lock')
    monkeypatch.setattr(ex, 'runtime_identity', lambda: {'fixture': 'software-only'})
    monkeypatch.setattr(ex, 'validate_model', lambda *args: {'fixture': 'software-only'})
    monkeypatch.setattr(ex, 'require_network_denial', lambda: None)
    monkeypatch.setattr(ex, 'inspect_video', lambda *args: (_ for _ in ()).throw(RuntimeError('native failure')))
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(__version__='fixture', getBuildInformation=lambda: 'fixture'))
    monkeypatch.setitem(sys.modules, 'mediapipe', SimpleNamespace(__version__='fixture'))
    video = tmp_path / 'fixture.mp4'
    video.write_bytes(b'synthetic')
    out = tmp_path / 'attempt'
    with pytest.raises(RuntimeError, match='native failure'):
        ex.extract_pose(video=video, model='fixture', model_manifest='fixture', output=out, **context())
    run = json.loads((out / 'run.json').read_text())
    assert run['state'] == 'failed' and run['pid'] > 0 and run['video_sha256'] == ex.file_sha256(video)
    assert run['advice'] == [] and run['validated_for_coaching'] is False
    assert not (out / 'result.json').exists()


@pytest.mark.parametrize('failure_errno', [1, 13])
def test_network_denial_guard_requires_both_families(monkeypatch, failure_errno):
    calls = []

    def denied(family, kind):
        calls.append(family)
        raise OSError(failure_errno, 'denied by fixture')

    monkeypatch.setattr(ex.socket, 'socket', denied)
    ex.require_network_denial()
    assert calls == [ex.socket.AF_INET, ex.socket.AF_INET6]


@pytest.mark.parametrize('failure_errno', [111, 61, 110, 97])
def test_connection_refused_timeout_or_unsupported_is_not_denial(monkeypatch, failure_errno):
    def not_denied(*args):
        raise OSError(failure_errno, 'not proven denied')
    monkeypatch.setattr(ex.socket, 'socket', not_denied)
    with pytest.raises(ValueError, match='unverified'):
        ex.require_network_denial()


def test_successful_probe_is_refused_and_socket_closed(monkeypatch):
    calls = []
    connection = SimpleNamespace(settimeout=lambda _: None, connect=lambda _: None,
                                 close=lambda: calls.append('closed'))
    monkeypatch.setattr(ex.socket, 'socket', lambda *args: connection)
    with pytest.raises(ValueError, match='remains available'):
        ex.require_network_denial()
    assert calls == ['closed']


def test_forged_model_and_manifest_with_same_url_cannot_rebind_registry(tmp_path, monkeypatch):
    model, manifest = model_fixture(tmp_path, monkeypatch)
    model.write_bytes(b'a different model')
    doc = json.loads(manifest.read_text())
    doc.update(sha256=ex.file_sha256(model), size_bytes=model.stat().st_size)
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='independently pinned model bytes'):
        ex.validate_model(model, manifest)


def test_license_policy_cannot_be_replaced_by_truthy_claim(tmp_path, monkeypatch):
    model, manifest = model_fixture(tmp_path, monkeypatch)
    doc = json.loads(manifest.read_text())
    doc['license_evidence'] = {'license': 'invented unrestricted rights'}
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='offline-privacy policy'):
        ex.validate_model(model, manifest)


def test_failure_publishing_final_ledger_removes_completed_result(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(ex, 'LOCK', tmp_path / 'active.lock')
    monkeypatch.setattr(ex, 'runtime_identity', lambda: {'fixture': 'software-only'})
    monkeypatch.setattr(ex, 'validate_model', lambda *args: {'fixture': 'software-only'})
    monkeypatch.setattr(ex, 'require_network_denial', lambda: None)
    clocks = video_clock([(5000, 1, 1000)])
    geometry = dict(width=640, height=360)
    monkeypatch.setattr(ex, 'inspect_video', lambda *args: (clocks, geometry))
    monkeypatch.setattr(ex, '_infer_frames', lambda *args: ([{
        'pts_seconds': 5., **pose_observation([])}], .001))
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(__version__='fixture', getBuildInformation=lambda: 'fixture'))

    class Detector:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class BaseOptions:
        Delegate = SimpleNamespace(CPU='CPU')

        def __init__(self, **kwargs):
            pass

    mp = SimpleNamespace(__version__='fixture', tasks=SimpleNamespace(
        BaseOptions=BaseOptions, vision=SimpleNamespace(
            PoseLandmarkerOptions=lambda **kwargs: None, RunningMode=SimpleNamespace(VIDEO='VIDEO'),
            PoseLandmarker=SimpleNamespace(create_from_options=lambda _: Detector()))))
    monkeypatch.setitem(sys.modules, 'mediapipe', mp)
    original_publish = ex.publish_json

    def fail_completed_ledger(path, value, **kwargs):
        if path.name == 'run.json' and value['state'] == 'completed':
            raise OSError('fixture final ledger failure')
        original_publish(path, value, **kwargs)

    monkeypatch.setattr(ex, 'publish_json', fail_completed_ledger)
    video = tmp_path / 'fixture.mp4'
    video.write_bytes(b'synthetic')
    out = tmp_path / 'attempt'
    with pytest.raises(OSError, match='final ledger failure'):
        ex.extract_pose(video=video, model='fixture', model_manifest='fixture', output=out, **context())
    assert json.loads((out / 'run.json').read_text())['state'] == 'failed'
    assert not (out / 'result.json').exists()


def test_multiple_pose_anywhere_suppresses_whole_spatial_sequence():
    result = {'schema_version': 1, 'state': 'completed', 'context': validate_context(**context()),
              'identity_ambiguous': False, 'video': {'width': 640, 'height': 360},
              'frames': [dict(pts_seconds=0., **pose_observation([[point() for _ in range(33)]])),
                         dict(pts_seconds=.03, **pose_observation([[point() for _ in range(33)],
                                                                [point() for _ in range(33)]]))]}
    with pytest.raises(ValueError, match='identity ambiguous'):
        as_pose_sequence(result)
