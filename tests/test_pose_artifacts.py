"""Synthetic completed-ledger attacks; no landmark accuracy or native inference."""
from dataclasses import asdict
import fcntl
import json

import pytest

from backend.swingsync.movement import pose_artifacts as pa
from backend.swingsync.pose_runtime.contracts import pose_observation, validate_context, video_clock


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, 'LOCK', tmp_path / 'lock')
    video = tmp_path / 'fixture.mp4'
    video.write_bytes(b'software-only fake video; decoder mocked')
    geometry = dict(width=640, height=360, pixel_aspect='1/1',
                    display_transform='coded-upright-unrotated', crop=None, resize=None, frame_count=2)
    clocks = [asdict(c) for c in video_clock([(5000, 1, 1000), (5034, 1, 1000)])]
    monkeypatch.setattr(pa, '_inspect_bounded', lambda _: {'clocks': clocks, 'geometry': geometry})
    manifest = dict(schema_version=1, url=pa.MODEL_URL, sha256=pa.MODEL_SHA256,
                    size_bytes=pa.MODEL_SIZE, members=pa.MEMBERS, license_evidence=pa.LICENSE_EVIDENCE)
    run = dict(schema_version=1, state='completed', job_id='synthetic', created_at='fixture',
               completed_at='fixture', pid=1, video_sha256=pa.sha256(video), video=geometry,
               context=validate_context('face-on', 'right', True, True, 'synthetic software fixture', 'reencoded'),
               model=dict(manifest=manifest, bundle_sha256=pa.MODEL_SHA256, manifest_sha256='1'*64),
               runtime=dict(pose_lock_sha256=pa.sha256(pa.POSE_LOCK), delegate='CPU',
                            adapter_source_sha256={str(p.relative_to(pa.ROOT)): pa.sha256(p)
                                                   for p in pa._source_paths()},
                            installed_distributions={'mediapipe': '1.2.2', 'numpy': '2.2.5',
                                                     'opencv-contrib-python': '4.12.0.88', 'av': '16.0.1'}),
               config=dict(running_mode='VIDEO', num_poses=2, delegate='CPU',
                           min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
                           min_tracking_confidence=.5, presence_filter_floor=.5, output_segmentation_masks=False,
                           clock='exact-rational-relative-half-up-ms-v1',
                           limits=dict(bytes=150*1024*1024, frames=3600, seconds=30)),
               network_policy='OS deny-all network via supported launcher; IPv4/IPv6 denial guard passed',
               advice=[], validated_for_coaching=False, resume_supported=False,
               status_counts={'no_pose': 2})
    result = {**run, 'frames': [{**c, **pose_observation([])} for c in clocks],
              'identity_ambiguous': False, 'clock_origin_seconds': 5.,
              'api_clock_is_measurement_coordinate': False}
    run_path = tmp_path / 'run.json'

    def publish():
        result_path = tmp_path / 'result.json'
        result_path.write_text(json.dumps(result))
        run['result_sha256'] = pa.sha256(result_path)
        run_path.write_text(json.dumps(run))
        return run_path, pa.sha256(run_path)

    publish()
    return run, result, video, publish


def load(saved):
    _, _, video, publish = saved
    path, digest = publish()
    return pa.load_pose_artifact(path, expected_run_sha256=digest, video_path=video)


def test_completed_ledger_and_exact_pts_admitted_without_inference(saved):
    artifact = load(saved)
    assert artifact.result['frames'][0]['pts_seconds'] == 5.
    assert artifact.result['frames'][1]['api_timestamp_ms'] == 34
    pa.assert_artifact_unchanged(artifact)


def test_external_known_hash_required_before_decode(saved, monkeypatch):
    _, _, video, publish = saved
    path, _ = publish()
    monkeypatch.setattr(pa, '_inspect_bounded', lambda _: pytest.fail('Decoder must not start'))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        pa.load_pose_artifact(path, expected_run_sha256='0'*64, video_path=video)


@pytest.mark.parametrize('part,key,value', [
    ('run', 'state', 'running'), ('result', 'state', 'running'),
    ('result', 'identity_ambiguous', True), ('result', 'clock_origin_seconds', 0.),
    ('result', 'api_clock_is_measurement_coordinate', True),
    ('result', 'advice', ['invented advice']), ('run', 'validated_for_coaching', True),
])
def test_jointly_rehashed_invalid_state_still_rejected(saved, part, key, value):
    run, result, *_ = saved
    (run if part == 'run' else result)[key] = value
    with pytest.raises(ValueError):
        load(saved)


@pytest.mark.parametrize('field,value', [('pts', 5001), ('pts_seconds', 5.001),
                                        ('api_timestamp_ms', 1), ('frame_index', True),
                                        ('pts_seconds', True)])
def test_rehashed_source_clock_forgery_refused(saved, field, value):
    saved[1]['frames'][0][field] = value
    with pytest.raises(ValueError, match='PTS/API clock'):
        load(saved)


def test_tampered_filtered_points_cannot_disagree_with_raw(saved):
    saved[1]['frames'][0]['landmarks'][0] = [.1, .2, .3, .9]
    with pytest.raises(ValueError, match='raw/filtered'):
        load(saved)


def test_rehashed_pinned_model_forgery_refused(saved):
    saved[0]['model']['manifest']['sha256'] = '3'*64
    with pytest.raises(ValueError, match='model registration'):
        load(saved)


def test_missing_source_hash_entry_rejected(saved):
    saved[0]['runtime']['adapter_source_sha256'].pop('scripts/run_pose_offline.py')
    with pytest.raises(ValueError, match='source fingerprint'):
        load(saved)


def test_wrong_recorded_runtime_pin_refused(saved):
    saved[0]['runtime']['installed_distributions']['numpy'] = '1.26.4'
    with pytest.raises(ValueError, match='complete lock'):
        load(saved)


def test_media_and_in_memory_drift_refuse_derived_publication(saved):
    artifact = load(saved)
    artifact.result['frames'][0]['landmarks'][0][0] = .9
    with pytest.raises(ValueError, match='snapshot drift'):
        pa.assert_artifact_unchanged(artifact)
    artifact = load(saved)
    artifact.video_path.write_bytes(b'changed video')
    with pytest.raises(ValueError, match='video bytes changed'):
        pa.assert_artifact_unchanged(artifact)


def test_busy_shared_lock_refuses_decode(saved, monkeypatch):
    _, _, video, publish = saved
    path, digest = publish()
    monkeypatch.setattr(pa, '_inspect_bounded', lambda _: pytest.fail('Decoder must not start'))
    with pa.LOCK.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='shared lock'):
            pa.load_pose_artifact(path, expected_run_sha256=digest, video_path=video)


def test_json_duplicate_keys_and_nonfinite_rejected(tmp_path):
    path = tmp_path / 'invalid.json'
    for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
        path.write_text(raw)
        with pytest.raises(ValueError):
            pa._json(path)
