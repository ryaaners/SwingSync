"""Reference tooling software checks; synthetic fixtures are not pose accuracy."""
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from backend.swingsync.research import pose_evidence as evidence


def frame(index, *, status='single_pose'):
    return {'frame_index': index, 'pts': index * 1001, 'time_base_numerator': 1,
            'time_base_denominator': 30000, 'pts_seconds': index * 1001 / 30000,
            'api_timestamp_ms': round(index * 1001 / 30), 'status': status, 'raw_poses': [],
            'landmarks': [[.25, .5, 0., .9] for _ in range(33)] if status == 'single_pose'
                         else [[None] * 4 for _ in range(33)]}


def artifact():
    return SimpleNamespace(run={}, result={'schema_version': 1, 'state': 'completed',
        'video_sha256': 'a'*64, 'context': {'camera_view': 'face-on', 'handedness': 'right', 'mirrored': False,
            'rights_provenance': '</script><img src=x onerror=alert(1)>'},
        'video': {'width': 80, 'height': 40}, 'frames': [frame(i) for i in range(138)]},
        run_sha256='b'*64, result_sha256='c'*64, video_path=Path('synthetic.mp4'))


def template():
    return {'schema_version': 1, 'study_id': 'pose-validation-v1', 'coordinate_system': 'image-edge-pixels-v1',
        'video_sha256': 'a'*64, 'run_sha256': 'b'*64, 'result_sha256': 'c'*64,
        'reviewer': {'id': '', 'notes': '', 'model_blinded': False, 'independent': False},
        'context': {'camera_view': 'face-on', 'handedness': 'right', 'mirrored': False},
        'frames': [{'frame_index': i, 'pts_seconds': i * 1001 / 30000, 'width': 80, 'height': 40,
                    'landmarks': [{'landmark_id': j, 'status': 'unreviewed', 'x_px': None, 'y_px': None}
                                  for j in range(33)]} for i in evidence.REFERENCE_INDICES]}


@pytest.mark.parametrize('indices', [[1, 0], [0, 0], [True], [-1], [138], [], list(range(13)), [1.5]])
def test_frame_selection_rejects_invalid_or_cherry_picked_order(indices):
    with pytest.raises(ValueError):
        evidence.validate_indices(indices, 138, 80, 40)


def test_image_pixel_budget_and_indices_use_actual_frame_numbers():
    assert evidence.validate_indices([0, 37, 68, 87, 137], 138, 80, 40) == evidence.REFERENCE_INDICES
    with pytest.raises(ValueError, match='pixel'):
        evidence.validate_indices([0, 1], 2, 3840, 2160)


def test_clean_page_has_no_model_predictions_and_unchecked_attestations():
    images = {str(i): {'clean': 'CLEAN_FRAME', 'overlay': 'MODEL_OVERLAY_SECRET'} for i in evidence.REFERENCE_INDICES}
    page = evidence.annotation_page(template(), images)
    assert 'MODEL_OVERLAY_SECRET' not in page and 'raw_poses' not in page
    assert 'CLEAN_FRAME' in page and '"x_px":null' in page
    assert '"model_blinded":false' in page and '"independent":false' in page
    assert 'id="blinded" type="checkbox">' in page and 'id="independent" type="checkbox">' in page
    assert "requestToken!==loadToken" in page  # stale image loads cannot change current-frame correspondence
    assert "rect.width" in page and 'canvas.width' in page
    assert 'connect-src &#x27;none&#x27;' in page


def test_script_serialization_escapes_html_breakouts():
    value = {'id': '</script><script>alert(1)</script>&'}
    encoded = evidence._json(value)
    assert '</script>' not in encoded and '<script>' not in encoded
    assert json.loads(encoded) == value


def test_diagnostic_page_escapes_operator_metadata():
    images = {str(i): {'clean': 'CLEAN', 'overlay': 'OVERLAY'} for i in evidence.REFERENCE_INDICES}
    page = evidence.review_page(artifact(), evidence.REFERENCE_INDICES, images,
                               {'status': 'unavailable', 'reason': 'No manually supplied events'})
    assert '<img src=x' not in page
    assert '&lt;/script&gt;' in page and 'OVERLAY' in page
    assert 'No manually supplied events' in page and 'unvalidated' in page


def manual_doc(events, **updates):
    return {'schema_version': 1, 'video_sha256': 'a'*64, 'events': events} | updates


def test_absent_manual_events_do_not_generate_labels_or_measurements():
    result = evidence.manual_measurements(artifact(), None)
    assert result['measurements'] == [] and result['advice'] == []


@pytest.mark.parametrize('events', [
    [{'name': 'Address', 'frame_index': 0, 'source': 'predicted'}],
    [{'name': 'Top', 'frame_index': 0, 'source': 'manual'}, {'name': 'Address', 'frame_index': 1, 'source': 'manual'}],
    [{'name': 'Address', 'frame_index': True, 'source': 'manual'}],
    [{'name': 'Address', 'frame_index': 0, 'source': 'manual'}, {'name': 'Top', 'frame_index': 0, 'source': 'manual'}],
])
def test_manual_annotation_contract_refuses_predicted_reversed_or_reused_frames(events):
    with pytest.raises(ValueError, match='Manual marks'):
        evidence.manual_measurements(artifact(), manual_doc(events))


def test_manual_marks_bound_to_source_bytes():
    with pytest.raises(ValueError, match='source SHA'):
        evidence.manual_measurements(artifact(), manual_doc([], video_sha256='d'*64))


def test_output_bundle_refuses_different_selection_before_native_work(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, '_assert_current', lambda _: None)
    monkeypatch.setattr(evidence, 'render_frames_bounded', lambda *_: pytest.fail('No native work expected'))
    with pytest.raises(ValueError, match='prospectively fixed'):
        evidence.build_pose_evidence(artifact(), [0, 37], tmp_path/'out')


def fake_bundle(monkeypatch):
    monkeypatch.setattr(evidence, '_assert_current', lambda _: None)
    monkeypatch.setattr(evidence, '_reference_template', lambda *_: template())
    monkeypatch.setattr(evidence, 'render_frames_bounded', lambda *_: {
        str(i): {'clean': 'CLEAN', 'overlay': 'OVERLAY'} for i in evidence.REFERENCE_INDICES})


def test_bundle_hashes_pages_and_refuses_existing_output(tmp_path, monkeypatch):
    fake_bundle(monkeypatch)
    output = tmp_path/'review'
    manifest = evidence.build_pose_evidence(artifact(), list(evidence.REFERENCE_INDICES), output)
    assert manifest['state'] == 'completed' and manifest['advice'] == []
    assert not manifest['validated_for_coaching'] and not manifest['accuracy_evaluation_completed']
    assert set(manifest['files']) == {'review.html', 'reference.html', 'reference-template.json'}
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == digest
    before = (output/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        evidence.build_pose_evidence(artifact(), list(evidence.REFERENCE_INDICES), output)
    assert (output/'manifest.json').read_bytes() == before


def test_drift_before_publication_removes_incomplete_bundle(tmp_path, monkeypatch):
    fake_bundle(monkeypatch)
    calls = []
    def drift(_):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError('Fixture provenance drift')
    monkeypatch.setattr(evidence, '_assert_current', drift)
    output = tmp_path/'review'
    with pytest.raises(ValueError, match='provenance drift'):
        evidence.build_pose_evidence(artifact(), list(evidence.REFERENCE_INDICES), output)
    assert not output.exists()


def test_full_raster_overlay_uses_width_height_not_minus_one_and_keeps_missing_blank():
    Image = pytest.importorskip('PIL.Image')
    image = Image.new('RGB', (80, 40), (0, 0, 0))
    observed = frame(0)
    rendered = evidence._overlay(image, observed)
    assert rendered.size == (80, 40) and rendered.getpixel((20, 20)) != (0, 0, 0)
    missing = evidence._overlay(image, frame(0, status='no_pose'))
    assert missing.tobytes() == image.tobytes()


def test_canonical_child_decode_preserves_source_pixels_and_actual_pts(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    video = tmp_path/'native.mp4'
    with av.open(str(video), 'w') as output:
        stream = output.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 80, 40, 'yuv420p'
        for i in range(2):
            value = av.VideoFrame.from_ndarray(np.full((40,80,3), i*80, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(value):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(video)) as source:
        native = list(source.decode(video=0))
    records = []
    for i, f in enumerate(native):
        records.append(frame(i, status='no_pose') | {'pts': f.pts,
            'time_base_numerator': f.time_base.numerator, 'time_base_denominator': f.time_base.denominator,
            'pts_seconds': float(f.pts*f.time_base)})
    request = {'video': str(video), 'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
               'geometry': {'width':80, 'height':40}, 'frames':records, 'indices':[0,1]}
    images = evidence._decode_request(request)
    for i, f in enumerate(native):
        clean = Image.open(io.BytesIO(base64.b64decode(images[str(i)]['clean'])))
        assert np.array_equal(np.asarray(clean), f.to_ndarray(format='rgb24'))
        assert images[str(i)]['clean'] == images[str(i)]['overlay']
    request['frames'][1]['pts'] += 1
    with pytest.raises(ValueError, match='PTS/raster'):
        evidence._decode_request(request)


def test_malformed_extra_raw_landmark_is_shown_without_inventing_anatomical_id():
    fixture = artifact()
    row = fixture.result['frames'][0]
    row.update(status='malformed_pose', landmarks=[[None]*4 for _ in range(33)],
               raw_poses=[[{'x':None,'y':None,'visibility':None,'presence':None} for _ in range(34)]])
    images = {str(i): {'clean':'CLEAN','overlay':'UNCHANGED'} for i in evidence.REFERENCE_INDICES}
    page = evidence.review_page(fixture, evidence.REFERENCE_INDICES, images,
                               {'status':'unavailable','reason':'No manually supplied events'})
    assert 'Unknown landmark index 33' in page and 'malformed_pose' in page


@pytest.mark.parametrize('changed', ['experiments/pose-validation-v1.json',
                                   'backend/swingsync/research/pose_evidence.py'])
def test_protocol_or_renderer_drift_during_render_removes_derived_bundle(tmp_path, monkeypatch, changed):
    fake_bundle(monkeypatch)
    source_root = tmp_path/'source'
    for relative in evidence.DERIVATION_FILES:
        path = source_root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic source identity fixture')
    monkeypatch.setattr(evidence, 'ROOT', source_root)
    def changed_during_decode(*_):
        (source_root/changed).write_text('Changed after pre-render snapshot')
        return {str(i): {'clean':'CLEAN','overlay':'OVERLAY'} for i in evidence.REFERENCE_INDICES}
    monkeypatch.setattr(evidence, 'render_frames_bounded', changed_during_decode)
    output = tmp_path/'review'
    with pytest.raises(ValueError, match='source/protocol drift'):
        evidence.build_pose_evidence(artifact(), list(evidence.REFERENCE_INDICES), output)
    assert not output.exists()


def test_derived_manifest_binds_exact_protocol_and_all_renderer_source_bytes(tmp_path, monkeypatch):
    fake_bundle(monkeypatch)
    output = tmp_path/'review'
    manifest = evidence.build_pose_evidence(artifact(), list(evidence.REFERENCE_INDICES), output)
    expected = {name: hashlib.sha256((evidence.ROOT/name).read_bytes()).hexdigest()
                for name in evidence.DERIVATION_FILES}
    assert {'backend/swingsync/movement/measurements.py',
            'backend/swingsync/movement/contracts.py',
            'backend/swingsync/pose_runtime/contracts.py',
            'uv.lock'} <= set(manifest['derivation_source_sha256'])
    assert manifest['derivation_source_sha256'] == expected
    assert manifest['protocol_sha256'] == expected['experiments/pose-validation-v1.json']
