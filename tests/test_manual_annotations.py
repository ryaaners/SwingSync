"""Synthetic VFR media exercises annotation plumbing, not ML/coaching accuracy."""
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
import json
import struct
import subprocess
import threading
from types import SimpleNamespace
import zlib

import av
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.swingsync import app as application
from backend.swingsync.annotations import AnnotationUpdate
from backend.swingsync.frame_images import extract_frames, extract_frames_bounded, file_sha256
from backend.swingsync.store import Store
from backend.swingsync.video import inspect_video
from test_application import send_video, wait_for_result


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / 'manual-software.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=25)
        stream.width, stream.height, stream.pix_fmt = 96, 64, 'yuv420p'
        stream.codec_context.time_base = Fraction(1, 1000)
        for i, pts in enumerate([5000, 5040, 5100, 5180, 5300, 5480, 5700, 5830]):
            frame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), i * 28, dtype=np.uint8), format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def upload(client, recording):
    result = send_video(client, recording.read_bytes(), club='<script>unsafe</script>')
    assert result.status_code == 202
    item = wait_for_result(client, result.json()['id'])
    assert item['status'] == 'completed', item
    return item


def marks(revision=0, events=None):
    if events is None:
        events = [{'name': 'Address', 'frame_index': 0}, {'name': 'Top', 'frame_index': 3},
                  {'name': 'Impact', 'frame_index': 7}]
    return {'expected_revision': revision, 'events': events}


def decoded_png(content):
    """Independent simple PNG parser checks exact RGB pixel content."""
    assert content.startswith(b'\x89PNG\r\n\x1a\n')
    offset, compressed = 8, b''
    width = height = None
    while offset < len(content):
        length = struct.unpack('>I', content[offset:offset + 4])[0]
        kind = content[offset + 4:offset + 8]
        data = content[offset + 8:offset + 8 + length]
        if kind == b'IHDR':
            width, height = struct.unpack('>II', data[:8])
        elif kind == b'IDAT':
            compressed += data
        offset += length + 12
    scanlines = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(height, width * 3 + 1)
    assert np.all(scanlines[:, 0] == 0)
    return scanlines[:, 1:].reshape(height, width, 3)


def test_manual_annotations_exact_pts_recompute_history_restart_report_and_delete(tmp_path, recording):
    storage = tmp_path / 'storage'
    with TestClient(application.create_app(storage)) as client:
        item = upload(client, recording)
        ident = item['id']
        assert item['video']['sha256'] == file_sha256(recording)
        assert item['annotation_revision'] == 0 and item['annotation_history'] == []
        response = client.put(f'/api/analyses/{ident}/events', json=marks())
        assert response.status_code == 200, response.text
        first = response.json()
        assert first['annotation_revision'] == 1 and first['model']['status'] == 'unavailable'
        assert first['observations'] == first['advice'] == []
        assert [e['timestamp_seconds'] for e in first['events']] == pytest.approx([0, .18, .83])
        assert [e['source_timestamp_seconds'] for e in first['events']] == pytest.approx([5, 5.18, 5.83])
        assert all(e['source'] == 'manual' for e in first['events'])
        values = {m['name']: m for m in first['measurements']}
        assert values['backswing_duration_seconds']['value'] == pytest.approx(.18)
        assert values['downswing_duration_seconds']['value'] == pytest.approx(.65)
        assert values['backswing_downswing_ratio']['value'] == pytest.approx(.18 / .65)
        assert all(m['frame_indices'] and 'Unvalidated' in m['reliability'] and m['reference']
                   for m in values.values())
        assert values['backswing_downswing_ratio']['interval_seconds'] == pytest.approx([0, .83])
        corrected = marks(1)
        corrected['events'][1]['frame_index'] = 4
        latest = client.put(f'/api/analyses/{ident}/events', json=corrected).json()
        assert latest['annotation_revision'] == 2 and len(latest['annotation_history']) == 2
        assert latest['annotation_history'][0]['events'][1]['frame_index'] == 3
        assert latest['measurements'][0]['value'] == pytest.approx(.3)
        report = client.get(f'/api/analyses/{ident}/report')
        assert report.status_code == 200
        assert latest['video']['sha256'] in report.text and ident in report.text
        assert report.text.count('data:image/png;base64,') == 3
        assert 'Manual revision 1 (superseded)' in report.text and 'Manual revision 2 (current)' in report.text
        assert 'native PTS 5.300000' in report.text and 'coded decoded orientation' in report.text
        assert 'Slow motion or speed ramps' in report.text and 'No automatic event estimates' in report.text
        assert '<script>' not in report.text and '&lt;script&gt;' in report.text
    with TestClient(application.create_app(storage)) as client:
        assert client.get(f'/api/analyses/{ident}').json()['annotation_revision'] == 2
        assert client.delete(f'/api/analyses/{ident}').status_code == 204
        assert client.put(f'/api/analyses/{ident}/events', json=marks(2)).status_code == 404
        assert client.get(f'/api/analyses/{ident}/frames/0').status_code == 404
        assert client.get(f'/api/analyses/{ident}/report').status_code == 404
        assert client.get('/api/analyses').json() == []
        assert not list(storage.glob('*.mp4'))


def test_canonical_frame_pixels_headers_and_bounds(tmp_path, recording):
    with TestClient(application.create_app(tmp_path / 'storage')) as client:
        item = upload(client, recording)
        response = client.get(f'/api/analyses/{item["id"]}/frames/3')
        assert response.status_code == 200 and response.headers['content-type'] == 'image/png'
        assert response.headers['cache-control'] == 'no-store'
        assert response.headers['x-swingsync-frame-index'] == '3'
        assert float(response.headers['x-swingsync-timestamp-seconds']) == pytest.approx(.18)
        assert float(response.headers['x-swingsync-source-timestamp-seconds']) == pytest.approx(5.18)
        assert 'coded-frame' in response.headers['x-swingsync-orientation']
        with av.open(str(recording)) as container:
            expected = list(container.decode(video=0))[3].to_ndarray(format='rgb24')
        np.testing.assert_array_equal(decoded_png(response.content), expected)
        for index in ('-1', '8', 'true', 'not-an-index'):
            assert client.get(f'/api/analyses/{item["id"]}/frames/{index}').status_code == 422


@pytest.mark.parametrize('events', [
    [{'name': 'Top', 'frame_index': True}], [{'name': 'Top', 'frame_index': 3.0}],
    [{'name': 'Top', 'frame_index': '3'}], [{'name': 'Top', 'frame_index': -1}],
    [{'name': 'Top', 'frame_index': 8}], [{'name': 'Top', 'frame_index': float('inf')}],
    [{'name': 'Top', 'frame_index': float('-inf')}], [{'name': 'Top', 'frame_index': float('nan')}],
    [{'name': 'Unknown', 'frame_index': 1}], [{'name': 'Top', 'frame_index': 1, 'timestamp_seconds': .1}],
    [{'name': 'Top', 'frame_index': 2}, {'name': 'Top', 'frame_index': 3}],
    [{'name': 'Address', 'frame_index': 3}, {'name': 'Top', 'frame_index': 2}],
    [{'name': 'Address', 'frame_index': 3}, {'name': 'Top', 'frame_index': 3}],
])
def test_invalid_events_do_not_change_revision(tmp_path, recording, events):
    with TestClient(application.create_app(tmp_path / 'storage')) as client:
        item = upload(client, recording)
        # Raw JSON permits the nonfinite negative fixture to reach strict validation.
        response = client.put(f'/api/analyses/{item["id"]}/events',
                              content=json.dumps(marks(events=events)), headers={'Content-Type': 'application/json'})
        assert response.status_code == 422
        saved = client.get(f'/api/analyses/{item["id"]}').json()
        assert saved['annotation_revision'] == 0 and saved['events'] == []


def test_partial_clear_and_stale_revision_preserve_all_corrections(tmp_path, recording):
    with TestClient(application.create_app(tmp_path / 'storage')) as client:
        item = upload(client, recording)
        url = f'/api/analyses/{item["id"]}/events'
        result = client.put(url, json=marks(events=[{'name': 'Top', 'frame_index': 3}])).json()
        assert result['measurements'] == [] and result['annotation_revision'] == 1
        assert client.put(url, json=marks()).status_code == 409
        cleared = client.put(url, json=marks(1, [])).json()
        assert cleared['events'] == cleared['measurements'] == []
        assert cleared['annotation_revision'] == 2 and len(cleared['annotation_history']) == 2
        report = client.get(f'/api/analyses/{item["id"]}/report')
        assert 'Manual revision 1 (superseded)' in report.text
        assert 'Manual revision 2 (current)' in report.text and 'All event marks cleared' in report.text
        for revision in (True, 2.0, '2', -1):
            assert client.put(url, json=marks(revision, [])).status_code == 422


def test_atomic_revision_cas_and_no_deleted_row_resurrection(tmp_path, recording):
    storage = tmp_path / 'storage'
    with TestClient(application.create_app(storage)) as client:
        item = upload(client, recording)
        url = f'/api/analyses/{item["id"]}/events'
        with ThreadPoolExecutor(max_workers=2) as workers:
            responses = list(workers.map(lambda frame: client.put(url, json=marks(
                events=[{'name': 'Top', 'frame_index': frame}])), (2, 3)))
        assert sorted(r.status_code for r in responses) == [200, 409]
        saved = client.get(f'/api/analyses/{item["id"]}').json()
        assert saved['annotation_revision'] == 1 and len(saved['annotation_history']) == 1
        assert client.delete(f'/api/analyses/{item["id"]}').status_code == 204
        with pytest.raises(KeyError):
            client.app.state.store.update_annotations(item['id'], AnnotationUpdate(**marks(1, [])))
        assert Store(storage / 'analyses.sqlite3').get(item['id']) is None


def test_changed_media_or_saved_pts_and_legacy_evidence_fail_closed(tmp_path, recording):
    storage = tmp_path / 'storage'
    with TestClient(application.create_app(storage)) as client:
        item = upload(client, recording)
        ident = item['id']
        store = client.app.state.store
        original = store.get(ident)
        broken = deepcopy(original)
        broken['video'].pop('sha256')
        store.save(broken)
        assert client.get(f'/api/analyses/{ident}/frames/0').status_code == 409
        assert client.put(f'/api/analyses/{ident}/events', json=marks()).status_code == 409
        store.save(original)
        path = storage / original['file_name']
        path.write_bytes(path.read_bytes() + b'changed')
        assert client.get(f'/api/analyses/{ident}/frames/0').status_code == 409
        assert client.put(f'/api/analyses/{ident}/events', json=marks()).status_code == 409
    info = inspect_video(recording)
    info['sha256'] = file_sha256(recording)
    info['source_timestamps_seconds'][2] += .001
    with pytest.raises(ValueError, match='timestamps differ'):
        extract_frames(recording, info, [2])


def test_bounded_native_timeout_and_output_limit(monkeypatch, recording):
    from backend.swingsync import frame_images
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 45)
    monkeypatch.setattr(frame_images.subprocess, 'run', timeout)
    with pytest.raises(ValueError, match='45 seconds'):
        extract_frames_bounded(recording, {}, [0])
    monkeypatch.setattr(frame_images.subprocess, 'run', lambda *_, **kwargs:
                        SimpleNamespace(stdout='x' * (frame_images.MAX_OUTPUT_BYTES + 1)))
    with pytest.raises(ValueError, match='output limit'):
        extract_frames_bounded(recording, {}, [0])


def test_frame_decoder_concurrency_is_bounded_and_releases_slots(monkeypatch, tmp_path, recording):
    with TestClient(application.create_app(tmp_path / 'storage')) as client:
        item = upload(client, recording)
        url = f'/api/analyses/{item["id"]}/frames/0'
        ready, release = threading.Barrier(3), threading.Event()
        png = base64.b64encode(b'PNG-fixture-only').decode()
        def blocked(*_):
            ready.wait(timeout=5)
            assert release.wait(timeout=5)
            return {'0': png}
        monkeypatch.setattr(application, 'extract_frames_bounded', blocked)
        with ThreadPoolExecutor(max_workers=2) as workers:
            pending = [workers.submit(client.get, url) for _ in range(2)]
            ready.wait(timeout=5)
            try:
                response = client.get(url)
                assert response.status_code == 429 and response.headers['retry-after'] == '1'
            finally:
                release.set()
            assert [future.result().status_code for future in pending] == [200, 200]
        monkeypatch.setattr(application, 'extract_frames_bounded', lambda *_: {'0': png})
        assert client.get(url).status_code == 200


def test_canonical_png_resize_preserves_coded_aspect_without_rotation():
    from backend.swingsync.frame_images import _png
    pixels = np.zeros((120, 960, 3), dtype=np.uint8)
    pixels[:, :480, 0] = 255
    pixels[:, 480:, 2] = 255
    frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
    result = decoded_png(_png(frame))
    assert result.shape == (80, 640, 3)
    assert result[40, 100, 0] > 250 and result[40, 100, 2] < 5
    assert result[40, 540, 2] > 250 and result[40, 540, 0] < 5
