"""Synthetic draft fixtures validate handoff software, never model accuracy."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from types import SimpleNamespace

import av
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.swingsync.app import create_app
from backend.swingsync.annotations import AnnotationUpdate, apply_annotations
from backend.swingsync.proposal_handoff import (CHECKPOINT_SHA, EVENT_NAMES, FLAGS, MAX_DRAFT_BYTES,
                                               PROTOCOL_SHA, export_draft, parse_document, preview_draft,
                                               pts_digest)
from test_application import send_video, wait_for_result


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(doc):
    raw = (json.dumps(doc, indent=2, allow_nan=False)+'\n').encode()
    return raw, digest(raw)


def document(item):
    info = item['video']
    return {'schema_version': 1, 'kind': 'swingsync-research-event-draft', 'video_sha256': info['sha256'],
            'frame_count': info['frame_count'], 'native_pts_sha256': pts_digest(info['source_timestamps_seconds']),
            'metadata': {k: item['metadata'][k] for k in ('camera_view', 'handedness', 'club', 'mirrored')},
            'provenance': {'run_sha256': 'a'*64, 'result_sha256': 'b'*64, 'arrays_sha256': 'c'*64,
                           'protocol_sha256': PROTOCOL_SHA, 'checkpoint_sha256': CHECKPOINT_SHA,
                           'head': 'temporal', 'context': 'reference-64-reset',
                           'handoff_source_sha256': {'backend/swingsync/proposal_handoff.py': 'd'*64,
                                                     'scripts/export_event_draft.py': 'e'*64}},
            'events': [{'name': name, 'frame_index': i, 'source_timestamp_seconds': info['source_timestamps_seconds'][i],
                        'score': .2, 'background_probability': .7} for i, name in enumerate(EVENT_NAMES)],
            'strict_order': True, 'duplicate_peak_count': 0, 'flags': {k: False for k in FLAGS}, 'advice': []}


@pytest.fixture
def item():
    native = [5.0+i*.04 for i in range(9)]
    return {'id': 'software-only', 'status': 'completed', 'annotation_revision': 0, 'annotation_history': [],
            'events': [], 'measurements': [], 'metadata': {'camera_view': 'face-on', 'handedness': 'right', 'club': '7 iron', 'mirrored': False},
            'video': {'sha256': '0'*64, 'frame_count': 9, 'source_start_seconds': native[0],
                      'source_timestamps_seconds': native, 'timestamps_seconds': [v-native[0] for v in native]}}


def test_preview_keeps_opaque_bytes_and_never_mutates_or_repairs(item):
    doc = document(item)
    original = deepcopy(item)
    doc['events'][0]['frame_index'] = 8
    doc['events'][0]['source_timestamp_seconds'] = item['video']['source_timestamps_seconds'][8]
    doc['strict_order'] = False
    raw, sha = encoded(doc)
    preview = preview_draft(item, raw, sha)
    assert preview['document_json'].encode() == raw
    assert preview['expected_sha256'] == sha and preview['strict_order'] is False
    assert preview['events'][0]['frame_index'] == 8
    assert 'uncalibrated' in preview['warning'] and item == original
    with pytest.raises(ValueError, match='swing-event order'):
        apply_annotations(item, AnnotationUpdate.model_validate({'expected_revision': 0, 'events': preview['events'],
                          'research_draft': {'document_json': raw.decode(), 'expected_sha256': sha, 'confirmed': True}}))


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(schema_version=True), lambda d: d.update(frame_count=True),
    lambda d: d.update(extra='unknown'), lambda d: d.update(strict_order=1),
    lambda d: d.update(duplicate_peak_count=False), lambda d: d['events'][0].update(frame_index=True),
    lambda d: d['events'][0].update(name='Finish'), lambda d: d['events'][0].update(score=1.01),
    lambda d: d['events'][0].update(source_timestamp_seconds=False), lambda d: d['events'][0].update(score='0.2'),
    lambda d: d['events'].pop(), lambda d: d['metadata'].update(mirrored=0),
    lambda d: d['metadata'].update(camera_view='side'), lambda d: d['flags'].update(normal_model_available=True),
    lambda d: d['flags'].update(physical_timing_validated=0), lambda d: d.update(advice=['practice']),
    lambda d: d['provenance'].update(head='frame'), lambda d: d['provenance'].update(protocol_sha256='f'*64),
    lambda d: d['provenance'].update(checkpoint_sha256='f'*64), lambda d: d['provenance'].update(run_sha256='no'),
    lambda d: d['provenance'].update(extra='unknown'), lambda d: d.update(native_pts_sha256='bad'),
])
def test_strict_document_semantics(item, mutation):
    doc = document(item)
    mutation(doc)
    with pytest.raises(ValueError):
        parse_document(*encoded(doc))


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(video_sha256='9'*64), lambda d: d.update(native_pts_sha256='9'*64),
    lambda d: d['metadata'].update(club='driver'), lambda d: d['metadata'].update(handedness='left'),
    lambda d: d['metadata'].update(camera_view='down-the-line'),
    lambda d: d['events'][0].update(source_timestamp_seconds=5.001),
])
def test_wrong_source_time_metadata_refused(item, mutation):
    doc = document(item)
    mutation(doc)
    with pytest.raises(ValueError):
        preview_draft(item, *encoded(doc))


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]', b'\xff', b'{', b'['*2000+b']'*2000,
                                b' '* (MAX_DRAFT_BYTES+1)])
def test_bounded_malformed_duplicate_nonfinite_depth(item, raw):
    with pytest.raises(ValueError):
        preview_draft(item, raw, digest(raw))


def test_checksum_and_explicit_confirmation(item):
    raw, sha = encoded(document(item))
    with pytest.raises(ValueError, match='checksum'):
        preview_draft(item, raw+b' ', sha)
    request = {'expected_revision': 0, 'events': [{'name': 'Address', 'frame_index': 0}],
               'research_draft': {'document_json': raw.decode(), 'expected_sha256': sha, 'confirmed': False}}
    with pytest.raises(ValueError, match='Explicit'):
        apply_annotations(item, AnnotationUpdate.model_validate(request))
    request['research_draft']['confirmed'] = 'true'
    with pytest.raises(ValueError):
        AnnotationUpdate.model_validate(request)


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / 'software-only.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 64, 'yuv420p'
        stream.codec_context.time_base = Fraction(1, 1000)
        for i, pts in enumerate([5000, 5040, 5100, 5180, 5300, 5480, 5700, 5830, 5940]):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i*20, dtype=np.uint8), format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_api_preview_confirm_correction_conflict_clear_restart_report(tmp_path, recording):
    storage = tmp_path / 'storage'
    with TestClient(create_app(storage)) as client:
        response = send_video(client, recording.read_bytes())
        item = wait_for_result(client, response.json()['id'])
        ident = item['id']
        raw, sha = encoded(document(item))
        endpoint = f'/api/analyses/{ident}/research-draft'
        preview = client.post(endpoint, data={'expected_sha256': sha}, files={'file': ('draft.json', raw)})
        assert preview.status_code == 200, preview.text
        assert client.get(f'/api/analyses/{ident}').json() == item
        payload = {'expected_revision': 0, 'events': preview.json()['events'],
                   'research_draft': {'document_json': preview.json()['document_json'], 'expected_sha256': sha, 'confirmed': False}}
        assert client.put(f'/api/analyses/{ident}/events', json=payload).status_code == 422
        payload['research_draft']['confirmed'] = True
        # Correct Finish, retaining the original proposal in immutable revision history.
        payload['events'][-1]['frame_index'] = 8
        saved = client.put(f'/api/analyses/{ident}/events', json=payload)
        assert saved.status_code == 200, saved.text
        result = saved.json()
        assert result['annotation_revision'] == 1 and result['model']['status'] == 'unavailable'
        origin = result['annotation_history'][0]['research_origin']
        assert origin['original_events'][-1]['frame_index'] == 7 and origin['confirmed_events'][-1]['frame_index'] == 8
        assert origin['draft_sha256'] == sha and origin['initial_confirmation_explicit'] is True
        assert all(e['source'] == 'manual' and e['research_draft_sha256'] == sha for e in result['events'])
        assert len(result['measurements']) == 3 and result['observations'] == result['advice'] == []
        assert client.put(f'/api/analyses/{ident}/events', json=payload).status_code == 409
        cleared = client.put(f'/api/analyses/{ident}/events', json={'expected_revision': 1, 'events': []}).json()
        assert cleared['research_origin']['confirmed_events'] == []
        assert cleared['annotation_history'][0]['research_origin'] == origin
        report = client.get(f'/api/analyses/{ident}/report').text
        assert 'Research-assisted manual review provenance' in report and sha in report
        assert 'Original proposal frames' in report and 'No automatic inference was performed by the app' in report
        assert 'not model estimates' not in report
    with TestClient(create_app(storage)) as client:
        result = client.get(f'/api/analyses/{ident}').json()
        assert result['annotation_revision'] == 2 and result['annotation_history'][0]['research_origin'] == origin
        assert result['model']['status'] == 'unavailable' and result['measurements'] == []


def test_api_wrong_file_oversize_wrong_source_depth_and_changed_video(tmp_path, recording):
    storage = tmp_path / 'storage'
    with TestClient(create_app(storage)) as client:
        item = wait_for_result(client, send_video(client, recording.read_bytes()).json()['id'])
        ident = item['id']
        endpoint = f'/api/analyses/{ident}/research-draft'
        raw, sha = encoded(document(item))
        assert client.post(endpoint, data={'expected_sha256': '0'*64}, files={'file': ('draft.json', raw)}).status_code == 422
        oversized = b' '* (MAX_DRAFT_BYTES+1)
        assert client.post(endpoint, data={'expected_sha256': digest(oversized)}, files={'file': ('draft.json', oversized)}).status_code == 413
        nested = b'['*2000+b']'*2000
        assert client.post(endpoint, data={'expected_sha256': digest(nested)}, files={'file': ('draft.json', nested)}).status_code == 422
        wrong = document(item)
        wrong['video_sha256'] = '9'*64
        wrongraw, wrongsha = encoded(wrong)
        assert client.post(endpoint, data={'expected_sha256': wrongsha}, files={'file': ('draft.json', wrongraw)}).status_code == 422
        from backend.swingsync.store import Store
        saved = Store(storage/'analyses.sqlite3').get(ident)
        (storage/saved['file_name']).write_bytes(b'changed')
        assert client.post(endpoint, data={'expected_sha256': sha}, files={'file': ('draft.json', raw)}).status_code == 409


def test_export_exclusive_and_publication_drift(item, tmp_path, monkeypatch):
    from backend.swingsync.research import event_proposal_report as report
    doc = document(item)
    events = [{**e, 'pts_seconds': e['source_timestamp_seconds']} for e in doc['events']]
    evidence = SimpleNamespace(run={'result_sha256': 'b'*64}, run_sha256='a'*64, arrays_sha256='c'*64,
        result={'heads': {'temporal': {'events': events, 'strict_order': True, 'duplicate_peak_count': 0}},
                'video_sha256': item['video']['sha256'], 'frames': [{'pts_seconds': t} for t in item['video']['source_timestamps_seconds']],
                'context': item['metadata'], 'identity': {'protocol_sha256': PROTOCOL_SHA,
                    'selected_heads': {'temporal': {'checkpoint_sha256': CHECKPOINT_SHA}}}})
    monkeypatch.setattr(report, 'assert_unchanged', lambda _: None)
    target = tmp_path/'draft.json'
    result = export_draft(evidence, target)
    assert digest(target.read_bytes()) == result['sha256']
    assert preview_draft(item, target.read_bytes(), result['sha256'])['strict_order'] is True
    with pytest.raises(FileExistsError):
        export_draft(evidence, target)
    assert target.exists()
    new = tmp_path/'drift.json'
    calls = []
    def drift(_):
        calls.append(1)
        if new.exists():
            raise ValueError('changed after publication')
    monkeypatch.setattr(report, 'assert_unchanged', drift)
    with pytest.raises(ValueError, match='publication'):
        export_draft(evidence, new)
    assert not new.exists() and len(calls) == 3


@pytest.mark.parametrize('field', ['score', 'source_timestamp_seconds', 'background_probability'])
def test_api_huge_numeric_values_return422_without_mutation(tmp_path, recording, field):
    with TestClient(create_app(tmp_path/'storage')) as client:
        item = wait_for_result(client, send_video(client, recording.read_bytes()).json()['id'])
        ident = item['id']
        doc = document(item)
        doc['events'][0][field] = 10**309
        raw, sha = encoded(doc)
        response = client.post(f'/api/analyses/{ident}/research-draft',
                               data={'expected_sha256': sha}, files={'file': ('draft.json', raw)})
        assert response.status_code == 422
        response = client.put(f'/api/analyses/{ident}/events', json={'expected_revision': 0, 'events': [],
            'research_draft': {'document_json': raw.decode(), 'expected_sha256': sha, 'confirmed': True}})
        assert response.status_code == 422
        assert client.get(f'/api/analyses/{ident}').json() == item


@pytest.mark.parametrize('source_name', ['proposal_handoff.py', 'export_event_draft.py'])
def test_export_rejects_derivation_source_drift_after_publication(item, tmp_path, monkeypatch, source_name):
    from pathlib import Path
    from backend.swingsync.research import event_proposal_report as report
    doc = document(item)
    evidence = SimpleNamespace(run={'result_sha256': 'b'*64}, run_sha256='a'*64, arrays_sha256='c'*64,
        result={'heads': {'temporal': {'events': [{**e, 'pts_seconds': e['source_timestamp_seconds']} for e in doc['events']],
                                      'strict_order': True, 'duplicate_peak_count': 0}},
                'video_sha256': item['video']['sha256'], 'frames': [{'pts_seconds': t} for t in item['video']['source_timestamps_seconds']],
                'context': item['metadata'], 'identity': {'protocol_sha256': PROTOCOL_SHA,
                    'selected_heads': {'temporal': {'checkpoint_sha256': CHECKPOINT_SHA}}}})
    monkeypatch.setattr(report, 'assert_unchanged', lambda _: None)
    output = tmp_path/'drift.json'
    original = Path.read_bytes
    def drift(path):
        raw = original(path)
        return raw+b'changed' if path.name == source_name and output.exists() else raw
    monkeypatch.setattr(Path, 'read_bytes', drift)
    with pytest.raises(ValueError, match='derivation source'):
        export_draft(evidence, output)
    assert not output.exists()
