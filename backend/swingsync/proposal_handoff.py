"""Checksum-bound research drafts; no inference or automatic annotation saving.

An independently trusted exporter checksum authenticates bytes, not a digital
signature or scientific validity. Importing a file never enables model serving.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from .movement.contracts import EventName

MAX_DRAFT_BYTES = 64 * 1024
EVENT_NAMES = tuple(event.value for event in EventName)
PROTOCOL_SHA = '30b270ab29362a7645598e65a953108a1b2582c06acec48fc9b68ab3f76f8caa'
CHECKPOINT_SHA = '910f6d06a253e8117e825f5d625a1dcc11fe6db61344d169daac3fd973ceb5c3'
FLAGS = ('normal_model_available', 'validated_for_serving', 'accuracy_metrics_computed',
         'domain_compatibility_validated', 'physical_timing_validated', 'scores_are_calibrated',
         'order_constrained_decoding', 'outcome_integration_permitted')
WARNING = ('Unvalidated research proposals, not confirmed events. Review every canonical frame and correct '
           'the draft before explicitly confirming it. Scores are uncalibrated; background may exceed '
           'event scores. No accuracy, physical timing, pose, shot prediction or coaching is validated.')


class ResearchDraftInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_json: StrictStr = Field(max_length=MAX_DRAFT_BYTES)
    expected_sha256: StrictStr = Field(pattern=r'^[0-9a-f]{64}$')
    confirmed: StrictBool


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def pts_digest(values):
    return sha_bytes(json.dumps(values, separators=(',', ':'), allow_nan=False).encode('utf-8'))


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError('Research draft fields differ from the supported schema.')


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Research draft requires lowercase SHA-256 fingerprints.')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate research draft JSON key.')
        result[key] = value
    return result


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _invalid(_):
    raise ValueError('Nonfinite research draft JSON is forbidden.')


def parse_document(raw, expected_sha256):
    _sha(expected_sha256)
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_DRAFT_BYTES:
        raise ValueError('Research draft must contain at most64KiB of UTF-8 JSON.')
    if sha_bytes(raw) != expected_sha256:
        raise ValueError('Research draft checksum differs from the independently recorded exporter checksum.')
    try:
        text = raw.decode('utf-8')
        doc = json.loads(text, object_pairs_hook=_unique, parse_constant=_invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Research draft is not supported UTF-8 JSON.') from exc
    _keys(doc, ('schema_version', 'kind', 'video_sha256', 'frame_count', 'native_pts_sha256',
                'metadata', 'provenance', 'events', 'strict_order', 'duplicate_peak_count', 'flags', 'advice'))
    if type(doc['schema_version']) is not int or doc['schema_version'] != 1 or doc['kind'] != 'swingsync-research-event-draft':
        raise ValueError('Unsupported research draft version or kind.')
    for key in ('video_sha256', 'native_pts_sha256'):
        _sha(doc[key])
    if type(doc['frame_count']) is not int or not 2 <= doc['frame_count'] <= 3600:
        raise ValueError('Research draft frame count is invalid.')
    m = doc['metadata']
    _keys(m, ('camera_view', 'handedness', 'club', 'mirrored'))
    if (m['camera_view'] not in ('face-on', 'down-the-line') or m['handedness'] not in ('right', 'left')
            or type(m['club']) is not str or not 0 < len(m['club']) <= 80 or m['mirrored'] is not False):
        raise ValueError('Research draft operator metadata are unsupported.')
    p = doc['provenance']
    _keys(p, ('run_sha256', 'result_sha256', 'arrays_sha256', 'protocol_sha256', 'checkpoint_sha256', 'head', 'context', 'handoff_source_sha256'))
    for key in ('run_sha256', 'result_sha256', 'arrays_sha256', 'protocol_sha256', 'checkpoint_sha256'):
        _sha(p[key])
    _keys(p['handoff_source_sha256'], ('backend/swingsync/proposal_handoff.py', 'scripts/export_event_draft.py'))
    for digest in p['handoff_source_sha256'].values():
        _sha(digest)
    if (p['protocol_sha256'] != PROTOCOL_SHA or p['checkpoint_sha256'] != CHECKPOINT_SHA
            or p['head'] != 'temporal' or p['context'] != 'reference-64-reset'):
        raise ValueError('Research draft does not identify the fixed registered temporal proposal pipeline.')
    _keys(doc['flags'], FLAGS)
    if any(doc['flags'][key] is not False for key in FLAGS) or doc['advice'] != []:
        raise ValueError('Research draft cannot claim accuracy, serving, calibration, outcomes or coaching.')
    events = doc['events']
    if type(events) is not list or len(events) != 8:
        raise ValueError('Eight semantic event proposals are required.')
    for name, event in zip(EVENT_NAMES, events):
        _keys(event, ('name', 'frame_index', 'source_timestamp_seconds', 'score', 'background_probability'))
        if event['name'] != name or type(event['frame_index']) is not int or not 0 <= event['frame_index'] < doc['frame_count']:
            raise ValueError('Research event name/order/frame index is invalid.')
        for key in ('source_timestamp_seconds', 'score', 'background_probability'):
            value = event[key]
            if not _finite(value):
                raise ValueError('Research proposal values must be finite numbers.')
        if any(not 0 <= event[key] <= 1 for key in ('score', 'background_probability')):
            raise ValueError('Research proposal scores must lie within0..1.')
    indices = [e['frame_index'] for e in events]
    if (type(doc['strict_order']) is not bool
            or doc['strict_order'] != all(a < b for a, b in zip(indices, indices[1:]))
            or type(doc['duplicate_peak_count']) is not int
            or doc['duplicate_peak_count'] != 8-len(set(indices))):
        raise ValueError('Research proposal order diagnostics differ; no repair is permitted.')
    return text, doc


def preview_draft(item, raw, expected_sha256):
    from .annotations import validated_video_info
    if item.get('status') != 'completed':
        raise ValueError('Complete video inspection before importing research proposals.')
    text, doc = parse_document(raw, expected_sha256)
    info = validated_video_info(item.get('video'))
    if (doc['video_sha256'] != info['sha256'] or doc['frame_count'] != info['frame_count']
            or doc['native_pts_sha256'] != pts_digest(info['source_timestamps_seconds'])):
        raise ValueError('Research draft belongs to a different recording or native timestamp sequence.')
    if any(doc['metadata'][key] != item.get('metadata', {}).get(key) for key in doc['metadata']):
        raise ValueError('Research draft camera, handedness, club or mirroring differs from this upload.')
    if any(e['source_timestamp_seconds'] != info['source_timestamps_seconds'][e['frame_index']] for e in doc['events']):
        raise ValueError('Research proposal frame and native timestamp do not correspond.')
    return {'document_json': text, 'expected_sha256': expected_sha256,
            'events': [{'name': e['name'], 'frame_index': e['frame_index']} for e in doc['events']],
            'strict_order': doc['strict_order'], 'duplicate_peak_count': doc['duplicate_peak_count'], 'warning': WARNING}


def research_origin(item, supplied, confirmed_events):
    """Validate explicit initial confirmation; retain origin across later manual edits."""
    if supplied is not None:
        if supplied.confirmed is not True:
            raise ValueError('Explicit review confirmation is required before saving research-assisted marks.')
        preview_draft(item, supplied.document_json.encode('utf-8'), supplied.expected_sha256)
        _, doc = parse_document(supplied.document_json.encode('utf-8'), supplied.expected_sha256)
        origin = {'draft_sha256': supplied.expected_sha256, **doc['provenance'],
                  'original_events': [{key: e[key] for key in ('name', 'frame_index', 'source_timestamp_seconds')} for e in doc['events']],
                  'research_flags': deepcopy(doc['flags']), 'initial_confirmation_explicit': True}
        confirmation = 'explicit-research-review'
    else:
        origin = deepcopy(item.get('research_origin'))
        confirmation = 'subsequent-manual-edit'
    if origin is None:
        return None
    origin['confirmed_events'] = [{key: e[key] for key in ('name', 'frame_index', 'source_timestamp_seconds')} for e in confirmed_events]
    origin['revision_action'] = confirmation
    return origin


def export_draft(evidence, output):
    """Export only authenticated saved observations; no model forwards or fitted code."""
    from .research.event_proposal_report import assert_unchanged
    assert_unchanged(evidence)
    root = Path(__file__).resolve().parents[2]
    source_files = ('backend/swingsync/proposal_handoff.py', 'scripts/export_event_draft.py')
    source_hashes = {name: sha_bytes((root/name).read_bytes()) for name in source_files}
    def sources_unchanged():
        if {name: sha_bytes((root/name).read_bytes()) for name in source_files} != source_hashes:
            raise ValueError('Research handoff derivation source changed during publication.')
    result = evidence.result
    p = result['heads']['temporal']
    doc = {'schema_version': 1, 'kind': 'swingsync-research-event-draft',
           'video_sha256': result['video_sha256'], 'frame_count': len(result['frames']),
           'native_pts_sha256': pts_digest([f['pts_seconds'] for f in result['frames']]),
           'metadata': {key: result['context'][key] for key in ('camera_view', 'handedness', 'club', 'mirrored')},
           'provenance': {'run_sha256': evidence.run_sha256, 'result_sha256': evidence.run['result_sha256'],
                          'arrays_sha256': evidence.arrays_sha256, 'protocol_sha256': result['identity']['protocol_sha256'],
                          'checkpoint_sha256': result['identity']['selected_heads']['temporal']['checkpoint_sha256'],
                          'head': 'temporal', 'context': 'reference-64-reset',
                          'handoff_source_sha256': source_hashes},
           'events': [{'name': e['name'], 'frame_index': e['frame_index'], 'source_timestamp_seconds': e['pts_seconds'],
                       'score': e['score'], 'background_probability': e['background_probability']} for e in p['events']],
           'strict_order': p['strict_order'], 'duplicate_peak_count': p['duplicate_peak_count'],
           'flags': {key: False for key in FLAGS}, 'advice': []}
    raw = (json.dumps(doc, indent=2, allow_nan=False)+'\n').encode('utf-8')
    digest = sha_bytes(raw)
    parse_document(raw, digest)
    output = Path(output)
    created = False
    try:
        assert_unchanged(evidence)
        sources_unchanged()
        with output.open('xb') as stream:
            created = True
            stream.write(raw)
        assert_unchanged(evidence)
        sources_unchanged()
        if output.read_bytes() != raw:
            raise ValueError('Research handoff output changed during publication.')
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
    return {'path': str(output), 'sha256': digest, 'event_count': 8, 'warning': WARNING}
