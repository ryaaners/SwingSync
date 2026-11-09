"""Strict manual-reference pilot contracts and descriptive 2D agreement.

This module never creates human labels, fits a filter, validates coaching or
promotes a model to app serving. Callers must authenticate the completed pose
artifact separately; self-consistent dictionaries are not proof of provenance.
"""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path


PROTOCOL_PATH = Path(__file__).resolve().parents[3] / 'experiments/pose-validation-v1.json'
STATES = {'visible', 'occluded', 'out_of_frame', 'unreviewed'}
COORDINATES = 'image-edge-pixels-v1'


def load_protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{label} has missing or unexpected fields')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _admit(result, run_sha256, result_sha256, protocol):
    """Bind this pilot to the registered artifact; no new source silently enters."""
    if protocol.get('schema_version') != 1 or protocol.get('coordinate_system') != COORDINATES:
        raise ValueError('Unsupported pose reference protocol')
    source = protocol['artifact']
    if (not _sha(run_sha256) or not _sha(result_sha256)
            or run_sha256 != source['run_sha256'] or result_sha256 != source['result_sha256']):
        raise ValueError('Pose run/result differs from registered pilot')
    if (result.get('schema_version') != 1 or result.get('state') != 'completed'
            or result.get('video_sha256') != source['video_sha256']):
        raise ValueError('Completed source-bound pose result required')
    context = {name: result['context'].get(name) for name in ('camera_view', 'handedness', 'mirrored')}
    if context != source['context']:
        raise ValueError('Pose context differs from registered pilot')
    if (result['video']['width'], result['video']['height']) != (source['width'], source['height']):
        raise ValueError('Pose raster differs from registered pilot')
    indices = source['frame_indices']
    if (not indices or any(type(i) is not int or i < 0 or i >= len(result['frames']) for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError('Registered frame indices must be ordered, unique and covered')
    if protocol.get('landmark_ids') != list(range(33)):
        raise ValueError('Protocol must retain all 33 anatomical landmark IDs')
    floor = protocol['visibility_floor']
    if not _finite(floor) or not 0 <= floor <= 1:
        raise ValueError('Finite registered visibility floor required')
    for index in indices:
        row = result['frames'][index]
        if row['frame_index'] != index or not _finite(row['pts_seconds']):
            raise ValueError('Registered frame mapping is malformed')
    return source, context


def make_reference_template(result, *, run_sha256, result_sha256, protocol=None):
    """Create only unreviewed slots, never reference coordinates or attestations."""
    protocol = load_protocol() if protocol is None else protocol
    source, context = _admit(result, run_sha256, result_sha256, protocol)
    return {
        'schema_version': 1, 'study_id': protocol['study_id'], 'coordinate_system': COORDINATES,
        'video_sha256': source['video_sha256'], 'run_sha256': run_sha256, 'result_sha256': result_sha256,
        'reviewer': {'id': '', 'notes': '', 'model_blinded': False, 'independent': False},
        'context': copy.deepcopy(context),
        'frames': [{'frame_index': index, 'pts_seconds': result['frames'][index]['pts_seconds'],
                    'width': source['width'], 'height': source['height'],
                    'landmarks': [{'landmark_id': joint, 'status': 'unreviewed', 'x_px': None, 'y_px': None}
                                  for joint in range(33)]} for index in source['frame_indices']],
    }


def validate_reference(reference, result, *, run_sha256, result_sha256, protocol=None, require_scoring=False):
    """Validate drafts or independently attested labels; never mutate either input."""
    protocol = load_protocol() if protocol is None else protocol
    source, context = _admit(result, run_sha256, result_sha256, protocol)
    _keys(reference, ('schema_version', 'study_id', 'coordinate_system', 'video_sha256', 'run_sha256',
                      'result_sha256', 'reviewer', 'context', 'frames'), 'Reference')
    if (type(reference['schema_version']) is not int or reference['schema_version'] != 1
            or reference['study_id'] != protocol['study_id'] or reference['coordinate_system'] != COORDINATES):
        raise ValueError('Reference schema/study/coordinate units differ')
    for field in ('video_sha256', 'run_sha256', 'result_sha256'):
        if not _sha(reference[field]) or reference[field] != source[field]:
            raise ValueError('Reference source/run/result SHA differs')
    _keys(reference['context'], ('camera_view', 'handedness', 'mirrored'), 'Context')
    if type(reference['context']['mirrored']) is not bool or reference['context'] != context:
        raise ValueError('Reference context differs')
    reviewer = reference['reviewer']
    _keys(reviewer, ('id', 'notes', 'model_blinded', 'independent'), 'Reviewer')
    if (not isinstance(reviewer['id'], str) or len(reviewer['id']) > 200
            or not isinstance(reviewer['notes'], str) or len(reviewer['notes']) > 4000
            or type(reviewer['model_blinded']) is not bool or type(reviewer['independent']) is not bool):
        raise ValueError('Reviewer identity/attestations malformed')
    if require_scoring and (not reviewer['id'].strip() or not reviewer['model_blinded'] or not reviewer['independent']):
        raise ValueError('Scoring requires manual independent/model-blinded attestations and reviewer identity')
    frames = reference['frames']
    if not isinstance(frames, list) or len(frames) != len(source['frame_indices']):
        raise ValueError('Every registered frame must remain explicit')
    visible = 0
    for frame, expected_index in zip(frames, source['frame_indices']):
        _keys(frame, ('frame_index', 'pts_seconds', 'width', 'height', 'landmarks'), 'Reference frame')
        if (type(frame['frame_index']) is not int or frame['frame_index'] != expected_index
                or type(frame['width']) is not int or frame['width'] != source['width']
                or type(frame['height']) is not int or frame['height'] != source['height']
                or not _finite(frame['pts_seconds'])
                or frame['pts_seconds'] != result['frames'][expected_index]['pts_seconds']):
            raise ValueError('Reference frame native PTS/raster/index differs')
        points = frame['landmarks']
        if not isinstance(points, list) or len(points) != 33:
            raise ValueError('All 33 anatomical landmark entries required')
        if [p.get('landmark_id') if isinstance(p, dict) else None for p in points] != list(range(33)):
            raise ValueError('Anatomical landmark IDs must occur exactly once in canonical order')
        for point in points:
            _keys(point, ('landmark_id', 'status', 'x_px', 'y_px'), 'Reference landmark')
            if (type(point['landmark_id']) is not int or not isinstance(point['status'], str)
                    or point['status'] not in STATES):
                raise ValueError('Unsupported anatomical landmark/status')
            x, y = point['x_px'], point['y_px']
            if point['status'] == 'visible':
                if not _finite(x) or not _finite(y) or not 0 <= x <= source['width'] or not 0 <= y <= source['height']:
                    raise ValueError('Visible reference needs finite image-edge pixel coordinates')
                visible += 1
            elif x is not None or y is not None:
                raise ValueError('Occluded/out-of-frame/unreviewed references must not invent coordinates')
    if require_scoring and not visible:
        raise ValueError('No visible manual reference point available; no accuracy result')
    return copy.deepcopy(reference)


def _stats(values):
    scale = max(values) if values else 0
    # Scaled aggregation avoids overflow even for a nearly degenerate manual torso.
    mean = math.fsum(v / scale for v in values) / len(values) * scale if scale else 0.0
    rmse = math.sqrt(math.fsum((v / scale) ** 2 for v in values) / len(values)) * scale if scale else 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2 else
              ordered[middle - 1] / 2 + ordered[middle] / 2) if ordered else None
    return {'count': len(values), 'mean': mean if values else None,
            'median': median,
            'rmse': rmse if values else None, 'max': max(values) if values else None}


def _reference_torso(points):
    if any(points[j]['status'] != 'visible' for j in (11, 12, 23, 24)):
        return None
    shoulder = tuple((points[11][a] + points[12][a]) / 2 for a in ('x_px', 'y_px'))
    hip = tuple((points[23][a] + points[24][a]) / 2 for a in ('x_px', 'y_px'))
    scale = math.dist(shoulder, hip)
    return scale if math.isfinite(scale) and scale > 0 else None


def evaluate_reference(reference, result, *, run_sha256, result_sha256, protocol=None):
    """Describe operator-reference agreement with coverage and every abstention.

    Reported frames are correlated samples from ONE source, not independent
    golfers. The software pilot can never pass product/coaching admission.
    """
    protocol = load_protocol() if protocol is None else protocol
    ref = validate_reference(reference, result, run_sha256=run_sha256, result_sha256=result_sha256,
                             protocol=protocol, require_scoring=True)
    ambiguous = result.get('identity_ambiguous', False) or any(
        f.get('status') == 'multiple_poses' or f.get('pose_count', 0) > 1 for f in result['frames'])
    states, abstentions = Counter(), Counter()
    errors, normalized, details = [], [], []
    per_joint = {j: {'visible': 0, 'errors': [], 'normalized': [], 'abstentions': Counter()} for j in range(33)}
    scale_unavailable = 0
    for frame in ref['frames']:
        observed = result['frames'][frame['frame_index']]
        scale = _reference_torso(frame['landmarks'])
        for point in frame['landmarks']:
            states[point['status']] += 1
            if point['status'] != 'visible':
                continue
            j = point['landmark_id']
            per_joint[j]['visible'] += 1
            reason = 'sequence_identity_ambiguous' if ambiguous else None
            if reason is None and observed['status'] != 'single_pose':
                reason = 'model_' + observed['status']
            row = observed['landmarks'][j]
            if reason is None and (not isinstance(row, (list, tuple)) or len(row) != 4
                                   or not all(_finite(v) for v in row) or not 0 <= row[3] <= 1):
                reason = 'model_point_missing_or_presence_filtered'
            if reason is None and row[3] < protocol['visibility_floor']:
                reason = 'model_visibility_below_registered_floor'
            if reason is None and not (0 <= row[0] <= 1 and 0 <= row[1] <= 1):
                reason = 'model_point_out_of_image'
            detail = {'frame_index': frame['frame_index'], 'pts_seconds': frame['pts_seconds'],
                      'landmark_id': j, 'error_px': None, 'error_reference_torso_lengths': None,
                      'abstention_reason': reason}
            if reason:
                abstentions[reason] += 1
                per_joint[j]['abstentions'][reason] += 1
            else:
                error = math.hypot(row[0] * frame['width'] - point['x_px'],
                                   row[1] * frame['height'] - point['y_px'])
                errors.append(error)
                per_joint[j]['errors'].append(error)
                detail['error_px'] = error
                norm = error / scale if scale is not None else None
                if norm is not None and math.isfinite(norm):
                    normalized.append(norm)
                    per_joint[j]['normalized'].append(norm)
                    detail['error_reference_torso_lengths'] = norm
                else:
                    scale_unavailable += 1
            details.append(detail)
    return {'schema_version': 1, 'study_id': protocol['study_id'],
            'scope': 'operator-reference software pilot agreement; not ground-truth accuracy or admission',
            'coordinate_system': COORDINATES,
            'reference_sha256': hashlib.sha256(json.dumps(ref, sort_keys=True, allow_nan=False,
                                                         separators=(',', ':')).encode()).hexdigest(),
            'video_sha256': ref['video_sha256'], 'run_sha256': run_sha256, 'result_sha256': result_sha256,
            'source_recording_id': protocol['artifact']['source_recording_id'], 'independent_sources': 1,
            'selected_frame_count': len(ref['frames']), 'reference_state_counts': dict(sorted(states.items())),
            'reference_slot_count': len(ref['frames']) * 33, 'eligible_visible_count': states['visible'],
            'scored_count': len(errors), 'model_abstention_count': sum(abstentions.values()),
            'abstentions': dict(sorted(abstentions.items())), 'pixel_error': _stats(errors),
            'reference_torso_normalized_error': _stats(normalized),
            'scored_points_without_reference_scale': scale_unavailable,
            'per_joint': [{'landmark_id': j, 'eligible_visible_count': v['visible'],
                           'scored_count': len(v['errors']), 'abstentions': dict(sorted(v['abstentions'].items())),
                           'pixel_error': _stats(v['errors']),
                           'reference_torso_normalized_error': _stats(v['normalized'])}
                          for j, v in per_joint.items()],
            'point_results': details,
            'validated_for_coaching': False, 'app_integration_permitted': False,
            'outcome_integration_permitted': False, 'advice': [],
            'limitations': list(protocol['limitations']) + [
                'Reference states, blinding and independence are operator declarations, not independently proven',
                'Unreviewed points and model abstentions remain explicit; error summaries are conditional on scored points',
                'No universal pass threshold, causal coaching implication or automatic gate promotion exists']}
