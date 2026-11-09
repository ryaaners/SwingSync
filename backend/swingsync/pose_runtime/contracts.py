"""Pure timestamp, input and raw-observation contracts (no native imports)."""
from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real


@dataclass(frozen=True)
class FrameTime:
    frame_index: int
    pts: int
    time_base_numerator: int
    time_base_denominator: int
    pts_seconds: float
    api_timestamp_ms: int


def video_clock(ticks_and_bases):
    """Exact relative half-up milliseconds; never replace the source PTS coordinate."""
    rows, previous, previous_float, origin, previous_ms = [], None, None, None, None
    for index, (ticks, numerator, denominator) in enumerate(ticks_and_bases):
        if any(type(v) is not int for v in (ticks, numerator, denominator)):
            raise ValueError('Integer PTS and rational time base required')
        if not -(2**63) <= ticks < 2**63 or not 0 < numerator < 2**63 or not 0 < denominator < 2**63:
            raise ValueError('PTS or time base outside signed-int64 contract')
        exact = ticks * Fraction(numerator, denominator)
        seconds = float(exact)
        if not math.isfinite(seconds) or (previous is not None and exact <= previous):
            raise ValueError('Actual PTS must increase strictly')
        if previous_float is not None and seconds <= previous_float:
            raise ValueError('Float64 actual PTS cannot distinguish these frames')
        if origin is None:
            origin = exact
        milliseconds = (exact - origin) * 1000
        rounded = (2 * milliseconds.numerator + milliseconds.denominator) // (2 * milliseconds.denominator)
        if rounded >= 2**63 or (previous_ms is not None and rounded <= previous_ms):
            raise ValueError('VIDEO integer-ms collision/overflow; no jitter, drop or fallback permitted')
        rows.append(FrameTime(index, ticks, numerator, denominator, seconds, rounded))
        previous, previous_float, previous_ms = exact, seconds, rounded
    if not rows:
        raise ValueError('No decoded frames')
    return tuple(rows)


def validate_context(camera_view, handedness, unmirrored, single_golfer_confirmed, rights, timestamp_basis):
    if camera_view not in {'face-on', 'down-the-line'} or handedness not in {'left', 'right'}:
        raise ValueError('Explicit supported camera view and handedness required')
    if unmirrored is not True or single_golfer_confirmed is not True:
        raise ValueError('Reviewed unmirrored, single-golfer input required; no identity inference')
    if not isinstance(rights, str) or not rights.strip() or len(rights) > 2000:
        raise ValueError('Record explicit local-use media rights/provenance')
    if timestamp_basis not in {'original', 'reencoded'}:
        raise ValueError('Timestamp basis must be original or reencoded')
    return {'camera_view': camera_view, 'handedness': handedness, 'mirrored': False,
            'single_golfer_confirmed_by_operator': True, 'rights_provenance': rights.strip(),
            'timestamp_basis': timestamp_basis, 'anatomical_identity_empirically_validated': False}


def _number(value):
    return (float(value) if not isinstance(value, bool) and isinstance(value, Real)
            and math.isfinite(value) else None)


def pose_observation(poses, *, presence_floor=0.5):
    """Retain raw scores. Missing/multiple/malformed observations never yield points.

    The presence floor is an engineering filter, not measured golf reliability.
    x/y outside the image are retained, never clamped. Anatomical IDs are unchanged.
    """
    if (_number(presence_floor) is None or not 0 <= presence_floor <= 1):
        raise ValueError('Presence floor must be finite within [0,1]')
    if not isinstance(poses, (list, tuple)):
        raise ValueError('Pose result must be a sequence')
    blank = [[None] * 4 for _ in range(33)]
    raw = []
    malformed = False
    for pose in poses:
        if not isinstance(pose, (list, tuple)):
            raise ValueError('Pose landmarks must be a sequence')
        points = []
        if len(pose) != 33:
            malformed = True
        for point in pose:
            row = {field: _number(getattr(point, field, None)) for field in ('x', 'y', 'z', 'visibility', 'presence')}
            if any(v is None for v in row.values()) or any(
                row[f] is not None and not 0 <= row[f] <= 1 for f in ('visibility', 'presence')
            ):
                malformed = True
            points.append(row)
        raw.append(points)
    status = ('no_pose' if len(poses) == 0 else 'multiple_poses' if len(poses) > 1
              else 'malformed_pose' if malformed else 'single_pose')
    result = {'status': status, 'pose_count': len(poses), 'raw_poses': raw,
              'landmarks': blank, 'presence': [None] * 33, 'presence_filtered_indices': []}
    if status == 'single_pose':
        result['presence'] = [p['presence'] for p in raw[0]]
        for index, point in enumerate(raw[0]):
            if point['presence'] < presence_floor:
                result['presence_filtered_indices'].append(index)
            else:
                result['landmarks'][index] = [point[f] for f in ('x', 'y', 'z', 'visibility')]
    return result


def as_pose_sequence(result):
    """Construct the immutable measurement contract from completed observations.

    This is descriptive input, with no validated identity/landmark accuracy claim.
    """
    import numpy as np
    from backend.swingsync.movement.contracts import PoseSequence
    if result.get('schema_version') != 1 or result.get('state') != 'completed':
        raise ValueError('Completed pose artifact required')
    if result.get('identity_ambiguous', False) or any(
        f.get('status') == 'multiple_poses' or f.get('pose_count', 0) > 1 for f in result['frames']
    ):
        raise ValueError('Multiple-pose observation makes sequence identity ambiguous; spatial sequence suppressed')
    context = result['context']
    validate_context(context['camera_view'], context['handedness'], context['mirrored'] is False,
                     context['single_golfer_confirmed_by_operator'], context['rights_provenance'],
                     context['timestamp_basis'])
    return PoseSequence(np.array([f['pts_seconds'] for f in result['frames']], dtype=np.float64),
                        np.array([f['landmarks'] for f in result['frames']], dtype=np.float64),
                        result['video']['width'], result['video']['height'],
                        context['camera_view'], context['handedness'], False)
