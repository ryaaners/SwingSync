"""Pure, descriptive measurements from supplied pose samples and event times.

No detector, event predictor, smoothing, interpolation or advice runs here. The
formulas describe image-plane geometry, not 3D motion or pressure. The visibility
and sample-gap defaults are engineering filters, not validated accuracy limits.
Synthetic tests check arithmetic and suppression only; every result remains
empirically unvalidated, even when its event annotations were supplied manually.

MediaPipe's anatomical index definitions and image-coordinate normalization:
https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmark
https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
These references define the input, not golf coaching standards or reliability.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .contracts import EventName, PoseEvent, PoseSequence


LANDMARK_REFERENCE = 'https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmark'
IMAGE_REFERENCE = 'https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker'
RELIABILITY = 'unvalidated: software arithmetic only; no empirical measurement or event-error calibration'
ADVICE_SUPPRESSION = 'No validated measurement/event reliability or coaching evidence; descriptive results only.'


@dataclass(frozen=True)
class Measurement:
    name: str
    status: Literal['measured', 'unavailable']
    value: float | None
    unit: str
    event_names: tuple[str, ...]
    interval_seconds: tuple[float, float] | None
    events: tuple[PoseEvent, ...]
    frame_indices: tuple[int, ...]
    frame_timestamps_seconds: tuple[float, ...]
    camera_view: str
    method: str
    reference: str
    reliability: str = RELIABILITY
    reason: str | None = None


@dataclass(frozen=True)
class MeasurementReport:
    measurements: tuple[Measurement, ...]
    camera_view: str
    handedness: str
    mirrored: bool
    visibility_threshold: float
    max_sample_gap_seconds: float
    advice: tuple[str, ...] = ()
    advice_suppression_reason: str = ADVICE_SUPPRESSION


def extract_measurements(
    sequence: PoseSequence,
    events: Sequence[PoseEvent],
    *,
    visibility_threshold: float = 0.5,
    max_sample_gap_seconds: float = 0.25,
) -> MeasurementReport:
    """Recompute six descriptive measurements; return explicit unavailable reasons.

    Durations use the supplied event times directly, including between-frame
    manual annotations. They require covered, strictly ordered event times, but
    do not require visible landmarks. Pose-dependent results require the relevant
    event time to coincide with an observed PTS (absolute tolerance 1e-9 seconds),
    visible in-image landmarks and, for a trajectory, no gap over the gap budget.
    No frame-rate conversion or nearest-frame snapping is permitted.

    Head displacement is the largest sampled 2D distance of the mean of nose and
    both ears from its Address position over Address..Impact, divided by Address
    shoulder-midpoint to hip-midpoint distance. Its view is recorded explicitly;
    views are not interchangeable. It measures camera-relative projected motion,
    so camera motion and perspective remain confounds. All three head landmarks
    must remain available: switching head points would create artificial motion.

    Elbow values are projected shoulder-elbow-wrist angles at Top. The lead arm
    is anatomical left for a right-handed golfer and right for a left-handed
    golfer. Mirroring reflects only x; it never swaps anatomical indices. Neither
    angle is a calibrated anatomical joint angle or a recommendation.
    """
    for value, label in ((visibility_threshold, 'visibility_threshold'),
                         (max_sample_gap_seconds, 'max_sample_gap_seconds')):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f'{label} must be finite real')
    if not 0 <= visibility_threshold <= 1:
        raise ValueError('visibility_threshold must lie within [0,1]')
    if max_sample_gap_seconds <= 0:
        raise ValueError('max_sample_gap_seconds must be positive')

    events = tuple(events)
    if any(not isinstance(event, PoseEvent) for event in events):
        raise TypeError('events must contain PoseEvent objects')
    event_map = {event.name: event for event in events}
    ordered = [event_map[name.value] for name in EventName if name.value in event_map]
    global_reason = None
    if len(event_map) != len(events):
        global_reason = 'duplicate_event_names'
    elif any(b.timestamp_seconds <= a.timestamp_seconds for a, b in zip(ordered, ordered[1:])):
        global_reason = 'events_not_strictly_ordered'

    times = sequence.timestamps_seconds
    xy = sequence.landmarks[..., :2].copy()
    if sequence.mirrored:
        xy[..., 0] = 1 - xy[..., 0]
    # Normalize each axis using its own dimension before Euclidean geometry.
    pixels = xy * np.array([sequence.frame_width, sequence.frame_height])
    visibility = sequence.landmarks[..., 3]
    rows: list[Measurement] = []

    def matching_frame(timestamp: float) -> int | None:
        matches = np.flatnonzero(np.isclose(times, timestamp, rtol=0, atol=1e-9))
        return int(matches[0]) if len(matches) == 1 else None

    def supporting_frames(selected: tuple[PoseEvent, ...]) -> tuple[int, ...]:
        indices: set[int] = set()
        for event in selected:
            exact = matching_frame(event.timestamp_seconds)
            if exact is not None:
                indices.add(exact)
            elif times[0] <= event.timestamp_seconds <= times[-1]:
                upper = int(np.searchsorted(times, event.timestamp_seconds))
                indices.update((upper - 1, upper))
        return tuple(sorted(indices))

    def event_reason(names: tuple[str, ...]) -> str | None:
        if global_reason:
            return global_reason
        if any(name not in event_map for name in names):
            return 'missing_required_event'
        if any(not times[0] <= event_map[name].timestamp_seconds <= times[-1] for name in names):
            return 'event_outside_sample_coverage'
        return None

    def add(name, value, unit, names, method, reference, reason=None, frames=None):
        if reason is None and (value is None or not np.isfinite(value)):
            reason = 'nonfinite_measurement'
        selected = tuple(event_map[event_name] for event_name in names if event_name in event_map)
        if frames is None:
            frames = supporting_frames(selected)
        interval = None
        if len(selected) == len(names):
            interval = (selected[0].timestamp_seconds, selected[-1].timestamp_seconds)
        rows.append(Measurement(
            name=name, status='measured' if reason is None else 'unavailable',
            value=float(value) if reason is None else None, unit=unit, event_names=names,
            interval_seconds=interval, events=selected, frame_indices=tuple(frames),
            frame_timestamps_seconds=tuple(float(times[i]) for i in frames),
            camera_view=sequence.camera_view, method=method, reference=reference, reason=reason,
        ))

    for name, names in (
        ('backswing_duration_seconds', ('Address', 'Top')),
        ('downswing_duration_seconds', ('Top', 'Impact')),
        ('backswing_downswing_ratio', ('Address', 'Top', 'Impact')),
    ):
        reason = event_reason(names)
        value = None
        if reason is None:
            t = [event_map[n].timestamp_seconds for n in names]
            value = t[1] - t[0] if len(t) == 2 else (t[1] - t[0]) / (t[2] - t[1])
        method = ('Supplied end event timestamp minus supplied start event timestamp; seconds, no FPS.'
                  if len(names) == 2 else '(Top-Address)/(Impact-Top) from supplied timestamps; no ideal ratio.')
        add(name, value, 'seconds' if len(names) == 2 else 'ratio', names, method,
            'Defined timestamp arithmetic; event annotations are not validated by this extractor.', reason)

    def visible(frame_indices, landmark_indices) -> bool:
        coords = xy[np.ix_(frame_indices, landmark_indices)]
        scores = visibility[np.ix_(frame_indices, landmark_indices)]
        return bool(np.isfinite(coords).all() and np.all((coords >= 0) & (coords <= 1))
                    and np.isfinite(scores).all() and np.all(scores >= visibility_threshold))

    names = ('Address', 'Impact')
    reason = event_reason(names)
    frames = supporting_frames(tuple(event_map[n] for n in names if n in event_map))
    value = None
    if reason is None and sequence.camera_view == 'unknown':
        reason = 'camera_view_required'
    if reason is None:
        start, end = (matching_frame(event_map[n].timestamp_seconds) for n in names)
        if start is None or end is None:
            reason = 'event_not_sampled'
        else:
            frames = tuple(range(start, end + 1))
            if np.any(np.diff(times[list(frames)]) > max_sample_gap_seconds):
                reason = 'sample_gap_exceeds_budget'
            elif not visible(frames, (0, 7, 8)) or not visible((start,), (11, 12, 23, 24)):
                reason = 'head_or_baseline_landmarks_unavailable'
            else:
                torso = (pixels[start, 11] + pixels[start, 12]) / 2 - (
                    pixels[start, 23] + pixels[start, 24]) / 2
                scale = float(np.linalg.norm(torso))
                if scale <= 1e-9:
                    reason = 'degenerate_address_body_scale'
                else:
                    head = pixels[list(frames)][:, (0, 7, 8)].mean(axis=1)
                    value = float(np.linalg.norm(head - head[0], axis=1).max() / scale)
    add('normalized_head_max_displacement', value, 'address_torso_lengths', names,
        'Max sampled norm(head_centroid(t)-head_centroid(Address))/Address torso length; '
        'head=mean(nose,left_ear,right_ear); torso=shoulder midpoint to hip midpoint; '
        'aspect-correct 2D pixels; camera-relative; no z or cross-view equivalence.',
        IMAGE_REFERENCE, reason, frames)

    lead = (11, 13, 15) if sequence.handedness == 'right' else (12, 14, 16)
    trail = (12, 14, 16) if sequence.handedness == 'right' else (11, 13, 15)
    for arm, joints in (('lead', lead), ('trail', trail)):
        names = ('Top',)
        reason = event_reason(names)
        value = None
        frames = supporting_frames(tuple(event_map[n] for n in names if n in event_map))
        if reason is None and sequence.camera_view == 'unknown':
            reason = 'camera_view_required'
        if reason is None:
            frame = matching_frame(event_map['Top'].timestamp_seconds)
            if frame is None:
                reason = 'event_not_sampled'
            elif not visible((frame,), joints):
                reason = 'elbow_landmarks_unavailable'
            else:
                shoulder, elbow, wrist = pixels[frame, list(joints)]
                upper, lower = shoulder - elbow, wrist - elbow
                lengths = np.linalg.norm(upper), np.linalg.norm(lower)
                if min(lengths) <= 1e-9:
                    reason = 'degenerate_projected_arm'
                else:
                    cosine = np.dot(upper / lengths[0], lower / lengths[1])
                    value = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        add(f'{arm}_elbow_angle_at_top', value, 'degrees', names,
            'acos(dot(unit(shoulder-elbow),unit(wrist-elbow))) in aspect-correct 2D pixels; '
            f'anatomical landmark indices {joints}; lead arm follows handedness; '
            'mirror reflects x only; projected angle, no z or ideal threshold.',
            LANDMARK_REFERENCE, reason, frames)

    return MeasurementReport(tuple(rows), sequence.camera_view, sequence.handedness, sequence.mirrored,
                             float(visibility_threshold), float(max_sample_gap_seconds))
