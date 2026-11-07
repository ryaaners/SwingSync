"""Manual frame annotations and descriptive recording-time arithmetic only.

Annotations never imply event detection, landmark inference or validated coaching.
All times are looked up from inspected actual PTS; callers cannot supply a time.
"""
from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .movement.contracts import EventName

EVENT_NAMES = tuple(event.value for event in EventName)
RELIABILITY = 'Unvalidated manual annotation; no event-accuracy or coaching calibration.'
LIMITATION = ('Recording-time measurements depend on your manual frame choices. Slow motion or speed ramps '
              'may change physical timing. No ideal tempo or practice advice is inferred.')


class EventInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: EventName
    frame_index: Annotated[StrictInt, Field(ge=0)]


class AnnotationUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_revision: Annotated[StrictInt, Field(ge=0)]
    events: Annotated[list[EventInput], Field(max_length=8)]


class AnnotationConflict(ValueError):
    """Current state cannot be safely annotated; return an explicit conflict."""


def validated_video_info(info):
    """Validate saved inspection evidence before using its frame/time mapping."""
    if not isinstance(info, dict):
        raise AnnotationConflict('Video inspection is unavailable. Upload the recording again.')
    relative = info.get('timestamps_seconds')
    native = info.get('source_timestamps_seconds')
    count = info.get('frame_count')
    start = info.get('source_start_seconds')
    sha = info.get('sha256')
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise AnnotationConflict('This older recording has no media fingerprint. Upload it again to mark verified frames.')
    if (type(count) is not int or not 2 <= count <= 3600 or not isinstance(relative, list)
            or not isinstance(native, list) or len(relative) != count or len(native) != count
            or type(start) not in (int, float) or not math.isfinite(start)):
        raise AnnotationConflict('Saved video timestamps are unavailable. Upload the recording again.')
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in relative + native):
        raise AnnotationConflict('Saved video timestamps are invalid. Upload the recording again.')
    if (any(b <= a for a, b in zip(native, native[1:]))
            or any(b <= a for a, b in zip(relative, relative[1:]))
            or relative[0] != 0 or native[0] != start
            or any(abs((t - start) - r) > 1e-9 for t, r in zip(native, relative))):
        raise AnnotationConflict('Saved video timestamp correspondence is invalid. Upload the recording again.')
    return info


def apply_annotations(item, update: AnnotationUpdate):
    """Build a new revision; transaction and compare-and-swap belong to Store."""
    if item.get('status') != 'completed':
        raise AnnotationConflict('Complete video inspection before marking events.')
    revision = item.get('annotation_revision', 0)
    if update.expected_revision != revision:
        raise AnnotationConflict('Annotations changed in another tab. Reload the recording before saving again.')
    info = validated_video_info(item.get('video'))
    supplied = {event.name.value: event.frame_index for event in update.events}
    if len(supplied) != len(update.events):
        raise ValueError('Each event name may be marked only once.')
    if any(index >= info['frame_count'] for index in supplied.values()):
        raise ValueError('Event frame index lies outside this recording.')
    ordered = [(name, supplied[name]) for name in EVENT_NAMES if name in supplied]
    if any(b[1] <= a[1] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Marked events must occupy distinct frames in swing-event order.')
    now = datetime.now(timezone.utc).isoformat()
    events = [{'name': name, 'frame_index': index, 'timestamp_seconds': info['timestamps_seconds'][index],
               'source_timestamp_seconds': info['source_timestamps_seconds'][index],
               'source': 'manual', 'updated_at': now} for name, index in ordered]
    by_name = {event['name']: event for event in events}
    measurements = []
    if all(name in by_name for name in ('Address', 'Top', 'Impact')):
        address, top, impact = (by_name[name] for name in ('Address', 'Top', 'Impact'))
        backswing = top['timestamp_seconds'] - address['timestamp_seconds']
        downswing = impact['timestamp_seconds'] - top['timestamp_seconds']
        for name, value, unit, selected, method in (
            ('backswing_duration_seconds', backswing, 'seconds', (address, top),
             'Top timestamp minus Address timestamp; actual recorded PTS.'),
            ('downswing_duration_seconds', downswing, 'seconds', (top, impact),
             'Impact timestamp minus Top timestamp; actual recorded PTS.'),
            ('backswing_downswing_ratio', backswing / downswing, 'ratio', (address, top, impact),
             '(Top-Address)/(Impact-Top) from supplied manual frames; no ideal ratio.'),
        ):
            if not math.isfinite(value):
                raise ValueError('Manual timing is not finite; verify the recording timestamps.')
            measurements.append({'name': name, 'value': value, 'unit': unit,
                                 'event_names': [event['name'] for event in selected],
                                 'frame_indices': [event['frame_index'] for event in selected],
                                 'interval_seconds': [selected[0]['timestamp_seconds'], selected[-1]['timestamp_seconds']],
                                 'source': 'manual', 'reliability': RELIABILITY, 'method': method,
                                 'reference': 'Defined timestamp arithmetic on user-selected decoded frames.',
                                 'limitation': LIMITATION})
    result = deepcopy(item)
    result.update(annotation_revision=revision + 1, events=events, measurements=measurements,
                  observations=[], advice=[], advice_suppression_reason='No validated coaching evidence; manual descriptive timing only.')
    result.setdefault('annotation_history', []).append({'revision': revision + 1, 'updated_at': now,
                                                       'events': deepcopy(events)})
    return result
