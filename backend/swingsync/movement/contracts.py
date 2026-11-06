"""Explicit 2D landmark/event contracts; no calibrated 3D or advice validation."""
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import numpy as np


class EventName(str, Enum):
    ADDRESS = 'Address'
    TOE_UP = 'Toe-up'
    MID_BACKSWING = 'Mid-backswing'
    TOP = 'Top'
    MID_DOWNSWING = 'Mid-downswing'
    IMPACT = 'Impact'
    MID_FOLLOW_THROUGH = 'Mid-follow-through'
    FINISH = 'Finish'


@dataclass(frozen=True)
class PoseEvent:
    name: str
    timestamp_seconds: float
    source: str = 'manual'
    confidence: float | None = None

    def __post_init__(self):
        try:
            value = EventName(self.name).value
        except (ValueError, TypeError) as exc:
            raise ValueError('Unsupported swing event name') from exc
        if (isinstance(self.timestamp_seconds, (bool, complex))
                or not isinstance(self.timestamp_seconds, (int, float, np.integer, np.floating))
                or not np.isfinite(self.timestamp_seconds)):
            raise ValueError('Event time must be finite real seconds')
        if self.source not in {'manual', 'predicted'}:
            raise ValueError('Event source must be manual or predicted')
        if self.confidence is not None and (
            isinstance(self.confidence, (bool, complex)) or not isinstance(self.confidence, (int, float))
            or not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError('Confidence must be within [0,1]; it is not calibrated reliability')
        object.__setattr__(self, 'name', value)
        object.__setattr__(self, 'timestamp_seconds', float(self.timestamp_seconds))


@dataclass(frozen=True)
class PoseSequence:
    """MediaPipe anatomical indices with normalized image x/y, unused z, visibility.

    Timestamps are actual decoded PTS in seconds. Coordinates describe the stored
    image. `mirrored` means x may be reflected into an unmirrored display; semantic
    left/right indices are already anatomical and must never be guessed/swapped.
    This does not assert that re-inference on a mirrored image preserves identity.
    Missing landmarks may be NaN, and visibility must be checked before use.
    """
    timestamps_seconds: np.ndarray
    landmarks: np.ndarray
    frame_width: int
    frame_height: int
    camera_view: str
    handedness: str
    mirrored: bool = False

    def __post_init__(self):
        for dimension in (self.frame_width, self.frame_height):
            if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 1:
                raise ValueError('Frame dimensions must be positive integers')
        if self.camera_view not in {'face-on', 'down-the-line', 'unknown'}:
            raise ValueError('Unsupported camera view')
        if self.handedness not in {'right', 'left'} or not isinstance(self.mirrored, bool):
            raise ValueError('Explicit handedness and mirrored flag required')
        times = np.asarray(self.timestamps_seconds)
        points = np.asarray(self.landmarks)
        if (times.ndim != 1 or len(times) < 1 or times.dtype.kind not in 'iuf'
                or not np.isfinite(times).all() or np.any(np.diff(times.astype(np.float64)) <= 0)):
            raise ValueError('Pose timestamps must be finite, real and strictly increasing')
        if points.shape != (len(times), 33, 4) or points.dtype.kind not in 'iuf':
            raise ValueError('Landmarks must be real numeric [T,33,4] x/y/z/visibility')
        if np.isinf(points).any():
            raise ValueError('Infinite landmark values are malformed; use NaN for missing observations')
        visibility = points[..., 3]
        if np.any(np.isfinite(visibility) & ((visibility < 0) | (visibility > 1))):
            raise ValueError('Finite landmark visibility must lie within [0,1]')
        # Own immutable copies: changing the caller's arrays cannot rewrite evidence.
        times = times.astype(np.float64, copy=True)
        points = points.astype(np.float64, copy=True)
        times.setflags(write=False)
        points.setflags(write=False)
        object.__setattr__(self, 'timestamps_seconds', times)
        object.__setattr__(self, 'landmarks', points)
