"""GolfDB's full-clip benchmark, separately from any order-constrained decoder.

Protocol reference (not vendored code): McNally et al., GolfDB (2019),
https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/util.py
https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/eval.py

The reference uses NumPy's *default* argsort and takes its last index for each
class. It does not specify stable sorting. We deliberately call the same sort,
record NumPy's version, and make no promise of tie parity across NumPy builds.
These functions evaluate predictions; they do not run or validate a model.
"""
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

EVENT_NAMES = (
    "Address", "Toe-up", "Mid-backswing", "Top", "Mid-downswing", "Impact",
    "Mid-follow-through", "Finish",
)
PROTOCOL = "golfdb-independent-events-v1"
TIE_POLICY = "numpy-default-argsort-last-index"


def _valid_length(length: int, valid_mask) -> int:
    if valid_mask is None:
        return length
    mask = np.asarray(valid_mask)
    if mask.shape != (length,) or mask.dtype != np.bool_:
        raise ValueError("valid_mask must be a boolean vector matching the frame count")
    count = int(mask.sum())
    if not np.array_equal(mask, np.arange(length) < count):
        raise ValueError("Benchmark masks must be a valid full-clip prefix followed by padding")
    return count


def events_from_labels(labels, *, valid_mask=None) -> tuple[int, ...]:
    """Recover class-specific indices, rejecting missing/duplicate classes.

    Valid labels are integer class IDs 0..8. Padded labels are ignored. A mask
    cannot legitimize a sampled subsequence: callers must supply the full clip.
    """
    array = np.asarray(labels)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer vector")
    valid = array[:_valid_length(len(array), valid_mask)]
    if np.any((valid < 0) | (valid > 8)):
        raise ValueError("Valid labels must be class IDs 0..8")
    indices = np.flatnonzero(valid < 8)
    if len(indices) != 8 or not np.array_equal(np.sort(valid[indices]), np.arange(8)):
        raise ValueError("Full clip must contain each event class 0..7 exactly once")
    return tuple(int(np.flatnonzero(valid == event)[0]) for event in range(8))


@dataclass(frozen=True)
class ClipEvaluation:
    annotation_event_frames: tuple[int, ...]
    annotation_order_anomaly: bool
    event_frames: tuple[int, ...]
    predicted_frames: tuple[int, ...]
    signed_error_frames: tuple[int, ...]
    absolute_error_frames: tuple[int, ...]
    tolerance_frames: int
    correct: tuple[bool, ...]
    signed_error_ms: tuple[float, ...] | None
    absolute_error_ms: tuple[float, ...] | None
    valid_frame_count: int
    numpy_version: str
    protocol: str = PROTOCOL
    tie_policy: str = TIE_POLICY

    def to_dict(self) -> dict:
        result = asdict(self)
        result["pce"] = float(np.mean(self.correct))
        result["pce_percent"] = result["pce"] * 100
        return result


def evaluate_clip(probabilities, event_frames, *, valid_mask=None,
                  timestamps_seconds=None) -> ClipEvaluation:
    """Evaluate one full clip, optionally with a padded suffix.

    probabilities are post-softmax [T,9], not logits. Predicted indices and
    tolerance are in full-clip frame coordinates. Input event_frames are class-
    specific annotation indices. To match the official reference, benchmark
    targets are sorted chronologically, exactly as np.where(labels < 8)[0].
    Nonchronological annotations are preserved and flagged, never silently
    corrected: official per-event slots then differ from semantic event labels.
    Milliseconds are computed
    only from supplied presentation timestamps; never inferred from FPS.
    Timestamp provenance (source versus re-encoded media) belongs in the run
    manifest. Scores/timestamps in padding are ignored, including sentinels.
    """
    scores = np.asarray(probabilities)
    if scores.ndim != 2 or scores.shape[1] != 9 or not np.issubdtype(scores.dtype, np.number):
        raise ValueError("probabilities must be a numeric [T,9] array")
    count = _valid_length(len(scores), valid_mask)
    if count < 8:
        raise ValueError("A full clip needs at least eight valid frames")
    scores = scores[:count]
    if (not np.isrealobj(scores) or not np.all(np.isfinite(scores))
            or np.any((scores < 0) | (scores > 1))
            or not np.allclose(scores.sum(axis=1), 1, rtol=1e-5, atol=1e-6)):
        raise ValueError("Valid probabilities must be finite, real, in [0,1] and sum to one")
    events = np.asarray(event_frames)
    if events.shape != (8,) or not np.issubdtype(events.dtype, np.integer):
        raise ValueError("event_frames must contain eight integer frame indices")
    if np.any(events < 0) or np.any(events >= count) or len(np.unique(events)) != 8:
        raise ValueError("Event indices must be distinct and within the valid clip")
    annotation_events = tuple(int(value) for value in events)
    events = np.sort(events)
    order_anomaly = annotation_events != tuple(int(value) for value in events)
    # Float conversion before arithmetic avoids overflow on narrow integer dtypes.
    tolerance = int(max(np.round((int(events[5]) - int(events[0])) / 30), 1))
    predictions = np.array([np.argsort(scores[:, i])[-1] for i in range(8)], dtype=np.int64)
    signed = predictions - events.astype(np.int64)
    absolute = np.abs(signed)
    signed_ms = absolute_ms = None
    if timestamps_seconds is not None:
        times = np.asarray(timestamps_seconds)
        if (times.shape != (len(probabilities),) or not np.issubdtype(times.dtype, np.number)
                or not np.isrealobj(times)):
            raise ValueError("timestamps_seconds must be real numeric values matching the padded frame count")
        times = times[:count].astype(np.float64)
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("Valid presentation timestamps must be finite and strictly increasing")
        signed_ms_array = (times[predictions] - times[events]) * 1000
        signed_ms = tuple(float(value) for value in signed_ms_array)
        absolute_ms = tuple(float(value) for value in np.abs(signed_ms_array))
    return ClipEvaluation(
        annotation_event_frames=annotation_events, annotation_order_anomaly=order_anomaly,
        event_frames=tuple(int(value) for value in events),
        predicted_frames=tuple(int(value) for value in predictions),
        signed_error_frames=tuple(int(value) for value in signed),
        absolute_error_frames=tuple(int(value) for value in absolute),
        tolerance_frames=tolerance,
        correct=tuple(bool(value) for value in absolute <= tolerance),
        signed_error_ms=signed_ms, absolute_error_ms=absolute_ms,
        valid_frame_count=count, numpy_version=np.__version__,
    )


def summarize_evaluations(results: Iterable[ClipEvaluation]) -> dict:
    """Equal-weight clip/event PCE and timing errors; no partial timestamp pool."""
    clips = tuple(results)
    if not clips:
        raise ValueError("Cannot summarize an empty evaluation")
    if any((r.protocol, r.tie_policy, r.numpy_version) !=
           (PROTOCOL, TIE_POLICY, clips[0].numpy_version) for r in clips):
        raise ValueError("Cannot mix evaluation protocols, tie policies or NumPy runtimes")
    timed = [r.absolute_error_ms is not None for r in clips]
    if any(timed) and not all(timed):
        raise ValueError("Do not silently summarize partial timestamp coverage")
    correct = np.asarray([r.correct for r in clips])
    frame_errors = np.asarray([r.absolute_error_frames for r in clips])
    ms_errors = np.asarray([r.absolute_error_ms for r in clips]) if all(timed) else None
    per_event = {}
    for i, name in enumerate(EVENT_NAMES):
        per_event[name] = {
            "pce": float(correct[:, i].mean()),
            "pce_percent": float(correct[:, i].mean() * 100),
            "mae_frames": float(frame_errors[:, i].mean()),
            "mae_ms": float(ms_errors[:, i].mean()) if ms_errors is not None else None,
        }
    return {
        "protocol": PROTOCOL, "tie_policy": TIE_POLICY,
        "numpy_version": clips[0].numpy_version, "clip_count": len(clips),
        "pce": float(correct.mean()), "pce_percent": float(correct.mean() * 100),
        "mae_frames": float(frame_errors.mean()),
        "mae_ms": float(ms_errors.mean()) if ms_errors is not None else None,
        "per_event": per_event, "order_constrained_decoding": False,
        "annotation_order_anomaly_count": sum(r.annotation_order_anomaly for r in clips),
        "target_mapping": "chronological occupied frames, matching reference np.where(labels < 8)",
    }
