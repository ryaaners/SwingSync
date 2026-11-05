"""Offline research mapping, not the web upload decoder or verified browser time.

Source indexing follows GolfDB's historical inclusive, one-based trim counter.
Preprocessed PTS belong to the re-encoded clip, never the unavailable original.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import numpy as np

from .golfdb import GolfDBRecord


@dataclass(frozen=True)
class ClipTimeline:
    clip_id: int
    origin: str
    timestamp_provenance: str
    decoded_indices: tuple[int, ...]
    historical_source_indices: tuple[int, ...]
    pts_seconds: tuple[float, ...]
    labels: tuple[int, ...]
    container_start_seconds: float | None

    def event_timestamps(self) -> tuple[float, ...]:
        return tuple(self.pts_seconds[self.labels.index(event)] for event in range(8))


def inspect_clip_timeline(
    path: str | Path,
    record: GolfDBRecord,
    *,
    origin: Literal['source', 'preprocessed'],
    max_decode_frames: int = 100_000,
) -> ClipTimeline:
    """Decode sequentially without guessed FPS, seeking, wrapping or dropped indices.

    Exhausts a preprocessed clip to check exact inclusive-trim frame count. For
    source media, reads through trim_end only. This is for trusted local research
    artifacts, not arbitrary uploads; web decoding has separate process limits.
    No images, crop equivalence, browser alignment or training result is certified.
    """
    if origin not in {'source', 'preprocessed'}:
        raise ValueError('origin must be source or preprocessed')
    if not isinstance(max_decode_frames, int) or max_decode_frames < 1:
        raise ValueError('max_decode_frames must be positive')
    start = record.source_frame_index(0) if origin == 'source' else 0
    count = record.expected_frame_count
    stop = start + count
    indices, times = [], []
    path = Path(path)
    if path.suffix.lower() not in {'.mp4', '.mov', '.webm', '.avi'}:
        raise ValueError('Unsupported local video extension')
    formats = {'.mp4': 'mov', '.mov': 'mov', '.webm': 'matroska', '.avi': 'avi'}
    with av.open(str(path), format=formats[path.suffix.lower()], options={'protocol_whitelist': 'file'}) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Exactly one video stream required')
        container_start = None if container.start_time is None else container.start_time / av.time_base
        previous_pts = None
        for index, frame in enumerate(container.decode(video=0)):
            if index >= max_decode_frames:
                raise ValueError('Decode frame budget exceeded')
            if index >= stop:
                if origin == 'preprocessed':
                    raise ValueError('Preprocessed frame count does not match inclusive annotation trim')
                break
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Missing decoded presentation timestamp')
            pts = float(frame.pts * frame.time_base)
            if not np.isfinite(pts) or (previous_pts is not None and pts <= previous_pts):
                raise ValueError('Non-increasing or invalid presentation timestamps')
            previous_pts = pts
            if index >= start:
                indices.append(index)
                times.append(pts)
            if origin == 'source' and index + 1 == stop:
                break
    if len(times) != count:
        raise ValueError('Decoded clip is shorter than inclusive annotation trim')
    labels = [8] * count
    for event, local_index in enumerate(record.event_indices):
        labels[local_index] = event
    return ClipTimeline(
        clip_id=record.clip_id, origin=origin,
        timestamp_provenance='original-source-pts' if origin == 'source' else 'reencoded-clip-pts',
        decoded_indices=tuple(indices),
        historical_source_indices=tuple(record.source_frame_index(i) for i in range(count)),
        pts_seconds=tuple(times), labels=tuple(labels), container_start_seconds=container_start,
    )
