"""Synthetic videos validate index/PTS mechanics, never GolfDB image alignment."""
from fractions import Fraction

import av
import numpy as np
import pytest

from backend.swingsync.research.golfdb import GolfDBRecord
from backend.swingsync.research.timeline import inspect_clip_timeline


def record():
    return GolfDBRecord(1, 'fixture', 'fixture', 'unknown', 'iron', 'face-on', 0,
                        (3, 3, 4, 5, 6, 7, 8, 9, 10, 11), (0., 0., 1., 1.), 1)


def write_video(path, pts):
    with av.open(str(path), 'w') as c:
        s = c.add_stream('mpeg4', rate=25)
        s.width = s.height = 32
        s.pix_fmt = 'yuv420p'
        s.codec_context.time_base = Fraction(1, 1000)
        for i, t in enumerate(pts):
            f = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i, np.uint8), format='rgb24')
            f.pts, f.time_base = t, Fraction(1, 1000)
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_source_inclusive_one_based_trim_and_variable_pts(tmp_path):
    p = tmp_path / 'source.mp4'
    pts = [1000 + i * 40 + i * i * 2 for i in range(14)]
    write_video(p, pts)
    timeline = inspect_clip_timeline(p, record(), origin='source')
    assert timeline.decoded_indices == tuple(range(2, 11))
    assert timeline.historical_source_indices == tuple(range(2, 11))
    assert timeline.pts_seconds == pytest.approx([t / 1000 for t in pts[2:11]])
    assert timeline.labels == (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert timeline.event_timestamps() == pytest.approx([t / 1000 for t in pts[2:10]])
    assert timeline.timestamp_provenance == 'original-source-pts'


def test_preprocessed_pts_never_claim_original_source_time(tmp_path):
    p = tmp_path / 'clip.mp4'
    write_video(p, [i * 40 for i in range(9)])
    timeline = inspect_clip_timeline(p, record(), origin='preprocessed')
    assert timeline.decoded_indices == tuple(range(9))
    assert timeline.historical_source_indices == tuple(range(2, 11))
    assert timeline.pts_seconds[0] == 0
    assert timeline.timestamp_provenance == 'reencoded-clip-pts'


@pytest.mark.parametrize('count', [8, 10])
def test_preprocessed_exact_frame_count_required(tmp_path, count):
    p = tmp_path / 'clip.mp4'
    write_video(p, [i * 40 for i in range(count)])
    with pytest.raises(ValueError, match='frame count|shorter'):
        inspect_clip_timeline(p, record(), origin='preprocessed')


def test_decode_budget_and_short_source_rejected(tmp_path):
    p = tmp_path / 'source.mp4'
    write_video(p, [i * 40 for i in range(9)])
    with pytest.raises(ValueError, match='budget'):
        inspect_clip_timeline(p, record(), origin='source', max_decode_frames=5)
    with pytest.raises(ValueError, match='shorter'):
        inspect_clip_timeline(p, record(), origin='source')
