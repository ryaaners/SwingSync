"""Landmark contract checks are software evidence, not pose/model validation."""
import numpy as np
import pytest

from backend.swingsync.movement.contracts import EventName, PoseEvent, PoseSequence


def sequence(**kwargs):
    values = dict(timestamps_seconds=np.array([1., 1.04, 1.17]),
                  landmarks=np.zeros((3, 33, 4)), frame_width=1920, frame_height=1080,
                  camera_view='face-on', handedness='right', mirrored=False)
    values.update(kwargs)
    return PoseSequence(**values)


def test_owned_arrays_preserve_evidence_against_caller_mutation():
    times = np.array([1., 1.04, 1.17])
    points = np.zeros((3, 33, 4))
    result = sequence(timestamps_seconds=times, landmarks=points)
    times[:] = 99
    points[:] = 99
    assert result.timestamps_seconds.tolist() == [1., 1.04, 1.17]
    assert np.all(result.landmarks == 0)
    with pytest.raises(ValueError):
        result.landmarks[0, 0, 0] = 1
    with pytest.raises(ValueError):
        result.timestamps_seconds[0] = 1


@pytest.mark.parametrize('times', [[1, 1, 2], [2, 1, 0], [1, np.nan, 3], [], np.array([1j, 2j])])
def test_invalid_pts_rejected(times):
    with pytest.raises(ValueError, match='timestamps'):
        sequence(timestamps_seconds=np.asarray(times))


@pytest.mark.parametrize('change', [dict(frame_width=True), dict(frame_height=0), dict(handedness='unknown'),
                                    dict(camera_view='side'), dict(mirrored='false'),
                                    dict(landmarks=np.zeros((3, 17, 4))),
                                    dict(landmarks=np.zeros((3, 33, 4), dtype=complex))])
def test_invalid_geometry_or_conventions_rejected(change):
    with pytest.raises(ValueError):
        sequence(**change)


def test_missing_nan_accepted_but_infinity_and_invalid_visibility_rejected():
    points = np.full((3, 33, 4), np.nan)
    assert np.isnan(sequence(landmarks=points).landmarks).all()
    points[0, 0, 3] = 1.1
    with pytest.raises(ValueError, match='visibility'):
        sequence(landmarks=points)
    points[0, 0, 3] = np.inf
    with pytest.raises(ValueError, match='Infinite'):
        sequence(landmarks=points)


def test_events_normalize_name_and_preserve_explicit_provenance():
    event = PoseEvent(EventName.TOP, 1.17, source='predicted', confidence=.99)
    assert event.name == 'Top'
    assert event.source == 'predicted'  # Probability is not empirical reliability.
    for values in [('Unknown', 1., 'manual', None), ('Top', True, 'manual', None),
                   ('Top', complex(1), 'manual', None), ('Top', np.inf, 'manual', None),
                   ('Top', 1., 'imagined', None), ('Top', 1., 'predicted', float('nan')),
                   ('Top', 1., 'predicted', True)]:
        with pytest.raises(ValueError):
            PoseEvent(*values)
