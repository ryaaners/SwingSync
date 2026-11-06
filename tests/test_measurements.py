"""Synthetic geometry checks only: no real pose accuracy or advice validation."""
from dataclasses import replace

import numpy as np
import pytest

from backend.swingsync.movement.contracts import PoseEvent, PoseSequence
from backend.swingsync.movement.measurements import extract_measurements


def fixture(*, camera_view='face-on', handedness='right', mirrored=False):
    times = np.array([10, 10.1, 10.25, 10.4, 10.6, 10.7, 10.9])
    landmarks = np.zeros((len(times), 33, 4))
    landmarks[..., :2] = .5
    landmarks[..., 3] = 1
    landmarks[:, (0, 7, 8), :2] = (.5, .2)
    landmarks[-1, (0, 7, 8), :2] += (.04, .03)
    landmarks[:, 11, :2], landmarks[:, 12, :2] = (.4, .4), (.6, .4)
    landmarks[:, 23, :2], landmarks[:, 24, :2] = (.4, .8), (.6, .8)
    landmarks[:, 13, :2], landmarks[:, 15, :2] = (.4, .5), (.45, .5)
    landmarks[:, 14, :2], landmarks[:, 16, :2] = (.6, .5), (.6, .6)
    if mirrored:
        landmarks[..., 0] = 1 - landmarks[..., 0]
    sequence = PoseSequence(times, landmarks, 200, 100, camera_view, handedness, mirrored)
    events = (PoseEvent('Address', 10), PoseEvent('Top', 10.6), PoseEvent('Impact', 10.9))
    return sequence, events


def rows(sequence, events, **kwargs):
    report = extract_measurements(sequence, events, **kwargs)
    return {row.name: row for row in report.measurements}


@pytest.mark.parametrize('camera_view', ['face-on', 'down-the-line'])
def test_actual_variable_pts_geometry_and_evidence(camera_view):
    sequence, events = fixture(camera_view=camera_view)
    measured = rows(sequence, events)
    assert len(measured) == 6
    assert all(row.status == 'measured' for row in measured.values())
    assert measured['backswing_duration_seconds'].value == pytest.approx(.6)
    assert measured['downswing_duration_seconds'].value == pytest.approx(.3)
    assert measured['backswing_downswing_ratio'].value == pytest.approx(2)
    head = measured['normalized_head_max_displacement']
    assert head.value == pytest.approx(np.hypot(8, 3) / 40)
    assert head.frame_indices == tuple(range(7))
    assert head.frame_timestamps_seconds == tuple(sequence.timestamps_seconds)
    assert head.interval_seconds == (10, 10.9)
    assert head.event_names == ('Address', 'Impact')
    assert head.events == (events[0], events[-1])
    assert head.camera_view == camera_view
    assert head.unit == 'address_torso_lengths'
    assert measured['lead_elbow_angle_at_top'].value == pytest.approx(90)
    assert measured['trail_elbow_angle_at_top'].value == pytest.approx(180)
    assert measured['lead_elbow_angle_at_top'].frame_indices == (4,)
    assert all(row.reference and row.method and 'unvalidated' in row.reliability for row in measured.values())


def test_uniform_resize_and_different_aspect_preserving_pixel_geometry():
    sequence, events = fixture()
    uniform = replace(sequence, frame_width=400, frame_height=200)
    points = sequence.landmarks.copy()
    points[..., 0] /= 2
    different_aspect = replace(sequence, frame_width=400, landmarks=points)
    expected = rows(sequence, events)
    for transformed in (uniform, different_aspect):
        actual = rows(transformed, events)
        for name in expected:
            assert actual[name].value == pytest.approx(expected[name].value)


def test_elbow_angle_uses_pixel_aspect_not_normalized_square():
    sequence, events = fixture()
    points = sequence.landmarks.copy()
    points[:, 15, :2] = (.45, .6)
    sequence = replace(sequence, landmarks=points)
    assert rows(sequence, events)['lead_elbow_angle_at_top'].value == pytest.approx(135)


@pytest.mark.parametrize('handedness,expected', [('right', (90, 180)), ('left', (180, 90))])
@pytest.mark.parametrize('mirrored', [False, True])
def test_handedness_and_mirrored_coordinates_preserve_anatomical_sides(handedness, expected, mirrored):
    sequence, events = fixture(handedness=handedness, mirrored=mirrored)
    measured = rows(sequence, events)
    assert measured['lead_elbow_angle_at_top'].value == pytest.approx(expected[0])
    assert measured['trail_elbow_angle_at_top'].value == pytest.approx(expected[1])
    assert measured['normalized_head_max_displacement'].value == pytest.approx(np.hypot(8, 3) / 40)


@pytest.mark.parametrize('landmark,component,value', [
    (0, 3, .49), (7, 0, np.nan), (8, 3, np.nan), (0, 0, -0.1), (7, 1, 1.1),
])
def test_head_trajectory_fails_closed_for_missing_invisible_and_offscreen_samples(landmark, component, value):
    sequence, events = fixture()
    points = sequence.landmarks.copy()
    points[2, landmark, component] = value
    measured = rows(replace(sequence, landmarks=points), events)
    head = measured['normalized_head_max_displacement']
    assert head.status == 'unavailable' and head.value is None
    assert head.reason == 'head_or_baseline_landmarks_unavailable'
    assert measured['backswing_duration_seconds'].status == 'measured'


def test_head_requires_visible_nondegenerate_address_scale():
    sequence, events = fixture()
    points = sequence.landmarks.copy()
    points[0, 23, 3] = .1
    assert rows(replace(sequence, landmarks=points), events)[
        'normalized_head_max_displacement'].reason == 'head_or_baseline_landmarks_unavailable'
    points = sequence.landmarks.copy()
    points[0, (23, 24), :2] = points[0, (11, 12), :2]
    assert rows(replace(sequence, landmarks=points), events)[
        'normalized_head_max_displacement'].reason == 'degenerate_address_body_scale'


@pytest.mark.parametrize('component,value', [(0, np.nan), (1, 1.2), (3, .1)])
def test_elbow_landmark_missing_only_disables_that_arm(component, value):
    sequence, events = fixture()
    points = sequence.landmarks.copy()
    points[4, 13, component] = value
    measured = rows(replace(sequence, landmarks=points), events)
    assert measured['lead_elbow_angle_at_top'].reason == 'elbow_landmarks_unavailable'
    assert measured['trail_elbow_angle_at_top'].status == 'measured'


def test_degenerate_arm_unavailable():
    sequence, events = fixture()
    points = sequence.landmarks.copy()
    points[4, 13, :2] = points[4, 11, :2]
    assert rows(replace(sequence, landmarks=points), events)[
        'lead_elbow_angle_at_top'].reason == 'degenerate_projected_arm'


def test_pose_gap_suppresses_trajectory_without_replacing_pts_or_tempo():
    sequence, events = fixture()
    # Omitted decoded samples leave an observed .5-second interval.
    keep = [0, 1, 4, 5, 6]
    sequence = replace(sequence, timestamps_seconds=sequence.timestamps_seconds[keep],
                       landmarks=sequence.landmarks[keep])
    measured = rows(sequence, events)
    assert measured['normalized_head_max_displacement'].reason == 'sample_gap_exceeds_budget'
    assert measured['backswing_duration_seconds'].value == pytest.approx(.6)
    assert measured['lead_elbow_angle_at_top'].frame_timestamps_seconds == (10.6,)
    assert rows(sequence, events, max_sample_gap_seconds=.51)[
        'normalized_head_max_displacement'].status == 'measured'


def test_offsample_annotation_preserves_timestamp_and_never_snaps_spatial_sample():
    sequence, events = fixture()
    events = (events[0], replace(events[1], timestamp_seconds=10.55), events[2])
    measured = rows(sequence, events)
    backswing = measured['backswing_duration_seconds']
    assert backswing.value == pytest.approx(.55)
    assert backswing.interval_seconds == (10, 10.55)
    assert backswing.frame_indices == (0, 3, 4)
    assert backswing.frame_timestamps_seconds == (10, 10.4, 10.6)
    assert measured['lead_elbow_angle_at_top'].reason == 'event_not_sampled'
    assert measured['lead_elbow_angle_at_top'].events[0].timestamp_seconds == 10.55


def test_unknown_view_disables_geometry_but_preserves_time_measurement():
    sequence, events = fixture(camera_view='unknown')
    measured = rows(sequence, events)
    assert measured['backswing_downswing_ratio'].status == 'measured'
    for name in ('normalized_head_max_displacement', 'lead_elbow_angle_at_top', 'trail_elbow_angle_at_top'):
        assert measured[name].reason == 'camera_view_required'


@pytest.mark.parametrize('events,reason', [
    ((PoseEvent('Address', 10), PoseEvent('Top', 10.6), PoseEvent('Top', 10.7), PoseEvent('Impact', 10.9)),
     'duplicate_event_names'),
    ((PoseEvent('Address', 10), PoseEvent('Top', 10.9), PoseEvent('Impact', 10.6)),
     'events_not_strictly_ordered'),
    ((PoseEvent('Address', 10), PoseEvent('Top', 10.6), PoseEvent('Impact', 10.6)),
     'events_not_strictly_ordered'),
    ((PoseEvent('Address', 10), PoseEvent('Top', 10.6), PoseEvent('Impact', 10.9), PoseEvent('Finish', 10.7)),
     'events_not_strictly_ordered'),
])
def test_duplicate_or_inconsistent_events_suppress_all_measurements(events, reason):
    sequence, _ = fixture()
    assert all(row.reason == reason and row.value is None for row in rows(sequence, events).values())


def test_missing_events_and_outside_coverage_return_reasons():
    sequence, events = fixture()
    missing = rows(sequence, events[1:])
    assert missing['backswing_duration_seconds'].reason == 'missing_required_event'
    assert missing['downswing_duration_seconds'].status == 'measured'
    assert missing['normalized_head_max_displacement'].interval_seconds is None
    outside = rows(sequence, (replace(events[0], timestamp_seconds=9.9), *events[1:]))
    assert outside['backswing_duration_seconds'].reason == 'event_outside_sample_coverage'
    assert outside['normalized_head_max_displacement'].reason == 'event_outside_sample_coverage'


def test_empty_events_never_fabricate_values():
    sequence, _ = fixture()
    assert all(row.reason == 'missing_required_event' and row.value is None
               for row in rows(sequence, ()).values())


def test_manual_correction_recomputes_without_mutating_prior_report():
    sequence, manual = fixture()
    predicted = tuple(replace(event, source='predicted', confidence=.99) for event in manual)
    before = extract_measurements(sequence, predicted)
    corrected = (predicted[0], replace(manual[1], timestamp_seconds=10.7), predicted[2])
    after = extract_measurements(sequence, corrected)
    before_rows = {row.name: row for row in before.measurements}
    after_rows = {row.name: row for row in after.measurements}
    assert before_rows['backswing_duration_seconds'].value == pytest.approx(.6)
    assert after_rows['backswing_duration_seconds'].value == pytest.approx(.7)
    assert after_rows['backswing_duration_seconds'].events[1].source == 'manual'
    assert before_rows['backswing_duration_seconds'].events[1].confidence == .99
    assert before.advice == after.advice == ()
    assert 'No validated' in after.advice_suppression_reason
    assert all('unvalidated' in row.reliability for row in after.measurements)
    assert extract_measurements(sequence, manual).advice == ()


def test_semantic_event_order_does_not_depend_on_input_list_order_and_z_is_unused():
    sequence, events = fixture()
    points = sequence.landmarks.copy()
    points[..., 2] = np.nan
    assert rows(sequence, events) == rows(replace(sequence, landmarks=points), events[::-1])


@pytest.mark.parametrize('kwargs', [
    {'visibility_threshold': -0.1}, {'visibility_threshold': 1.1}, {'visibility_threshold': True},
    {'visibility_threshold': np.nan}, {'max_sample_gap_seconds': 0},
    {'max_sample_gap_seconds': np.inf}, {'max_sample_gap_seconds': False},
])
def test_invalid_engineering_filter_config_rejected(kwargs):
    sequence, events = fixture()
    with pytest.raises(ValueError):
        extract_measurements(sequence, events, **kwargs)
