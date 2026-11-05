"""Synthetic score fixtures establish protocol behavior, never ML accuracy."""
import json

import numpy as np
import pytest

from backend.swingsync.research.evaluation import (
    EVENT_NAMES,
    evaluate_clip,
    events_from_labels,
    summarize_evaluations,
)


def scores_for(predictions, length):
    scores = np.full((length, 9), .001)
    scores[:, 8] = .992
    for event, index in enumerate(predictions):
        scores[index] = .001
        scores[index, event] = .992
    return scores


def test_matches_reference_calculation_and_uses_real_pts():
    events = np.array([1, 3, 5, 7, 9, 11, 13, 15])
    predictions = np.array([2, 3, 5, 7, 9, 11, 13, 17])
    scores = scores_for(predictions, 20)
    times = 10 + np.arange(20, dtype=float) ** 2 / 100
    result = evaluate_clip(scores, events, timestamps_seconds=times)
    # Independently reconstruct the published equations from their specification.
    expected = [int(np.argsort(scores[:, event])[-1]) for event in range(8)]
    assert result.predicted_frames == tuple(expected)
    assert result.tolerance_frames == 1
    assert result.correct == (True, True, True, True, True, True, True, False)
    assert result.signed_error_frames == (1, 0, 0, 0, 0, 0, 0, 2)
    assert result.absolute_error_ms == pytest.approx([30, 0, 0, 0, 0, 0, 0, 640])
    assert result.numpy_version == np.__version__
    assert result.to_dict()['pce_percent'] == 87.5
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(('gap', 'tolerance'), [(30, 1), (45, 2), (75, 2), (105, 4)])
def test_numpy_rounding_boundary_is_not_python_truncation(gap, tolerance):
    events = [0, 1, 2, 3, 4, gap, gap + 1, gap + 2]
    assert evaluate_clip(scores_for(events, gap + 3), events).tolerance_frames == tolerance


def test_independent_peaks_are_not_silently_order_constrained():
    events = [0, 2, 4, 6, 8, 10, 12, 14]
    predictions = [14, 12, 10, 8, 6, 4, 2, 0]
    result = evaluate_clip(scores_for(predictions, 16), events)
    assert result.predicted_frames == tuple(predictions)
    assert summarize_evaluations([result])['order_constrained_decoding'] is False


def test_ties_follow_runtime_default_argsort_including_background_tie():
    scores = np.full((80, 9), 1 / 9)
    events = [0, 2, 4, 6, 8, 10, 12, 14]
    result = evaluate_clip(scores, events)
    expected = tuple(int(np.argsort(scores[:, i])[-1]) for i in range(8))
    assert result.predicted_frames == expected
    assert result.tie_policy == 'numpy-default-argsort-last-index'


def test_padding_cannot_affect_predictions_tolerance_or_timestamp_errors():
    events = np.arange(8)
    scores = scores_for(events, 10)
    times = np.arange(10) * .04
    expected = evaluate_clip(scores, events, timestamps_seconds=times)
    padded = np.concatenate((scores, np.full((3, 9), np.nan)))
    padded_times = np.concatenate((times, [np.nan] * 3))
    actual = evaluate_clip(padded, events, valid_mask=np.array([True] * 10 + [False] * 3),
                           timestamps_seconds=padded_times)
    assert actual == expected
    labels = np.r_[np.arange(8), 8, 8, -100, -100, -100]
    assert events_from_labels(labels, valid_mask=np.arange(13) < 10) == tuple(events)


def test_label_conversion_rejects_duplicate_and_missing_events():
    assert events_from_labels(np.r_[8, np.arange(8), 8]) == tuple(range(1, 9))
    for labels in ([0, 1, 2, 3, 4, 5, 6, 6], [0, 1, 2, 3, 4, 5, 6],
                   [-1, 0, 1, 2, 3, 4, 5, 6]):
        with pytest.raises(ValueError):
            events_from_labels(labels)
    with pytest.raises(ValueError):
        events_from_labels(np.arange(8, dtype=float))


def test_aggregate_weights_every_clip_and_event_equally():
    events = np.arange(8)
    perfect = evaluate_clip(scores_for(events, 20), events)
    wrong = evaluate_clip(scores_for(events + 10, 20), events)
    report = summarize_evaluations([perfect, wrong])
    assert report['pce'] == .5
    assert report['pce_percent'] == 50
    assert report['mae_frames'] == 5
    assert report['mae_ms'] is None
    assert set(report['per_event']) == set(EVENT_NAMES)
    assert all(value['pce'] == .5 for value in report['per_event'].values())
    timed = evaluate_clip(scores_for(events, 20), events, timestamps_seconds=np.arange(20) * .04)
    with pytest.raises(ValueError, match='partial timestamp'):
        summarize_evaluations([perfect, timed])
    with pytest.raises(ValueError, match='empty'):
        summarize_evaluations([])


@pytest.mark.parametrize('events', [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 20],
                                   [0, 1, 2, 3, 4, 5, 5, 7], [0., 1, 2, 3, 4, 5, 6, 7],
                                   np.array([20, 9, 8, 7, 6, 5, 4, 3], dtype=np.uint8)])
def test_invalid_events_are_rejected(events):
    with pytest.raises(ValueError):
        evaluate_clip(np.full((20, 9), 1 / 9), events)


@pytest.mark.parametrize('mutation', ['nan', 'inf', 'logits', 'negative', 'shape'])
def test_invalid_probabilities_are_rejected(mutation):
    scores = np.full((20, 9), 1 / 9)
    if mutation == 'nan':
        scores[0, 0] = np.nan
    elif mutation == 'inf':
        scores[0, 0] = np.inf
    elif mutation == 'logits':
        scores *= 10
    elif mutation == 'negative':
        scores[0, 0] = -.1
    else:
        scores = scores[:, :8]
    with pytest.raises(ValueError):
        evaluate_clip(scores, np.arange(8))


def test_masks_and_times_require_full_clip_alignment():
    scores = np.full((20, 9), 1 / 9)
    for mask in (np.ones(19, dtype=bool), np.ones(20), np.arange(20) != 9, np.zeros(20, dtype=bool)):
        with pytest.raises(ValueError):
            evaluate_clip(scores, np.arange(8), valid_mask=mask)
    for times in (np.arange(19), np.zeros(20), np.r_[np.arange(19), np.nan]):
        with pytest.raises(ValueError):
            evaluate_clip(scores, np.arange(8), timestamps_seconds=times)


def test_nonchronological_annotations_match_official_occupied_frame_targets():
    # Real audited annotation677 has MFT before Impact, with tolerance affected.
    raw = np.array([92, 114, 131, 146, 158, 168, 167, 242])
    scores = scores_for(np.sort(raw), 250)
    labels = np.full(250, 8, dtype=int)
    labels[raw] = np.arange(8)
    assert events_from_labels(labels) == tuple(raw)
    official_events = np.flatnonzero(labels < 8)
    result = evaluate_clip(scores, raw)
    assert result.annotation_event_frames == tuple(raw)
    assert result.annotation_order_anomaly is True
    assert result.event_frames == tuple(official_events)
    assert result.tolerance_frames == 2  # actual class Impact gap would produce3
    assert all(result.correct)
    report = summarize_evaluations([result])
    assert report['annotation_order_anomaly_count'] == 1
    assert 'chronological' in report['target_mapping']


@pytest.mark.parametrize('times', [np.arange(20) + 1j, np.arange(20, dtype=complex),
                                  np.arange(20).astype(str)])
def test_reject_nonreal_or_nonnumeric_timestamps_before_float_conversion(times):
    with pytest.raises(ValueError, match='real numeric'):
        evaluate_clip(np.full((20, 9), 1 / 9), np.arange(8), timestamps_seconds=times)
