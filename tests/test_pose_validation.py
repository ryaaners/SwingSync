"""Synthetic reference contracts/arithmetic; no real annotation or accuracy claim."""
import copy
import json
import math

import pytest

from backend.swingsync.research.pose_validation import (
    evaluate_reference, load_protocol, make_reference_template, validate_reference,
)


@pytest.fixture
def fixture():
    protocol = load_protocol()
    protocol['artifact'].update(width=640, height=320, frame_indices=[0, 1, 2],
                                context={'camera_view': 'face-on', 'handedness': 'right', 'mirrored': False})
    result = {'schema_version': 1, 'state': 'completed',
              'video_sha256': protocol['artifact']['video_sha256'],
              'context': copy.deepcopy(protocol['artifact']['context']),
              'video': {'width': 640, 'height': 320}, 'identity_ambiguous': False,
              'frames': [{'frame_index': i, 'pts_seconds': 5.0005 + i * .033,
                          'status': 'single_pose', 'pose_count': 1,
                          'landmarks': [[.5, .5, 0., 1.] for _ in range(33)]} for i in range(3)]}
    args = dict(run_sha256=protocol['artifact']['run_sha256'],
                result_sha256=protocol['artifact']['result_sha256'], protocol=protocol)
    reference = make_reference_template(result, **args)
    return result, reference, args


def attest(reference):
    reference['reviewer'] = {'id': 'synthetic-test-operator', 'notes': 'Synthetic fixture only',
                             'model_blinded': True, 'independent': True}


def visible(reference, frame, joint, x, y):
    reference['frames'][frame]['landmarks'][joint].update(status='visible', x_px=x, y_px=y)


def test_template_has_no_coordinates_or_automatic_attestation(fixture):
    result, reference, args = fixture
    assert len(reference['frames']) == 3
    assert reference['reviewer']['id'] == '' and reference['reviewer']['model_blinded'] is False
    assert reference['reviewer']['independent'] is False
    assert all(p['status'] == 'unreviewed' and p['x_px'] is p['y_px'] is None
               for f in reference['frames'] for p in f['landmarks'])
    assert validate_reference(reference, result, **args) == reference
    with pytest.raises(ValueError, match='manual independent/model-blinded'):
        evaluate_reference(reference, result, **args)


def test_aspect_correct_known_pixel_error_and_independent_torso_scale(fixture):
    result, reference, args = fixture
    attest(reference)
    visible(reference, 0, 0, 323, 164)  # model point(.5,.5)->(320,160), radial error5
    for joint, xy in {11: (300, 120), 12: (340, 120), 23: (300, 220), 24: (340, 220)}.items():
        visible(reference, 0, joint, *xy)
        result['frames'][0]['landmarks'][joint] = [xy[0] / 640, xy[1] / 320, 0., 1.]
    report = evaluate_reference(reference, result, **args)
    point = next(p for p in report['point_results'] if p['landmark_id'] == 0)
    assert point['error_px'] == 5 and point['error_reference_torso_lengths'] == .05
    assert report['scored_count'] == report['eligible_visible_count'] == 5
    assert report['pixel_error']['mean'] == 1 and report['pixel_error']['rmse'] == pytest.approx(math.sqrt(5))
    assert report['reference_slot_count'] == 99 and report['reference_state_counts']['unreviewed'] == 94
    assert report['independent_sources'] == 1
    assert report['validated_for_coaching'] is report['app_integration_permitted'] is False
    assert report['advice'] == [] and report['outcome_integration_permitted'] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('model_status', ['no_pose', 'malformed_pose'])
def test_missing_model_returns_are_abstentions_with_visible_denominator(fixture, model_status):
    result, reference, args = fixture
    attest(reference)
    visible(reference, 0, 0, 320, 160)
    visible(reference, 2, 0, 321, 160)
    result['frames'][2]['status'] = model_status
    result['frames'][2]['landmarks'] = [[None] * 4 for _ in range(33)]
    report = evaluate_reference(reference, result, **args)
    assert report['eligible_visible_count'] == 2 and report['scored_count'] == 1
    assert report['model_abstention_count'] == 1 and report['abstentions'] == {'model_' + model_status: 1}
    assert report['pixel_error']['mean'] == 0
    assert report['reference_torso_normalized_error']['count'] == 0
    assert report['scored_points_without_reference_scale'] == 1


def test_multiple_anywhere_suppresses_all_reference_comparisons(fixture):
    result, reference, args = fixture
    attest(reference)
    visible(reference, 0, 0, 320, 160)
    result['frames'][2].update(status='multiple_poses', pose_count=2)
    report = evaluate_reference(reference, result, **args)
    assert report['scored_count'] == 0 and report['model_abstention_count'] == 1
    assert report['abstentions'] == {'sequence_identity_ambiguous': 1}
    assert report['pixel_error']['mean'] is None


@pytest.mark.parametrize('model_point,reason', [
    ([None] * 4, 'model_point_missing_or_presence_filtered'),
    ([.5, .5, 0, .499], 'model_visibility_below_registered_floor'),
    ([1.1, .5, 0, 1], 'model_point_out_of_image'),
    ([.5, .5, 0, 1.1], 'model_point_missing_or_presence_filtered'),
    ([True, .5, 0, 1], 'model_point_missing_or_presence_filtered'),
])
def test_engineering_filters_count_abstentions_without_coordinate_clamping(fixture, model_point, reason):
    result, reference, args = fixture
    attest(reference)
    visible(reference, 0, 0, 320, 160)
    result['frames'][0]['landmarks'][0] = model_point
    report = evaluate_reference(reference, result, **args)
    assert report['eligible_visible_count'] == report['model_abstention_count'] == 1
    assert report['abstentions'] == {reason: 1}


@pytest.mark.parametrize('status', ['occluded', 'out_of_frame', 'unreviewed'])
def test_nonvisible_references_remain_explicit_and_never_scored(fixture, status):
    result, reference, args = fixture
    attest(reference)
    visible(reference, 0, 0, 320, 160)
    reference['frames'][0]['landmarks'][1]['status'] = status
    report = evaluate_reference(reference, result, **args)
    assert report['eligible_visible_count'] == 1 and report['reference_slot_count'] == 99
    assert report['reference_state_counts'][status] >= 1
    reference['frames'][0]['landmarks'][1]['x_px'] = 2
    with pytest.raises(ValueError, match='must not invent'):
        evaluate_reference(reference, result, **args)


@pytest.mark.parametrize('field,value', [('model_blinded', False), ('independent', False), ('id', '')])
def test_manual_review_attestations_required_for_scoring(fixture, field, value):
    result, reference, args = fixture
    attest(reference)
    visible(reference, 0, 0, 320, 160)
    reference['reviewer'][field] = value
    with pytest.raises(ValueError, match='manual independent/model-blinded'):
        evaluate_reference(reference, result, **args)


@pytest.mark.parametrize('field,value', [('x_px', float('nan')), ('y_px', float('inf')),
                                      ('x_px', True), ('x_px', -.1), ('y_px', 320.1)])
def test_reference_coordinate_units_and_bounds_strict(fixture, field, value):
    result, reference, args = fixture
    visible(reference, 0, 0, 320, 160)
    reference['frames'][0]['landmarks'][0][field] = value
    with pytest.raises(ValueError, match='finite image-edge pixel'):
        validate_reference(reference, result, **args)


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(coordinate_system='normalized'),
    lambda r: r.update(video_sha256='0' * 64),
    lambda r: r.update(run_sha256='0' * 64),
    lambda r: r.update(result_sha256='0' * 64),
    lambda r: r.update(schema_version=True),
    lambda r: r['context'].update(mirrored=0),
    lambda r: r['context'].update(handedness='left'),
    lambda r: r['frames'].pop(),
    lambda r: r['frames'].reverse(),
    lambda r: r['frames'][0].update(pts_seconds=5.0),
    lambda r: r['frames'][0].update(width=320),
    lambda r: r['frames'][0]['landmarks'].pop(),
    lambda r: r['frames'][0]['landmarks'][1].update(landmark_id=0),
    lambda r: r['frames'][0]['landmarks'][0].update(landmark_id=True),
    lambda r: r['frames'][0]['landmarks'][0].update(status=[]),
    lambda r: r.update(model_outcomes={'DirectionAngle': 1}),
])
def test_forged_or_incomplete_reference_contract_refused(fixture, mutation):
    result, reference, args = fixture
    mutation(reference)
    with pytest.raises(ValueError):
        validate_reference(reference, result, **args)


def test_geometry_and_hashes_cannot_enter_other_study_silently(fixture):
    result, reference, args = fixture
    with pytest.raises(ValueError, match='registered pilot'):
        make_reference_template(result, **(args | {'run_sha256': '0' * 64}))
    result['video']['width'] = 320
    with pytest.raises(ValueError, match='raster differs'):
        validate_reference(reference, result, **args)


def test_reviewed_but_no_visible_points_is_no_accuracy_result(fixture):
    result, reference, args = fixture
    attest(reference)
    with pytest.raises(ValueError, match='No visible manual reference'):
        evaluate_reference(reference, result, **args)


def test_degenerate_independent_reference_torso_is_not_replaced_by_model_scale(fixture):
    result, reference, args = fixture
    attest(reference)
    for j in (0, 11, 12, 23, 24):
        visible(reference, 0, j, 320, 160)
    report = evaluate_reference(reference, result, **args)
    assert report['scored_count'] == 5 and report['reference_torso_normalized_error']['count'] == 0
    assert report['scored_points_without_reference_scale'] == 5


def test_returned_reference_and_report_cannot_mutate_original_inputs(fixture):
    result, reference, args = fixture
    admitted = validate_reference(reference, result, **args)
    admitted['frames'][0]['landmarks'][0]['x_px'] = 99
    assert reference['frames'][0]['landmarks'][0]['x_px'] is None
    before = copy.deepcopy(result)
    attest(reference)
    visible(reference, 0, 0, 323, 164)
    report = evaluate_reference(reference, result, **args)
    assert result == before
    report['point_results'][0]['error_px'] = 0
    assert reference['frames'][0]['landmarks'][0]['x_px'] == 323


def legacy_projection(u, v, rect_width, rect_height, angle):
    """Pinned upstream no-dimensions branch, transcribed only for source audit."""
    x, y = u - .5, v - .5
    return ((math.cos(angle) * x - math.sin(angle) * y) * rect_width + .5,
            (math.sin(angle) * x + math.cos(angle) * y) * rect_height + .5)


def physical_projection(u, v, rect_width, rect_height, width, height, angle):
    # Independent construction: center a point in physical pixel ROI, rotate it,
    # then normalize by raster axes. This is not a patch to native predictions.
    x, y = (u - .5) * rect_width * width, (v - .5) * rect_height * height
    return ((math.cos(angle) * x - math.sin(angle) * y) / width + .5,
            (math.sin(angle) * x + math.cos(angle) * y) / height + .5)


@pytest.mark.parametrize('width,height', [(640, 320), (160, 160), (320, 640)])
@pytest.mark.parametrize('angle', [0, math.pi / 2, -.78])
def test_projection_square_physical_roi_equivalence_even_rectangular_source(width, height, angle):
    side = 100
    rw, rh = side / width, side / height
    assert legacy_projection(.73, .27, rw, rh, angle) == pytest.approx(
        physical_projection(.73, .27, rw, rh, width, height, angle))


def test_projection_nonsquare_roi_counterexample_does_not_justify_silent_fix():
    # Same normalized width/height on a 2:1 raster is NOT a square pixel ROI.
    legacy = legacy_projection(.75, .5, .5, .5, math.pi / 2)
    physical = physical_projection(.75, .5, .5, .5, 640, 320, math.pi / 2)
    assert legacy == pytest.approx((.5, .625))
    assert physical == pytest.approx((.5, .75))
    assert abs(legacy[1] - physical[1]) * 320 == pytest.approx(40)
