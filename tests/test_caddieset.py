"""CaddieSet data-contract software fixtures, not model-performance evidence."""
from dataclasses import replace
import csv
import hashlib
import io

import numpy as np
import pytest

from backend.swingsync.research import caddieset as cd


def fixture_rows():
    result = []
    for view in cd.VIEWS:
        row = dict.fromkeys(cd.HEADER, '1')
        row.update(View=view, ClubType='W1', GolferId='1', DirectionAngle='-2.5')
        row['0-LOWER-ANGLE'] = ''
        row['0-SHOULDER-ANGLE'] = '#NAME?'
        row['0-SPINE-ANGLE'] = 'inf'
        result.append(row)
    return result


def write_csv(tmp_path, rows=None, header=cd.HEADER):
    path = tmp_path / 'fixture.csv'
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    writer.writerow(header)
    for row in fixture_rows() if rows is None else rows:
        writer.writerow([row[name] for name in header] if isinstance(row, dict) else row)
    path.write_text(buffer.getvalue())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixture(tmp_path, rows=None):
    path, digest = write_csv(tmp_path, rows)
    return cd.load_caddieset(path, expected_sha256=digest)


def test_exact_predictor_allowlist_and_missing_tokens(tmp_path):
    d = load_fixture(tmp_path)
    assert d.X.shape == (2, 69) and d.X.dtype == np.float64
    assert d.y.tolist() == [-2.5, -2.5]
    assert np.isnan(d.X[:, :3]).all()
    assert d.feature_names == cd.FEATURE_NAMES
    assert not set(d.feature_names) & (set(cd.OUTCOME_COLUMNS) | {'GolferId', 'View', 'ClubType'})
    assert d.audit['predictive_metadata'] == ['ClubType']
    assert d.audit['known_positive_inf_feature_cells'] == 2
    assert d.audit['known_name_error_cells'] == 2
    assert not d.audit['preprocessing_fitted']


def test_no_target_or_other_outcome_predictors_even_when_their_values_change(tmp_path):
    rows = fixture_rows()
    first = load_fixture(tmp_path, rows)
    for name in cd.OUTCOME_COLUMNS:
        for row in rows:
            row[name] = '999'
    second = load_fixture(tmp_path, rows)
    np.testing.assert_equal(first.X, second.X)
    assert second.y.tolist() == [999, 999]


def test_duplicate_and_alternate_view_records_remain_same_golfer_group(tmp_path):
    rows = fixture_rows()
    rows.append(rows[0].copy())
    d = load_fixture(tmp_path, rows)
    assert len(d.y) == 3 and d.row_ids.tolist() == [0, 1, 2]
    assert d.groups.tolist() == [1, 1, 1]
    assert d.audit['exact_duplicate_records_retained'] == 1
    assert d.audit['candidate_cross_view_outcome_groups_audit_only'] == 1
    assert not d.audit['authoritative_shot_identifier_available']


def test_arrays_are_owned_readonly_copies(tmp_path):
    d = load_fixture(tmp_path)
    source = np.ones((2, 69))
    copy = replace(d, X=source)
    source[0, 0] = 99
    assert copy.X[0, 0] == 1
    for arr in (copy.X, copy.y, copy.groups, copy.views, copy.clubs, copy.row_ids):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = arr[0]


def test_default_sha_refuses_synthetic_or_mutated_release(tmp_path):
    path, digest = write_csv(tmp_path)
    with pytest.raises(ValueError, match='fingerprint'):
        cd.load_caddieset(path)
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='fingerprint'):
        cd.load_caddieset(path, expected_sha256=digest)


@pytest.mark.parametrize('header', [cd.HEADER[::-1], (*cd.HEADER[:-1], 'DirectionAngle-copy'),
                                  (*cd.HEADER[:-1], 'DirectionAngle'), cd.HEADER[:-1]])
def test_changed_reordered_duplicate_or_renamed_schema_refused(tmp_path, header):
    # Explicit lists also support intentional duplicate header fixtures.
    path, digest = write_csv(tmp_path, [["1"] * len(header)], header)
    with pytest.raises(ValueError, match='schema'):
        cd.load_caddieset(path, expected_sha256=digest)


@pytest.mark.parametrize('token', ['+inf', '-inf', 'Inf', 'Infinity', 'NaN', 'nan', '#DIV/0!',
                                   '#N/A', ' 2', '2 ', '1_000', '1e999', 'true'])
def test_unregistered_feature_tokens_fail_closed(tmp_path, token):
    rows = fixture_rows()
    rows[0]['1-HEAD-LOC'] = token
    with pytest.raises(ValueError):
        load_fixture(tmp_path, rows)


@pytest.mark.parametrize('token', ['', '#NAME?', 'inf', 'NaN'])
def test_target_missing_or_invalid_is_not_imputed(tmp_path, token):
    rows = fixture_rows()
    rows[0]['DirectionAngle'] = token
    with pytest.raises(ValueError):
        load_fixture(tmp_path, rows)


def test_other_ball_outcomes_are_checked_but_never_imputed_or_used(tmp_path):
    rows = fixture_rows()
    rows[0]['BallSpeed'] = '#NAME?'
    with pytest.raises(ValueError):
        load_fixture(tmp_path, rows)


@pytest.mark.parametrize('column,value', [('View', 'FACE-ON'), ('ClubType', 'Unknown'),
                                         ('GolferId', '1.0'), ('GolferId', '9'), ('GolferId', '')])
def test_unsupported_metadata_fails_without_guessing(tmp_path, column, value):
    rows = fixture_rows()
    rows[0][column] = value
    with pytest.raises(ValueError):
        load_fixture(tmp_path, rows)


@pytest.mark.parametrize('length', [79, 81])
def test_missing_or_extra_row_fields_refused(tmp_path, length):
    path, digest = write_csv(tmp_path, [['1'] * length])
    with pytest.raises(ValueError, match='80 columns'):
        cd.load_caddieset(path, expected_sha256=digest)


def test_zero_records_refused(tmp_path):
    path, digest = write_csv(tmp_path, [])
    with pytest.raises(ValueError, match='no records'):
        cd.load_caddieset(path, expected_sha256=digest)


@pytest.mark.parametrize('change', [
    {'feature_names': ('DirectionAngle',) + cd.FEATURE_NAMES[1:]},
    {'groups': np.array([1., 2.])}, {'groups': np.array([1, 2**63], dtype=np.uint64)},
    {'row_ids': np.array([0, 2**63], dtype=np.uint64)},
    {'row_ids': np.array([0, 0])}, {'y': np.array([0., np.nan])},
    {'X': np.full((2, 69), np.inf)}, {'fingerprint': 'invented'},
])
def test_dataset_constructor_rejects_predictor_or_identity_corruption(tmp_path, change):
    with pytest.raises(ValueError):
        replace(load_fixture(tmp_path), **change)
