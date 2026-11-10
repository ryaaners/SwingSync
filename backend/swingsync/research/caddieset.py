"""Audited CaddieSet tabular inputs; no uploaded-video feature equivalence.

Outcome columns are parsed for audit/target only and never enter X. Every record
of a golfer must remain in one partition, retaining alternate views/duplicates.
"""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import re

import numpy as np

CSV_SHA256 = '646459b081863e244d75efe6f09fad4d34750d2fed7fde44db654e7720a9a306'
REVISION = '3c73d9d40580bb8a5a10711ad1fa10735a205ffe'
OUTCOME_COLUMNS = ('Distance', 'Carry', 'LrDistanceOut', 'DirectionAngle',
                   'SpinBack', 'SpinSide', 'SpinAxis', 'BallSpeed')
FEATURE_NAMES = (
    '0-LOWER-ANGLE', '0-SHOULDER-ANGLE', '0-SPINE-ANGLE', '0-STANCE-RATIO', '0-UPPER-TILT',
    '1-HEAD-LOC', '1-HIP-LINE', '1-HIP-ROTATION', '1-HIP-SHIFTED', '1-LEFT-ARM-ANGLE',
    '1-RIGHT-ARM-ANGLE', '1-SHOULDER-ANGLE', '1-SHOULDER-LOC', '1-SPINE-ANGLE',
    '2-HEAD-LOC', '2-HIP-ANGLE', '2-HIP-LINE', '2-HIP-ROTATION', '2-HIP-SHIFTED',
    '2-LEFT-ARM-ANGLE', '2-SHOULDER-ANGLE', '2-SHOULDER-LOC', '2-UPPER-TILT',
    '3-HEAD-LOC', '3-HIP-ANGLE', '3-HIP-LINE', '3-HIP-ROTATION', '3-HIP-SHIFTED',
    '3-LEFT-LEG-ANGLE', '3-RIGHT-ARM-ANGLE', '3-RIGHT-DISTANCE', '3-RIGHT-LEG-ANGLE',
    '3-SHOULDER-ANGLE', '3-SHOULDER-LOC',
    '4-HEAD-LOC', '4-HIP-ANGLE', '4-HIP-HANGING-BACK', '4-HIP-LINE', '4-HIP-ROTATION',
    '4-HIP-SHIFTED', '4-RIGHT-ARM-ANGLE', '4-RIGHT-ARMPIT-ANGLE',
    '4-SHOULDER-HANGING-BACK', '4-SPINE-ANGLE',
    '5-HEAD-LOC', '5-HIP-HANGING-BACK', '5-HIP-LINE', '5-HIP-SHIFTED',
    '5-LEFT-ARM-ANGLE', '5-LEFT-LEG-ANGLE', '5-RIGHT-ARM-ANGLE', '5-SHOULDER-ANGLE',
    '5-SHOULDER-HANGING-BACK', '5-SPINE-ANGLE',
    '6-HEAD-LOC', '6-HIP-ANGLE', '6-HIP-LINE', '6-HIP-SHIFTED', '6-LEFT-ARM-ANGLE',
    '6-RIGHT-LEG-ANGLE', '6-SHOULDER-ANGLE', '6-SPINE-ANGLE', '6-WEIGHT-SHIFT',
    '7-FINISH-ANGLE', '7-HIP-ANGLE', '7-HIP-LINE', '7-HIP-SHIFTED',
    '7-SHOULDER-ANGLE', '7-SPINE-ANGLE',
)
HEADER = ('View', 'ClubType', *OUTCOME_COLUMNS, 'GolferId', *FEATURE_NAMES)
VIEWS = ('FACEON', 'DTL')
CLUBS = ('W1', 'W3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9')
MAX_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class CaddieDataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    views: np.ndarray
    clubs: np.ndarray
    row_ids: np.ndarray
    feature_names: tuple[str, ...]
    fingerprint: str
    audit: dict

    def __post_init__(self):
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError('Only the exact 69 audited swing features may enter X')
        if not isinstance(self.fingerprint, str) or re.fullmatch('[0-9a-f]{64}', self.fingerprint) is None:
            raise ValueError('Dataset byte SHA256 required')
        x, y = np.asarray(self.X), np.asarray(self.y)
        groups, row_ids = np.asarray(self.groups), np.asarray(self.row_ids)
        views, clubs = np.asarray(self.views), np.asarray(self.clubs)
        n = len(y) if y.ndim == 1 else 0
        if (not n or x.shape != (n, 69) or x.dtype.kind not in 'iuf'
                or y.shape != (n,) or y.dtype.kind not in 'iuf'
                or np.isinf(x).any() or not np.isfinite(y).all()):
            raise ValueError('Finite target and numeric [N,69] features with NaN missingness required')
        if (groups.shape != (n,) or groups.dtype.kind not in 'iu' or np.any(groups < 1)
                or np.any(groups > np.iinfo(np.int64).max)
                or row_ids.shape != (n,) or row_ids.dtype.kind not in 'iu' or np.any(row_ids < 0)
                or np.any(row_ids > np.iinfo(np.int64).max)
                or len(np.unique(row_ids)) != n):
            raise ValueError('Positive integer golfer groups and unique nonnegative row IDs required')
        if (views.shape != (n,) or clubs.shape != (n,)
                or not set(views.tolist()) <= set(VIEWS) or not set(clubs.tolist()) <= set(CLUBS)):
            raise ValueError('Explicit audited camera views and club metadata required')
        for name, values, dtype in (
            ('X', x, np.float64), ('y', y, np.float64), ('groups', groups, np.int64),
            ('row_ids', row_ids, np.int64), ('views', views, np.str_), ('clubs', clubs, np.str_),
        ):
            owned = np.array(values, dtype=dtype, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
        object.__setattr__(self, 'feature_names', FEATURE_NAMES)


def _numeric(token, *, missing, label):
    if missing and token in ('', '#NAME?', 'inf'):
        return float('nan')
    if not token or re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', token) is None:
        raise ValueError(f'Unexpected numeric token in {label}')
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Unexpected nonnumeric token in {label}') from exc
    if not math.isfinite(value):
        raise ValueError(f'Infinite/nonfinite numeric token in {label}')
    return value


def load_caddieset(path, *, expected_sha256=CSV_SHA256):
    """Load exact ordered schema and a supplied byte fingerprint, with no fitting.

    The default binds the reviewed release. Synthetic software fixtures may pass
    their own explicit SHA; operational experiments must independently bind CSV_SHA256.
    No full-data missingness selector, imputer, scaler or feature fit runs here.
    """
    if not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('Expected lowercase dataset SHA256 required')
    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError('CaddieSet CSV exceeds byte budget or is empty')
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    if fingerprint != expected_sha256:
        raise ValueError('CaddieSet CSV byte fingerprint differs')
    reader = csv.reader(io.StringIO(raw.decode('utf-8-sig')), strict=True)
    if tuple(next(reader, ())) != HEADER:
        raise ValueError('Exact ordered 80-column CaddieSet schema required')
    rows = []
    for row in reader:
        if len(row) != len(HEADER):
            raise ValueError('Missing/extra CSV field; rows must have exactly 80 columns')
        rows.append(tuple(row))
        if len(rows) > 100_000:
            raise ValueError('CaddieSet row budget exceeded')
    if not rows:
        raise ValueError('CaddieSet contains no records')
    x, y, groups, views, clubs = [], [], [], [], []
    empty_cells = name_errors = infinity_errors = 0
    infinity_columns = Counter()
    by_golfer = defaultdict(lambda: Counter())
    by_golfer_club = defaultdict(Counter)
    by_view_club = defaultdict(Counter)
    outcome_groups = defaultdict(set)
    for index, row in enumerate(rows):
        view, club, golfer = row[0], row[1], row[10]
        if view not in VIEWS or club not in CLUBS or not re.fullmatch('[1-8]', golfer):
            raise ValueError('Unknown view, club or published golfer ID')
        outcome_values = tuple(_numeric(row[HEADER.index(name)], missing=False, label=name)
                               for name in OUTCOME_COLUMNS)
        features = [_numeric(row[11 + j], missing=True, label=name) for j, name in enumerate(FEATURE_NAMES)]
        x.append(features)
        y.append(outcome_values[OUTCOME_COLUMNS.index('DirectionAngle')])
        groups.append(int(golfer))
        views.append(view)
        clubs.append(club)
        empty_cells += row.count('')
        name_errors += row.count('#NAME?')
        infinity_errors += row.count('inf')
        infinity_columns.update(name for j, name in enumerate(FEATURE_NAMES) if row[11 + j] == 'inf')
        by_golfer[golfer][view] += 1
        by_golfer_club[golfer][club] += 1
        by_view_club[view][club] += 1
        # Audit only; never a predictor, split selector, authoritative shot ID or model input.
        outcome_groups[(golfer, club, *outcome_values)].add(view)
    matrix = np.asarray(x, dtype=np.float64)
    view_array = np.asarray(views)
    missing = {
        view: {feature: int(np.isnan(matrix[view_array == view, j]).sum())
               for j, feature in enumerate(FEATURE_NAMES)} for view in VIEWS
    }
    audit = {
        'schema_version': 1, 'dataset_sha256': fingerprint, 'upstream_revision': REVISION,
        'rows': len(rows), 'columns': len(HEADER), 'swing_feature_count': 69,
        'feature_names': list(FEATURE_NAMES), 'predictive_metadata': ['ClubType'],
        'excluded_outcomes': list(OUTCOME_COLUMNS), 'group_only': ['GolferId'],
        'partition_only': ['View'], 'independent_golfer_count': len(set(groups)),
        'view_counts': dict(Counter(views)),
        'golfer_view_counts': {g: dict(sorted(v.items())) for g, v in sorted(by_golfer.items())},
        'golfer_club_counts': {g: dict(sorted(v.items())) for g, v in sorted(by_golfer_club.items())},
        'view_club_counts': {g: dict(sorted(v.items())) for g, v in sorted(by_view_club.items())},
        'empty_cells': empty_cells, 'known_name_error_cells': name_errors,
        'known_positive_inf_feature_cells': infinity_errors,
        'positive_inf_feature_cells_by_column': dict(sorted(infinity_columns.items())),
        'exact_duplicate_records_retained': len(rows) - len(set(rows)),
        'candidate_cross_view_outcome_groups_audit_only': sum(len(v) > 1 for v in outcome_groups.values()),
        'authoritative_shot_identifier_available': False,
        'target': {'name': 'DirectionAngle', 'unit': 'degrees', 'interpretation': 'signed initial ball direction, not curvature',
                   'missing_count': 0, 'min': min(y), 'max': max(y)},
        'missing_feature_cells_by_view': missing,
        'preprocessing_fitted': False, 'rows_deduplicated': False,
        'integration_permitted': False, 'advice_permitted': False,
        'limits': [
            'Only eight golfers in the reviewed release; frames/view records are not independent golfers',
            'Club and view proportions are confounded with golfer; per-fold variation must be reported',
            'No authoritative shot IDs; whole-golfer grouping retains possible alternate views/duplicates together',
            'Known empty cells, #NAME? and literal inf feature tokens become NaN; fit all transforms in training folds',
            'WEIGHT-SHIFT is an upstream projected ankle-to-pelvis line angle, not measured pressure or force',
            'No source videos/extractor/handedness field or MediaPipe feature equivalence established',
        ],
    }
    if fingerprint == CSV_SHA256 and (
        len(rows) != 1757 or len(set(groups)) != 8 or empty_cells != 58917 or name_errors != 11 or infinity_errors != 11
        or len(rows) - len(set(rows)) != 6
    ):
        raise ValueError('Reviewed release counts differ despite byte registration')
    return CaddieDataset(matrix, np.asarray(y), np.asarray(groups), view_array,
                        np.asarray(clubs), np.arange(len(rows)), FEATURE_NAMES, fingerprint, audit)
