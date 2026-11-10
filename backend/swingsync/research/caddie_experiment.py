"""Registered, golfer-grouped tabular research; no uploaded-video inference.

Every imputer/scaler/encoder is fit inside the relevant training partition. A
single atomic ledger records each completed outer fold so resume cannot mistake
an unregistered orphan prediction file for completed work.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import subprocess
import time
import tomllib
import uuid

import numpy as np

from .caddieset import CSV_SHA256, FEATURE_NAMES, load_caddieset

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / 'artifacts/training/active-training.lock'
TABULAR_LOCK = ROOT / 'tabular_runtime/uv.lock'
EXPECTED = {
    'schema_version': 1, 'dataset_sha256': CSV_SHA256, 'target': 'DirectionAngle',
    'seed': 20261110, 'views': ['FACEON', 'DTL'], 'outer_golfers': list(range(1, 9)),
    'inner_splits': 3, 'ridge_alphas': [1, 10, 100],
    'hgb_candidates': [{'max_leaf_nodes': 7, 'l2_regularization': 1},
                       {'max_leaf_nodes': 15, 'l2_regularization': 10}],
    'hgb_shared': {'learning_rate': 0.05, 'max_iter': 100, 'min_samples_leaf': 20,
                   'early_stopping': False},
}
MODELS = ('club_mean', 'ridge', 'hgb')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _load(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    def invalid(_):
        raise ValueError('Nonfinite JSON value')
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def _publish(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.part')
    try:
        with temporary.open('x') as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_config(path, expected_sha256):
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)
            or sha256(path) != expected_sha256):
        raise ValueError('Independently recorded registration SHA256 required')
    config = _load(path)
    if _canonical(config) != _canonical(EXPECTED):
        raise ValueError('Configuration differs from fixed registered comparison')
    return config


def _sklearn():
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from threadpoolctl import threadpool_limits
    return {'ColumnTransformer': ColumnTransformer, 'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
            'SimpleImputer': SimpleImputer, 'Ridge': Ridge, 'GroupKFold': GroupKFold, 'Pipeline': Pipeline,
            'OneHotEncoder': OneHotEncoder, 'StandardScaler': StandardScaler, 'threadpool_limits': threadpool_limits}


def runtime_identity():
    _sklearn()
    if (sys.version_info[:2] != (3, 11) or sys.prefix == sys.base_prefix
            or Path(sys.prefix).resolve() == (ROOT / '.venv').resolve()):
        raise ValueError('Use isolated Python3.11 tabular runtime')
    packages = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in importlib.metadata.distributions()}
    locked = {p['name']: p['version'] for p in tomllib.loads(TABULAR_LOCK.read_text())['package']}
    if any(locked.get(k) != v for k, v in packages.items()):
        raise ValueError('Installed package outside tabular lock')
    pins = {'numpy': '2.2.5', 'scipy': '1.15.3', 'scikit-learn': '1.7.2', 'pandas': '2.3.3'}
    if any(packages.get(k) != v for k, v in pins.items()):
        raise ValueError('Tabular direct package pins differ')
    sources = [ROOT/'backend/swingsync/research/caddie_experiment.py',
               ROOT/'backend/swingsync/research/caddieset.py', ROOT/'scripts/run_caddieset.py', ROOT/'experiments/caddieset-v1.md',
               ROOT/'docs/caddieset-contract.md']
    return {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'python_executable_sha256': sha256(sys.executable), 'installed_packages': packages,
            'tabular_lock_sha256': sha256(TABULAR_LOCK), 'root_lock_sha256': sha256(ROOT/'uv.lock'),
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources}, 'threads': 1}


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if (y.ndim != 1 or y.shape != prediction.shape or not len(y)
            or not np.isfinite(y).all() or not np.isfinite(prediction).all()):
        raise ValueError('Finite aligned outcome/prediction vectors required')
    errors = y - prediction
    sse = float(np.dot(errors, errors))
    centered = y - y.mean()
    sst = float(np.dot(centered, centered))
    return {'count': len(y), 'mae': float(np.abs(errors).mean()),
            'rmse': float(np.sqrt(sse / len(y))),
            'r2': 1 - sse/sst if len(y) > 1 and sst > 0 else None}


def club_mean_predict(train_y, train_clubs, test_clubs):
    train_y = np.asarray(train_y, dtype=np.float64)
    train_clubs, test_clubs = np.asarray(train_clubs), np.asarray(test_clubs)
    if len(train_y) != len(train_clubs) or not len(train_y) or not np.isfinite(train_y).all():
        raise ValueError('Finite training targets and clubs required')
    means = {str(c): float(train_y[train_clubs == c].mean()) for c in np.unique(train_clubs)}
    fallback = float(train_y.mean())
    return np.array([means.get(str(c), fallback) for c in test_clubs]), {
        'club_means': means, 'global_mean_fallback': fallback,
        'unseen_club_count': int(sum(str(c) not in means for c in test_clubs))}


def design_matrix(data, indices):
    """Only the strict numeric swing allowlist and ClubType enter predictors."""
    if tuple(data.feature_names) != tuple(FEATURE_NAMES) or data.X.shape[1] != 69:
        raise ValueError('Exact swing feature allowlist required')
    return np.column_stack((data.X[indices].astype(object), data.clubs[indices].astype(object)))


def make_pipeline(model, candidate, config, n_features=69):
    api = _sklearn()
    impute = api['SimpleImputer'](strategy='median', keep_empty_features=True, add_indicator=True)
    steps = [('imputer', impute)]
    if model == 'ridge':
        steps.append(('scale', api['StandardScaler']()))
        estimator = api['Ridge'](alpha=candidate['alpha'], solver='svd')
    elif model == 'hgb':
        estimator = api['HistGradientBoostingRegressor'](
            **config['hgb_shared'], **candidate, random_state=config['seed'])
    else:
        raise ValueError('Unknown learned model')
    preprocessing = api['ColumnTransformer']([
        ('numeric', api['Pipeline'](steps), list(range(n_features))),
        ('club', api['OneHotEncoder'](handle_unknown='ignore', sparse_output=False), [n_features]),
    ], sparse_threshold=0)
    return api['Pipeline']([('preprocess', preprocessing), ('model', estimator)])


def grouped_inner_splits(data, train_indices, n_splits=3):
    groups = data.groups[train_indices]
    if len(np.unique(groups)) < n_splits:
        raise ValueError('Insufficient training golfers for grouped inner validation')
    api = _sklearn()
    splits, seen = [], []
    for a, b in api['GroupKFold'](n_splits=n_splits, shuffle=False).split(np.zeros(len(groups)), groups=groups):
        fit, validation = np.asarray(train_indices)[a], np.asarray(train_indices)[b]
        if set(data.groups[fit]) & set(data.groups[validation]):
            raise ValueError('Inner golfer leakage')
        splits.append((fit, validation))
        seen.extend(validation.tolist())
    if sorted(seen) != sorted(np.asarray(train_indices).tolist()):
        raise ValueError('Inner validation must cover each training row once')
    return splits


def candidates(model, config):
    return ([{'alpha': a} for a in config['ridge_alphas']] if model == 'ridge'
            else config['hgb_candidates'])


def select_candidate(data, train_indices, model, config):
    """Equal weight to all validation golfers, not equal weight to unequal folds."""
    splits = grouped_inner_splits(data, train_indices, config['inner_splits'])
    scores = []
    for candidate in candidates(model, config):
        golfer_mae = []
        for fit, validation in splits:
            pipeline = make_pipeline(model, candidate, config)
            pipeline.fit(design_matrix(data, fit), data.y[fit])
            pred = pipeline.predict(design_matrix(data, validation))
            for golfer in np.unique(data.groups[validation]):
                mask = data.groups[validation] == golfer
                golfer_mae.append({'golfer': int(golfer),
                                   'mae': metrics(data.y[validation][mask], pred[mask])['mae']})
        if len(golfer_mae) != len(np.unique(data.groups[train_indices])):
            raise ValueError('Every inner-validation golfer must contribute once')
        golfer_mae.sort(key=lambda s: s['golfer'])
        scores.append({'parameters': candidate, 'macro_golfer_mae': float(np.mean([s['mae'] for s in golfer_mae])),
                       'per_golfer': sorted(golfer_mae, key=lambda s: s['golfer'])})
    selected = min(range(len(scores)), key=lambda i: scores[i]['macro_golfer_mae'])
    split_evidence = [{'train_golfers': sorted(int(g) for g in np.unique(data.groups[a])),
                       'validation_golfers': sorted(int(g) for g in np.unique(data.groups[b])),
                       'train_row_ids': data.row_ids[a].tolist(), 'validation_row_ids': data.row_ids[b].tolist()}
                      for a, b in splits]
    return scores[selected]['parameters'], {'candidates': scores, 'selected_index': selected,
                                            'selection_metric': 'macro_validation_golfer_mae',
                                            'inner_splits': split_evidence}


def _preprocess_summary(pipeline):
    transform = pipeline.named_steps['preprocess']
    numeric = transform.named_transformers_['numeric']
    imputer = numeric.named_steps['imputer']
    result = {'numeric_imputer_statistics': imputer.statistics_.tolist(),
              'numeric_missing_indicator_features': imputer.indicator_.features_.tolist(),
              'club_categories': transform.named_transformers_['club'].categories_[0].tolist()}
    if 'scale' in numeric.named_steps:
        result.update(scaler_mean=numeric.named_steps['scale'].mean_.tolist(),
                      scaler_scale=numeric.named_steps['scale'].scale_.tolist())
    return result


def fold_indices(data, view, golfer):
    train = np.flatnonzero((data.views == view) & (data.groups != golfer))
    test = np.flatnonzero((data.views == view) & (data.groups == golfer))
    if not len(train) or not len(test) or golfer in set(data.groups[train]):
        raise ValueError('Outer fold has empty partitions or golfer leakage')
    return train, test


def evaluate_fold(data, view, golfer, config):
    train, test = fold_indices(data, view, golfer)
    baseline, baseline_info = club_mean_predict(data.y[train], data.clubs[train], data.clubs[test])
    models = {'club_mean': {'prediction': baseline.tolist(), 'metrics': metrics(data.y[test], baseline),
                            'training_summary': baseline_info}}
    for model in ('ridge', 'hgb'):
        parameters, selection = select_candidate(data, train, model, config)
        pipeline = make_pipeline(model, parameters, config)
        pipeline.fit(design_matrix(data, train), data.y[train])
        prediction = pipeline.predict(design_matrix(data, test))
        models[model] = {'parameters': parameters, 'selection': selection, 'prediction': prediction.tolist(),
                         'metrics': metrics(data.y[test], prediction), 'training_summary': _preprocess_summary(pipeline)}
    return {'view': view, 'heldout_golfer': golfer,
            'train_golfers': sorted(int(g) for g in np.unique(data.groups[train])),
            'train_row_ids': data.row_ids[train].tolist(), 'test_row_ids': data.row_ids[test].tolist(),
            'y_true': data.y[test].tolist(), 'models': models}


def _validate_fold(record, data, view, golfer, config):
    train, test = fold_indices(data, view, golfer)
    if (record['view'] != view or record['heldout_golfer'] != golfer
            or record['train_row_ids'] != data.row_ids[train].tolist()
            or record['test_row_ids'] != data.row_ids[test].tolist()
            or record['train_golfers'] != sorted(int(g) for g in np.unique(data.groups[train]))
            or record['y_true'] != data.y[test].tolist() or set(record['models']) != set(MODELS)):
        raise ValueError('Saved outer fold source/group/row coverage differs')
    for model in MODELS:
        saved = record['models'][model]
        if saved['metrics'] != metrics(data.y[test], saved['prediction']):
            raise ValueError('Saved prediction/metric arithmetic differs')
        if model == 'club_mean':
            prediction, summary = club_mean_predict(data.y[train], data.clubs[train], data.clubs[test])
            if saved['prediction'] != prediction.tolist() or saved['training_summary'] != summary:
                raise ValueError('Saved baseline is not training-only club mean')
        else:
            selection = saved['selection']
            expected_parameters = candidates(model, config)
            if [c['parameters'] for c in selection['candidates']] != expected_parameters:
                raise ValueError('Saved inner candidates differ')
            scores = []
            expected_golfers = sorted(int(g) for g in np.unique(data.groups[train]))
            for row in selection['candidates']:
                if [p['golfer'] for p in row['per_golfer']] != expected_golfers:
                    raise ValueError('Saved inner validation golfer coverage differs')
                maes = [p['mae'] for p in row['per_golfer']]
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in maes):
                    raise ValueError('Invalid saved inner MAE')
                score = float(np.mean(maes))
                if row['macro_golfer_mae'] != score:
                    raise ValueError('Saved inner macro score differs')
                scores.append(score)
            selected = min(range(len(scores)), key=lambda i: scores[i])
            if selection['selected_index'] != selected or saved['parameters'] != expected_parameters[selected]:
                raise ValueError('Saved candidate selection differs')
            expected_splits = []
            for a, b in grouped_inner_splits(data, train, config['inner_splits']):
                expected_splits.append({'train_golfers': sorted(int(g) for g in np.unique(data.groups[a])),
                                        'validation_golfers': sorted(int(g) for g in np.unique(data.groups[b])),
                                        'train_row_ids': data.row_ids[a].tolist(),
                                        'validation_row_ids': data.row_ids[b].tolist()})
            if selection['inner_splits'] != expected_splits:
                raise ValueError('Saved inner grouping differs')
            summary = saved['training_summary']
            raw = data.X[train]
            statistics = [float(np.median(column[np.isfinite(column)]))
                          if np.isfinite(column).any() else 0.0 for column in raw.T]
            indicators = np.flatnonzero(np.isnan(raw).any(axis=0)).tolist()
            if (not np.allclose(summary['numeric_imputer_statistics'], statistics, rtol=1e-12, atol=1e-12)
                    or summary['numeric_missing_indicator_features'] != indicators
                    or summary['club_categories'] != np.unique(data.clubs[train]).tolist()):
                raise ValueError('Saved preprocessing summary differs from outer training partition')
            if model == 'ridge':
                filled = np.where(np.isnan(raw), np.asarray(statistics)[None, :], raw)
                transformed = np.column_stack((filled, np.isnan(raw[:, indicators]).astype(float)))
                if (not np.allclose(summary['scaler_mean'], transformed.mean(axis=0), rtol=1e-12, atol=1e-12)
                        or not np.isfinite(summary['scaler_scale']).all()
                        or np.any(np.asarray(summary['scaler_scale']) <= 0)):
                    raise ValueError('Saved scaler summary differs from outer training partition')


def aggregate(folds, data, config):
    expected = [(v, g) for v in config['views'] for g in config['outer_golfers']]
    if [(f['view'], f['heldout_golfer']) for f in folds] != expected:
        raise ValueError('Incomplete exact outer coverage; no aggregate')
    for record, (view, golfer) in zip(folds, expected):
        _validate_fold(record, data, view, golfer, config)
    report = {'schema_version': 1, 'target': 'DirectionAngle', 'unit': 'degrees_initial_direction',
              'dataset_sha256': data.fingerprint, 'independent_golfers': 8,
              'views': {}, 'app_integration_permitted': False, 'feature_equivalence_validated': False,
              'coaching_permitted': False, 'advice': [],
              'limitations': ['Eight golfers; frames/view records are not independent golfers',
                              'No authoritative shot identifiers; entire golfers keep potential pairs together',
                              'Duplicates retained; view models and held-out records remain correlated',
                              'Club use is confounded with golfer identity',
                              'DirectionAngle is initial direction, not slice/hook curvature',
                              'Feature importance/associations do not imply causal swing fixes',
                              'MediaPipe feature equivalence is unestablished; research-only output']}
    for view in config['views']:
        selected = [f for f in folds if f['view'] == view]
        report['views'][view] = {}
        for model in MODELS:
            y = np.concatenate([f['y_true'] for f in selected])
            prediction = np.concatenate([f['models'][model]['prediction'] for f in selected])
            variation = {}
            for metric in ('mae', 'rmse', 'r2'):
                values = [f['models'][model]['metrics'][metric] for f in selected
                          if f['models'][model]['metrics'][metric] is not None]
                variation[metric] = {'fold_count': len(values),
                                     'mean': float(np.mean(values)) if values else None,
                                     'std_population': float(np.std(values)) if values else None,
                                     'min': min(values) if values else None, 'max': max(values) if values else None}
            report['views'][view][model] = {
                'pooled_row_weighted': metrics(y, prediction), 'macro_golfer_fold_variation': variation,
                'per_fold': [{'heldout_golfer': f['heldout_golfer'], **f['models'][model]['metrics']}
                             for f in selected]}
    return report


@contextmanager
def _guard():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another expensive job owns the shared lock') from exc
        yield


def _process_identity():
    try:
        started = subprocess.check_output(['ps', '-p', str(os.getpid()), '-o', 'lstart='],
                                          text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('Process start identity unavailable; no fit launched') from exc
    if not started:
        raise ValueError('Process start identity unavailable; no fit launched')
    return {'pid': os.getpid(), 'process_start': started}


def _now():
    return datetime.now(timezone.utc).isoformat()


def run_experiment(*, csv, config, run_dir, expected_config_sha256, resume=False, max_folds=None, smoke=False):
    csv, config, run_dir = map(Path, (csv, config, run_dir))
    if max_folds is not None and (type(max_folds) is not int or max_folds < 1):
        raise ValueError('max_folds must be a positive integer')
    with _guard():
        registered = validate_config(config, expected_config_sha256)
        data = load_caddieset(csv)
        if data.fingerprint != registered['dataset_sha256'] or sha256(csv) != data.fingerprint:
            raise ValueError('CSV differs from registered dataset')
        environment = runtime_identity()
        identity = {'csv_sha256': data.fingerprint, 'config_sha256': expected_config_sha256,
                    'config': registered, 'runtime': environment, 'mode': 'smoke' if smoke else 'comparison'}
        ledger_path, report_path = run_dir/'run.json', run_dir/'report.json'
        if resume:
            ledger = _load(ledger_path)
            if ledger.get('identity') != identity or ledger.get('schema_version') != 1:
                raise ValueError('Resume dataset/config/runtime/source/mode drift')
            if smoke and ledger.get('state') != 'completed':
                raise ValueError('Interrupted smoke is not resumed automatically; inspect it')
        else:
            if run_dir.exists():
                raise ValueError('Run directory exists; inspect saved identity and use explicit resume')
            run_dir.mkdir(parents=True, exist_ok=False)
            ledger = {'schema_version': 1, 'job_id': uuid.uuid4().hex, 'identity': identity,
                      'created_at': _now(), 'pid': os.getpid(), 'process_identity': _process_identity(),
                      'state': 'initializing', 'inflight_fold': None,
                      'folds': [], 'fold_sha256': [], 'attempts': [], 'audit': data.audit,
                      'app_integration_permitted': False, 'advice': []}
            _publish(ledger_path, ledger)
        plan = [(v, g) for v in registered['views'] for g in registered['outer_golfers']]
        if len(ledger['folds']) != len(ledger['fold_sha256']) or len(ledger['folds']) > len(plan):
            raise ValueError('Invalid saved fold inventory')
        for record, digest, (view, golfer) in zip(ledger['folds'], ledger['fold_sha256'], plan):
            if _hash(record) != digest:
                raise ValueError('Saved fold SHA256 mismatch')
            _validate_fold(record, data, view, golfer, registered)
        if ledger['state'] == 'completed':
            if sha256(report_path) != ledger['report_sha256']:
                raise ValueError('Completed report SHA256 mismatch')
            report = _load(report_path)
            if not smoke and report['results'] != aggregate(ledger['folds'], data, registered):
                raise ValueError('Completed aggregate differs')
            return ledger
        if report_path.exists():
            raise ValueError('Unbound report exists; inspect before continuing')
        prior_inflight = ledger.get('inflight_fold')
        process = _process_identity()
        ledger.update(state='running', pid=os.getpid(), process_identity=process)
        attempt = {'id': uuid.uuid4().hex, 'started_at': _now(), 'pid': os.getpid(),
                   'starting_fold_count': len(ledger['folds']), 'process_identity': process,
                   'restarted_incomplete_fold': prior_inflight}
        ledger['attempts'].append(attempt)
        _publish(ledger_path, ledger, replace=True)
        started = time.perf_counter()
        try:
            api = _sklearn()
            with api['threadpool_limits'](limits=1):
                if smoke:
                    ledger['inflight_fold'] = {'mode': 'train_only_smoke', 'attempt_id': attempt['id']}
                    _publish(ledger_path, ledger, replace=True)
                    train, _ = fold_indices(data, 'FACEON', 1)
                    selected_groups = sorted(np.unique(data.groups[train]))[:2]
                    selected = np.concatenate([train[data.groups[train] == g][:32] for g in selected_groups])
                    baseline, _ = club_mean_predict(data.y[selected], data.clubs[selected], data.clubs[selected])
                    if not np.isfinite(baseline).all():
                        raise ValueError('Smoke baseline produced nonfinite output')
                    for model in ('ridge', 'hgb'):
                        p = make_pipeline(model, candidates(model, registered)[0], registered)
                        p.fit(design_matrix(data, selected), data.y[selected])
                        if not np.isfinite(p.predict(design_matrix(data, selected))).all():
                            raise ValueError('Smoke produced nonfinite output')
                    results = {'scope': 'train-only software fit smoke, no held-out metric',
                               'train_golfers': [int(g) for g in selected_groups],
                               'train_row_ids': data.row_ids[selected].tolist(), 'models_fit': list(MODELS),
                               'heldout_rows_used': 0, 'app_integration_permitted': False, 'advice': []}
                else:
                    additional = 0
                    for view, golfer in plan[len(ledger['folds']):]:
                        if max_folds is not None and additional >= max_folds:
                            break
                        ledger['inflight_fold'] = {'view': view, 'heldout_golfer': golfer, 'attempt_id': attempt['id']}
                        _publish(ledger_path, ledger, replace=True)
                        record = evaluate_fold(data, view, golfer, registered)
                        if (sha256(csv) != data.fingerprint or sha256(config) != expected_config_sha256
                                or runtime_identity() != environment):
                            raise ValueError('Live dataset/config/runtime/source drift')
                        _validate_fold(record, data, view, golfer, registered)
                        ledger['folds'].append(record)
                        ledger['fold_sha256'].append(_hash(record))
                        ledger['inflight_fold'] = None
                        _publish(ledger_path, ledger, replace=True)
                        additional += 1
                    if len(ledger['folds']) != len(plan):
                        ledger.update(state='partial', updated_at=_now())
                        attempt.update(state='partial', completed_at=_now(), elapsed_seconds=time.perf_counter()-started)
                        _publish(ledger_path, ledger, replace=True)
                        return ledger
                    results = aggregate(ledger['folds'], data, registered)
                if (sha256(csv) != data.fingerprint or sha256(config) != expected_config_sha256
                        or runtime_identity() != environment):
                    raise ValueError('Live dataset/config/runtime/source drift')
                report = {'schema_version': 1, 'job_id': ledger['job_id'], 'identity': identity,
                          'completed_at': _now(), 'results': results}
                _publish(report_path, report)
                ledger.update(state='completed', inflight_fold=None, completed_at=report['completed_at'],
                              report_sha256=sha256(report_path))
                attempt.update(state='completed', completed_at=_now(), elapsed_seconds=time.perf_counter()-started)
                _publish(ledger_path, ledger, replace=True)
                return ledger
        except BaseException as exc:
            report_path.unlink(missing_ok=True)
            ledger.update(state='failed', error_type=type(exc).__name__, error=str(exc)[:2000], failed_at=_now())
            attempt.update(state='failed', elapsed_seconds=time.perf_counter()-started)
            _publish(ledger_path, ledger, replace=True)
            raise
