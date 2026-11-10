"""Leakage/resume/software checks with synthetic rows, never training evidence."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sklearn")
from backend.swingsync.research.caddieset import CaddieDataset, FEATURE_NAMES
from backend.swingsync.research import caddie_experiment as experiment


@pytest.fixture(autouse=True)
def single_threaded():
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def data():
    rng = np.random.default_rng(29)
    groups = np.repeat(np.arange(1, 9), 8)
    views = np.tile(np.repeat(["FACEON", "DTL"], 4), 8)
    features = rng.normal(size=(64, 69))
    features[:, 1] = np.nan
    return CaddieDataset(
        features,
        2 * features[:, 0] + groups / 20,
        groups,
        views,
        np.tile(["W1", "I7"], 32),
        np.arange(64),
        FEATURE_NAMES,
        "a" * 64,
        {},
    )


@pytest.mark.parametrize(
    "y,pred,expected",
    [
        ([1, 3], [2, 2], {"count": 2, "mae": 1.0, "rmse": 1.0, "r2": 0.0}),
        ([2, 2], [2, 2], {"count": 2, "mae": 0.0, "rmse": 0.0, "r2": None}),
        ([2], [3], {"count": 1, "mae": 1.0, "rmse": 1.0, "r2": None}),
    ],
)
def test_metric_arithmetic_and_meaningless_r2(y, pred, expected):
    assert experiment.metrics(y, pred) == expected


@pytest.mark.parametrize("y,pred", [([], []), ([1], [float("nan")]), ([1], [1, 2])])
def test_metric_invalid_vectors_rejected(y, pred):
    with pytest.raises(ValueError):
        experiment.metrics(y, pred)


def test_club_baseline_uses_training_only_and_unseen_global_fallback():
    pred, info = experiment.club_mean_predict([1, 3, 8], ["W1", "W1", "I7"], ["W1", "I9"])
    assert pred.tolist() == [2.0, 4.0]
    assert info["unseen_club_count"] == 1


def test_outer_inner_groups_and_rows_never_cross(data):
    for view in ("FACEON", "DTL"):
        covered = []
        for golfer in range(1, 9):
            train, test = experiment.fold_indices(data, view, golfer)
            assert not set(data.groups[train]) & set(data.groups[test])
            assert set(data.groups[test]) == {golfer}
            assert set(data.views[train]) == set(data.views[test]) == {view}
            inner = experiment.grouped_inner_splits(data, train)
            assert sorted(np.concatenate([b for a, b in inner]).tolist()) == train.tolist()
            assert sorted(len(np.unique(data.groups[b])) for a, b in inner) == [2, 2, 3]
            for a, b in inner:
                assert not set(data.groups[a]) & set(data.groups[b])
                assert golfer not in set(data.groups[a]) | set(data.groups[b])
            covered.extend(test.tolist())
        assert sorted(covered) == np.flatnonzero(data.views == view).tolist()


def test_predictor_matrix_contains_no_golfer_or_outcomes(data):
    matrix = experiment.design_matrix(data, np.array([0, 1]))
    assert matrix.shape == (2, 70)
    assert matrix[:, -1].tolist() == data.clubs[:2].tolist()
    bad = SimpleNamespace(X=data.X, clubs=data.clubs, feature_names=("DirectionAngle", *FEATURE_NAMES[1:]))
    with pytest.raises(ValueError, match="allowlist"):
        experiment.design_matrix(bad, [0])


def test_pipeline_allmissing_retained_trainonly_impute_scale_unknownclub():
    train = np.ones((4, 70), dtype=object)
    train[:, :69] = 0.0
    train[:, 0] = [1.0, 3.0, np.nan, np.nan]
    train[:, 1] = np.nan
    train[:, 69] = "W1"
    pipeline = experiment.make_pipeline("ridge", {"alpha": 1}, experiment.EXPECTED)
    pipeline.fit(train, [0.0, 1.0, 0.0, 1.0])
    test = train[:1].copy()
    test[0, 0], test[0, 1], test[0, 69] = 10000.0, 9000.0, "I9"
    assert np.isfinite(pipeline.predict(test)).all()
    summary = experiment._preprocess_summary(pipeline)
    assert summary["numeric_imputer_statistics"][0] == 2.0
    assert summary["numeric_imputer_statistics"][1] == 0.0
    assert summary["club_categories"] == ["W1"]
    assert summary["numeric_missing_indicator_features"] == [0, 1]
    assert summary["scaler_mean"][0] == 2.0


def test_hgb_disables_ungrouped_early_stopping():
    pipeline = experiment.make_pipeline("hgb", experiment.EXPECTED["hgb_candidates"][0], experiment.EXPECTED)
    assert pipeline.named_steps["model"].early_stopping is False
    assert pipeline.named_steps["model"].random_state == 20261110


def test_selection_weights_all_golfers_equally_not_folds_or_rows(monkeypatch):
    groups = np.concatenate([np.repeat(g, g) for g in range(1, 8)])
    n = len(groups)
    x = np.zeros((n, 69))
    x[:, 0] = groups
    data = CaddieDataset(
        x,
        np.zeros(n),
        groups,
        np.repeat("FACEON", n),
        np.repeat("W1", n),
        np.arange(n),
        FEATURE_NAMES,
        "a" * 64,
        {},
    )

    class FixturePipeline:
        def __init__(self, alpha):
            self.alpha = alpha

        def fit(self, x, y):
            return self

        def predict(self, x):
            return (
                np.where(np.asarray(x[:, 0], float) == 1, 10.0, 0.0)
                if self.alpha == 1
                else np.repeat(1.0 if self.alpha == 10 else 2.0, len(x))
            )

    monkeypatch.setattr(
        experiment, "make_pipeline", lambda model, candidate, config: FixturePipeline(candidate["alpha"])
    )
    selected, summary = experiment.select_candidate(data, np.arange(n), "ridge", experiment.EXPECTED)
    assert selected == {"alpha": 10}
    assert summary["candidates"][0]["macro_golfer_mae"] == pytest.approx(10 / 7)
    assert summary["candidates"][1]["macro_golfer_mae"] == 1.0


@pytest.fixture
def fitted_fold(data):
    return experiment.evaluate_fold(data, "FACEON", 1, experiment.EXPECTED)


def test_synthetic_fold_training_summaries_and_resume_validation(data, fitted_fold):
    experiment._validate_fold(fitted_fold, data, "FACEON", 1, experiment.EXPECTED)
    assert fitted_fold["models"]["ridge"]["training_summary"]["numeric_imputer_statistics"][1] == 0
    assert len(fitted_fold["models"]["ridge"]["selection"]["candidates"][0]["per_golfer"]) == 7


@pytest.mark.parametrize("attack", ["metric", "test_row", "inner_group", "inner_mean", "imputer"])
def test_saved_prediction_group_selection_and_preprocess_tampering_rejected(data, fitted_fold, attack):
    value = copy.deepcopy(fitted_fold)
    if attack == "metric":
        value["models"]["ridge"]["metrics"]["mae"] += 1
    if attack == "test_row":
        value["test_row_ids"][0] = 999
    if attack == "inner_group":
        value["models"]["ridge"]["selection"]["inner_splits"][0]["train_golfers"].append(1)
    if attack == "inner_mean":
        value["models"]["ridge"]["selection"]["candidates"][0]["macro_golfer_mae"] += 1
    if attack == "imputer":
        value["models"]["ridge"]["training_summary"]["numeric_imputer_statistics"][0] += 1
    with pytest.raises(ValueError):
        experiment._validate_fold(value, data, "FACEON", 1, experiment.EXPECTED)


def test_partial_results_cannot_be_aggregated(data, fitted_fold):
    with pytest.raises(ValueError, match="Incomplete exact outer coverage"):
        experiment.aggregate([fitted_fold], data, experiment.EXPECTED)


def test_registration_hash_and_exact_policy_are_required(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(experiment.EXPECTED))
    assert experiment.validate_config(path, experiment.sha256(path)) == experiment.EXPECTED
    with pytest.raises(ValueError, match="registration SHA256"):
        experiment.validate_config(path, "0" * 64)
    doc = copy.deepcopy(experiment.EXPECTED)
    doc["hgb_shared"]["early_stopping"] = True
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="fixed registered"):
        experiment.validate_config(path, experiment.sha256(path))


def test_atomic_publication_no_overwrite_or_nonfinite_file(tmp_path):
    dest = tmp_path / "result.json"
    experiment._publish(dest, {"software": True})
    before = dest.read_bytes()
    with pytest.raises(FileExistsError):
        experiment._publish(dest, {"software": False})
    assert dest.read_bytes() == before
    with pytest.raises(ValueError):
        experiment._publish(tmp_path / "bad.json", {"x": float("nan")})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]


def test_global_lock_refuses_duplicate_before_registration(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setattr(experiment, "LOCK", tmp_path / "active.lock")
    with experiment.LOCK.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="shared lock"):
            experiment.run_experiment(
                csv="missing", config="missing", run_dir=tmp_path / "run", expected_config_sha256="0" * 64
            )
    assert not (tmp_path / "run").exists()


@pytest.fixture
def operational_fixture(data, tmp_path, monkeypatch):
    from dataclasses import replace

    csv = tmp_path / "synthetic.csv"
    csv.write_bytes(b"Explicit software fixture; never the real CaddieSet dataset")
    digest = experiment.sha256(csv)
    fixture = replace(data, fingerprint=digest)
    policy = copy.deepcopy(experiment.EXPECTED)
    policy["dataset_sha256"] = digest
    monkeypatch.setattr(experiment, "EXPECTED", policy)
    monkeypatch.setattr(experiment, "LOCK", tmp_path / "active.lock")
    monkeypatch.setattr(experiment, "load_caddieset", lambda _: fixture)
    monkeypatch.setattr(experiment, "runtime_identity", lambda: {"fixture": "software-only"})
    monkeypatch.setattr(experiment, "_process_identity", lambda: {"pid": 1, "process_start": "synthetic"})
    config = tmp_path / "config.json"
    config.write_text(json.dumps(policy))
    return dict(
        csv=csv,
        config=config,
        expected_config_sha256=experiment.sha256(config),
        run_dir=tmp_path / "comparison",
    )


def test_explicit_partial_resume_and_completed_no_refit(operational_fixture, monkeypatch):
    calls = []
    actual = experiment.evaluate_fold

    def spy(data, view, golfer, config):
        calls.append((view, golfer))
        return actual(data, view, golfer, config)

    monkeypatch.setattr(experiment, "evaluate_fold", spy)
    partial = experiment.run_experiment(**operational_fixture, max_folds=1)
    assert partial["state"] == "partial" and len(partial["folds"]) == 1
    assert not (operational_fixture["run_dir"] / "report.json").exists()
    completed = experiment.run_experiment(**operational_fixture, resume=True)
    assert completed["state"] == "completed" and len(completed["folds"]) == 16
    assert calls == [(v, g) for v in ["FACEON", "DTL"] for g in range(1, 9)]
    ledger_path = operational_fixture["run_dir"] / "run.json"
    before = ledger_path.read_bytes()
    monkeypatch.setattr(
        experiment, "evaluate_fold", lambda *args: pytest.fail("Completed resume must not fit")
    )
    assert experiment.run_experiment(**operational_fixture, resume=True)["state"] == "completed"
    assert ledger_path.read_bytes() == before


def test_live_source_drift_fails_without_completed_fold(operational_fixture, fitted_fold, monkeypatch):
    def mutate(*args):
        operational_fixture["csv"].write_bytes(b"changed during synthetic fit")
        return copy.deepcopy(fitted_fold)

    monkeypatch.setattr(experiment, "evaluate_fold", mutate)
    with pytest.raises(ValueError, match="Live dataset/config/runtime/source drift"):
        experiment.run_experiment(**operational_fixture, max_folds=1)
    saved = json.loads((operational_fixture["run_dir"] / "run.json").read_text())
    assert saved["state"] == "failed" and saved["folds"] == []
    assert saved["inflight_fold"]["heldout_golfer"] == 1
    assert not (operational_fixture["run_dir"] / "report.json").exists()


def test_final_ledger_failure_removes_unbound_smoke_report(operational_fixture, monkeypatch):
    original = experiment._publish

    def fail_completed(path, value, **kwargs):
        if path.name == "run.json" and value["state"] == "completed":
            raise OSError("synthetic completed ledger failure")
        return original(path, value, **kwargs)

    monkeypatch.setattr(experiment, "_publish", fail_completed)
    with pytest.raises(OSError, match="synthetic completed ledger failure"):
        experiment.run_experiment(**operational_fixture, smoke=True)
    saved = json.loads((operational_fixture["run_dir"] / "run.json").read_text())
    assert saved["state"] == "failed" and saved["identity"]["mode"] == "smoke"
    assert not (operational_fixture["run_dir"] / "report.json").exists()
