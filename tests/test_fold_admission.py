"""Synthetic witnesses test admission behavior, not real training or accuracy.

Only this fixture replaces the independent known hashes and protocol validator;
production admits exactly the separately reviewed 015 witness and registration.
"""
from pathlib import Path
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from test_fold_cache_audit import toy as _toy_fixture  # noqa: E402 -- shared synthetic fixture
from backend.swingsync.research import fold_admission as admission  # noqa: E402
from backend.swingsync.research import fold_cache_audit as audit  # noqa: E402


toy = _toy_fixture


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@pytest.fixture
def witnessed(toy, monkeypatch):
    options, protocol, entries, dataset, _ = toy
    audit.run_fold_cache_audit(**options)
    state_path = options["output"] / "run.json"
    report_path = options["output"] / "audit.json"
    monkeypatch.setattr(admission, "KNOWN_AUDIT_RUN_SHA256", audit.sha(state_path))
    monkeypatch.setattr(admission, "KNOWN_AUDIT_REPORT_SHA256", audit.sha(report_path))
    return ({"annotations": options["annotations"], "manifests_dir": options["manifests_dir"],
             "audit_run": state_path, "protocol_path": options["protocol_path"], "fold": 2},
            protocol, entries, dataset)


def reseal(options, monkeypatch, report_edit=None, state_edit=None):
    # Deliberate hypothetical independently witnessed invalid artifacts exercise
    # semantic rejection separately from the unmodified production SHA gate.
    state_path = options["audit_run"]
    report_path = state_path.with_name("audit.json")
    state, report = audit._read(state_path), audit._read(report_path)
    if report_edit:
        report_edit(report)
    write(report_path, report)
    state["audit_sha256"] = audit.sha(report_path)
    if state_edit:
        state_edit(state)
    write(state_path, state)
    monkeypatch.setattr(admission, "KNOWN_AUDIT_RUN_SHA256", audit.sha(state_path))
    monkeypatch.setattr(admission, "KNOWN_AUDIT_REPORT_SHA256", audit.sha(report_path))


def test_production_witness_hashes_are_fixed():
    assert admission.KNOWN_AUDIT_RUN_SHA256 == "fedd0ab2d2e51cf3e22c6c4eb453cd0629e4fb6fc5fc29db03b677070109951f"
    assert admission.KNOWN_AUDIT_REPORT_SHA256 == "282f540880ca111cae2723c971951f7e577d67858ce6762beea1301f49b9666e"


@pytest.mark.parametrize("fold", [2, 3, 4])
def test_admission_returns_exact_train_inner_paths_and_train_only_counts(witnessed, fold):
    options, _, entries, _ = witnessed
    result = admission.admit_fold(**{**options, "fold": fold})
    partitions = result.manifest["partitions"]
    expected = set(partitions["train"]["clip_ids"] + partitions["validation"]["clip_ids"])
    assert set(result.cache_paths) == expected
    assert not set(result.cache_paths) & set(partitions["test"]["clip_ids"])
    assert all(path == Path(entries[i]["cache_path"]) for i, path in result.cache_paths.items())
    assert result.immutable["fold"] == fold
    assert result.immutable["outer_test_evaluated"] is False
    assert result.immutable["validated_for_serving"] is False
    report = audit._read(options["audit_run"].with_name("audit.json"))
    assert result.class_counts_train_only == report["folds"][str(fold)]["class_counts_train_only"]
    assert len(result.cache_paths) == 12
    admission.recheck(result)


@pytest.mark.parametrize("fold", [1, 0, 5, True, 2.0, "2", None])
def test_unregistered_fold_refused_before_any_witness_open(witnessed, fold, monkeypatch):
    options, *_ = witnessed
    monkeypatch.setattr(admission, "_known_json", lambda *a: pytest.fail("witness opened"))
    with pytest.raises(ValueError, match="Only registered"):
        admission.admit_fold(**{**options, "fold": fold})


@pytest.mark.parametrize("target", ["run.json", "audit.json"])
def test_unknown_witness_bytes_refused(witnessed, target):
    options, *_ = witnessed
    path = options["audit_run"].with_name(target)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="independently known SHA256"):
        admission.admit_fold(**options)


@pytest.mark.parametrize("edit", [
    lambda s: s.update(state="running"),
    lambda s: s.update(schema_version=True),
    lambda s: s.update(training_admitted=True),
    lambda s: s.update(advice=["unsupported"]),
    lambda s: s.update(completed_clip_count=True),
    lambda s: s.update(target_clip_count=15),
    lambda s: s.update(audit_sha256="0" * 64),
])
def test_incomplete_or_promoted_audit_refused(witnessed, monkeypatch, edit):
    options, *_ = witnessed
    reseal(options, monkeypatch, state_edit=edit)
    with pytest.raises(ValueError, match="witness|coverage"):
        admission.admit_fold(**options)


@pytest.mark.parametrize("edit", [
    lambda r: r.update(dataset_sha256="0" * 64),
    lambda r: r.update(protocol_sha256="0" * 64),
    lambda r: r.update(clip_count=True),
    lambda r: r["clips"].pop("0"),
    lambda r: r["clips"]["0"].update(source_id="wrong-source"),
    lambda r: r["clips"]["0"].update(official_outer_split=3),
    lambda r: r["clips"]["0"].update(clip_id=False),
    lambda r: r["clips"]["0"].update(cache_path="/private/tmp/nonexistent-cache.npz"),
    lambda r: r["clips"]["0"].update(prior_catalog_partition="wrong-role"),
    lambda r: r["feature_pipeline"].update(extraction_device="cuda"),
    lambda r: r["folds"]["2"]["partitions"]["train"]["source_ids"].append("fake-source"),
    lambda r: r["folds"]["2"]["class_counts_train_only"].__setitem__(0, 99),
    lambda r: r.update(frame_count=99),
])
def test_incompatible_resealed_report_refused(witnessed, monkeypatch, edit):
    options, *_ = witnessed
    reseal(options, monkeypatch, report_edit=edit)
    with pytest.raises(ValueError):
        admission.admit_fold(**options)


def test_dependency_drift_refused_before_selected_cache_access(witnessed, monkeypatch):
    options, protocol, *_ = witnessed
    registration = Path(protocol["feature_registration"]["path"])
    registration.write_text(registration.read_text() + " ")
    with pytest.raises(ValueError, match="audited input"):
        admission.admit_fold(**options)


def test_missing_dependency_binding_refused(witnessed, monkeypatch):
    options, protocol, *_ = witnessed
    path = protocol["feature_registration"]["path"]
    reseal(options, monkeypatch, state_edit=lambda s: s["inputs_sha256"].pop(path))
    with pytest.raises(ValueError, match="dependency coverage"):
        admission.admit_fold(**options)


def test_runtime_mismatch_refused(witnessed, monkeypatch):
    options, *_ = witnessed

    def fail(_):
        raise ValueError("historical frozen feature runtime differs")

    monkeypatch.setattr(audit, "_runtime_pipeline", fail)
    with pytest.raises(ValueError, match="runtime differs"):
        admission.admit_fold(**options)


def test_selected_cache_tampering_fails_closed(witnessed):
    options, _, entries, dataset = witnessed
    manifest = audit._manifest(dataset, options["manifests_dir"], 2, 20261104, 0.2)
    selected = manifest["partitions"]["train"]["clip_ids"][0]
    path = Path(entries[selected]["cache_path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="selected cache changed"):
        admission.admit_fold(**options)


def test_no_outer_cache_hash_or_array_video_model_access(witnessed, monkeypatch):
    options, _, entries, dataset = witnessed
    manifest = audit._manifest(dataset, options["manifests_dir"], 2, 20261104, 0.2)
    forbidden = {Path(entries[i]["cache_path"]).resolve() for i in manifest["partitions"]["test"]["clip_ids"]}
    original_sha = audit.sha

    def safe_sha(path):
        assert Path(path).resolve() not in forbidden, "outer cache opened"
        assert Path(path).suffix != ".mp4", "video opened"
        return original_sha(path)

    monkeypatch.setattr(audit, "sha", safe_sha)
    monkeypatch.setattr(audit, "_load_cache", lambda *a: pytest.fail("cache array opened"))
    result = admission.admit_fold(**options)
    admission.recheck(result)


@pytest.mark.parametrize("kind", ["cache", "registration", "watch-map"])
def test_post_admission_changes_refused(witnessed, kind):
    options, protocol, *_ = witnessed
    result = admission.admit_fold(**options)
    if kind == "watch-map":
        result.watched.pop(next(iter(result.watched)))
    else:
        path = (next(iter(result.cache_paths.values())) if kind == "cache"
                else Path(protocol["feature_registration"]["path"]))
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        admission.recheck(result)
