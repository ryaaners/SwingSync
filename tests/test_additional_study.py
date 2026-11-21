"""Explicit synthetic byte witnesses test metadata gates, never trained models.

These fixtures replace production known-hash constants only in tests. The prior
full admission is represented by declared byte bindings: no cache scan, tensor
loading, training, model construction or inference is needed by this unit suite.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from backend.swingsync.research import additional_study as study  # noqa: E402
from backend.swingsync.research import fold_admission as prior  # noqa: E402
from backend.swingsync.research import fold_cache_audit as audit  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@pytest.fixture
def ready(tmp_path, monkeypatch):
    dataset = SimpleNamespace(sha256="d" * 64)
    protocol_path = tmp_path / "SYNTHETIC-protocol.json"
    write(protocol_path, {"study_id": "additional-official-folds-v1", "synthetic": True})
    audit_run = tmp_path / "run.json"
    write(audit_run, {"synthetic": True})
    report_path = tmp_path / "audit.json"
    write(report_path, {"synthetic": True, "clips": {str(i): {"cache_path": "declared-only"} for i in range(10)}})
    monkeypatch.setattr(prior, "KNOWN_AUDIT_REPORT_SHA256", audit.sha(report_path))
    runs = tmp_path / "SYNTHETIC-training"
    runs.mkdir()
    folds, jobs, hashes = {}, [], {}
    for fold in (2, 3, 4):
        manifest = {"partitions": {"train": {"clip_ids": list(range(6)), "source_ids": list(range(6))},
                                   "validation": {"clip_ids": [6, 7], "source_ids": [6, 7]},
                                   "test": {"clip_ids": [8, 9], "source_ids": [8, 9]}}}
        watched, cache_paths = {}, {}
        for i in range(8):
            path = tmp_path / f"SYNTHETIC-{fold}-{i}.npz"
            path.write_bytes(f"SYNTHETIC non-array bytes {fold} {i}".encode())
            watched[str(path)] = audit.sha(path)
            cache_paths[i] = path
        initial = {m: {"initial_model_tensor_sha256": ("a" if m == "frame" else "b") * 64}
                   for m in ("frame", "temporal")}
        counts = [6] * 8 + [100]
        immutable = {"full_training_input_admitted": True,
                     "full_initialization": "fresh-per-fold-and-model-never-smoke-warm-start",
                     "smoke_witnesses": initial, "manifest_sha256": audit.json_hash(manifest),
                     "watched_sha256": audit.json_hash(watched)}
        base = prior.Admission(dataset, manifest, cache_paths, {"synthetic": True}, immutable, watched, counts)
        folds[fold] = base
        for model in ("frame", "temporal"):
            root = runs / f"fold{fold}-{model}-v1"
            root.mkdir()
            config_path = study.ROOT / f"experiments/{model}-v1.json"
            config = study.ExperimentConfig.from_dict(audit._read(config_path))
            history = [{"epoch": n, "training_loss": 1., "validation": {
                "pce": .5 if n in (2, 4) else .25, "clip_count": 2,
                "partition": "inner-validation-only", "temporal_context": "full-clip",
                "outer_test_evaluated": False}} for n in range(1, 21)]
            summary = {"optimizer_steps": 40, "completed_epochs": 20,
                       "clips_processed_in_current_epoch": 0, "state": "completed",
                       "class_counts_train_only": counts, "outer_test_evaluated": False,
                       "validated_for_serving": False, "history": history,
                       "best_epoch": 2, "best_validation_pce": .5}
            rows = []
            for epoch in range(1, 21):
                path = root / f"step-{epoch * 2:07d}.pt"
                path.write_bytes(f"SYNTHETIC non-checkpoint bytes {fold} {model} {epoch}".encode())
                rows.append({"optimizer_steps": epoch * 2, "path": str(path), "sha256": audit.sha(path),
                             "summary_sha256": study._summary_hash(summary) if epoch == 20 else "c" * 64})
            inverse = study.np.reciprocal(study.np.array(counts, dtype=study.np.float64))
            weights = (inverse / inverse.mean()).astype(study.np.float32).tolist()
            state = {"schema_version": 1, "state": "completed", "error": None,
                     "full_training_completed": True, "pending_advance": None,
                     "evidence_type": "fixed-additional-fold-inner-selection-not-outer-benchmark",
                     "outer_test_evaluated": False, "normal_model_available": False,
                     "validated_for_serving": False, "advice": [],
                     "job_id": f"{fold}{'1' if model == 'frame' else '2'}" * 16,
                     "git_base_commit": "e" * 40, "attempts": [{"state": "completed"}],
                     "class_counts_train_only": counts, "class_weights_train_only": weights,
                     "initial_model_tensor_sha256": initial[model]["initial_model_tensor_sha256"],
                     "final_model_tensor_sha256": "f" * 64, "summary": summary, "checkpoints": rows,
                     "immutable": {"admission": deepcopy(immutable), "config": config.to_dict(),
                                   "fold": fold, "model": model, "mode": "full",
                                   "total_optimizer_batches": 40, "checkpoint_batches": 500,
                                   "initialization": "fresh-seeded-head-never-smoke-or-prior-fold-warm-start",
                                   "runtime": study.full._runtime_identity(),
                                   "implementation_sha256": study._sources(config_path), "run_dir": str(root),
                                   "lock_path": str((study.ROOT / "artifacts/training/active-training.lock").resolve())}}
            write(root / "run.json", state)
            run_sha = audit.sha(root / "run.json")
            hashes[fold, model] = run_sha
            expected = {str(root / "run.json"): run_sha, **{r["path"]: r["sha256"] for r in rows}}
            noop_path = tmp_path / f"SYNTHETIC-noop-{fold}-{model}.json"
            write(noop_path, {"job_id": state["job_id"], "completed_resume_byte_identical": True,
                              "attempt_count_unchanged": 1, "optimizer_updates_unchanged": 40,
                              "before_sha256": expected, "after_sha256": expected})
            jobs.append({"fold": fold, "model": model, "run_path": str(root / "run.json"), "run_sha256": run_sha,
                         "state": "completed", "full_training_completed": True, "job_id": state["job_id"],
                         "config": config.to_dict(), "config_file_sha256": audit.sha(config_path),
                         "manifest_sha256": immutable["manifest_sha256"], "class_counts_train_only": counts,
                         "initial_model_tensor_sha256": state["initial_model_tensor_sha256"],
                         "final_model_tensor_sha256": state["final_model_tensor_sha256"],
                         "completed_epochs": 20, "optimizer_steps": 40, "best_epoch_inner_only": 2,
                         "best_pce_inner_selection_only": .5, "checkpoint_count": 20, "evaluation_state_key": "best_model",
                         "attempt_count": 1, "attempts_recorded": state["attempts"], "git_base_commit": state["git_base_commit"],
                         "completed_resume_byte_identical": True, "final_checkpoint_path": rows[-1]["path"],
                         "final_checkpoint_sha256": rows[-1]["sha256"], "selected_epoch_checkpoint_path": rows[1]["path"],
                         "selected_epoch_checkpoint_sha256": rows[1]["sha256"],
                         "completed_noop_witness_path": str(noop_path), "completed_noop_witness_sha256": audit.sha(noop_path)})
    receipt = {"schema_version": 1,
               "artifact_type": "independently-reviewed-additional-fold-training-completion-witness",
               "execution": "018", "training_execution": "017", "study_id": "additional-official-folds-v1",
               "state": "completed", "training_performed": True, "completed_head_epochs": 120,
               "durable_optimizer_updates": 240, "known_audit_run_sha256": prior.KNOWN_AUDIT_RUN_SHA256,
               "known_audit_report_sha256": prior.KNOWN_AUDIT_REPORT_SHA256,
               "known_smoke_run_sha256": study.full.KNOWN_SMOKE_RUN_SHA256,
               "dataset_sha256": dataset.sha256, "protocol_path": str(protocol_path),
               "protocol_sha256": audit.sha(protocol_path),
               "reviewed_training_source_sha256": {str(Path(study.__file__).resolve()): audit.sha(study.__file__)},
               "jobs": jobs, **study.CLOSED_GATES}
    receipt_path = tmp_path / "SYNTHETIC-completion-receipt.json"
    write(receipt_path, receipt)
    monkeypatch.setattr(study, "KNOWN_TRAINING_RECEIPT_SHA256", audit.sha(receipt_path))
    monkeypatch.setattr(study, "KNOWN_TRAINING_RUN_SHA256", hashes)
    calls = []
    def admission(**kwargs):
        calls.append(kwargs["fold"])
        return folds[kwargs["fold"]]
    monkeypatch.setattr(study.full, "admit_full_fold", admission)
    options = {"annotations": tmp_path / "declared-only.mat", "manifests_dir": tmp_path,
               "audit_run": audit_run, "smokes_dir": tmp_path, "training_runs_dir": runs,
               "training_receipt": receipt_path, "expected_receipt_sha256": audit.sha(receipt_path),
               "protocol_path": protocol_path}
    return options, receipt, folds, calls


def seal(options, receipt, monkeypatch):
    write(options["training_receipt"], receipt)
    digest = audit.sha(options["training_receipt"])
    monkeypatch.setattr(study, "KNOWN_TRAINING_RECEIPT_SHA256", digest)
    options["expected_receipt_sha256"] = digest


def reseal_run(options, receipt, monkeypatch, edit):
    job = receipt["jobs"][0]
    path = Path(job["run_path"])
    value = audit._read(path)
    edit(value)
    write(path, value)
    job["run_sha256"] = audit.sha(path)
    study.KNOWN_TRAINING_RUN_SHA256[job["fold"], job["model"]] = job["run_sha256"]
    noop_path = Path(job["completed_noop_witness_path"])
    noop = audit._read(noop_path)
    for key in ("before_sha256", "after_sha256"):
        noop[key][str(path)] = job["run_sha256"]
    write(noop_path, noop)
    job["completed_noop_witness_sha256"] = audit.sha(noop_path)
    seal(options, receipt, monkeypatch)


def test_production_receipt_and_six_run_hashes_are_fixed():
    assert study.KNOWN_TRAINING_RECEIPT_SHA256 == "7c37a7d1b04e618dda41673f2a0b5e252b23f1822e6d3fc394590cbc0b06b407"
    assert len(study.KNOWN_TRAINING_RUN_SHA256) == 6
    assert study.KNOWN_TRAINING_RUN_SHA256[4, "temporal"] == "08f0ce094785a80e9e331b3f2a5172c39295b917e5d5c5cd48e003a29ec2a33b"


def test_complete_synthetic_bytes_admit_all_six_without_model_or_tensor_load(ready, monkeypatch):
    options, _, _, calls = ready
    import torch
    from backend.swingsync.research import experiment, models
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Checkpoint tensor load"))
    monkeypatch.setattr(experiment, "HeadExperiment", lambda *a, **k: pytest.fail("Head constructed"))
    monkeypatch.setattr(models, "FrameClassifier", lambda *a, **k: pytest.fail("Model constructed"))
    monkeypatch.setattr(audit, "_load_cache", lambda *a, **k: pytest.fail("Cache arrays loaded"))
    result = study.admit_completed_training(**options)
    assert calls == [2, 3, 4]
    assert set(result.training) == {2, 3, 4}
    assert len([p for p in result.watched if p.endswith(".pt")]) == 120
    assert len(result.audit_report["clips"]) == 10
    for heads in result.training.values():
        for row in heads.values():
            assert isinstance(row["config"], study.ExperimentConfig)
            assert row["checkpoint_path"].endswith("step-0000040.pt")
            assert row["ledger"]["summary"]["best_epoch"] == 2
    assert result.immutable["evaluation_state_key"] == "best_model"
    assert result.immutable["normal_model_available"] is False
    study.recheck_study(result)


def test_unknown_expected_receipt_hash_refused_before_any_admission(ready):
    options, _, _, calls = ready
    options["expected_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="independently known"):
        study.admit_completed_training(**options)
    assert calls == []


@pytest.mark.parametrize("edit", [
    lambda r: r.update(state="running"), lambda r: r.update(schema_version=True),
    lambda r: r.update(execution="017"), lambda r: r.update(outer_test_evaluated=True),
    lambda r: r.update(normal_model_available=True), lambda r: r.update(advice=["unsupported"]),
    lambda r: r["jobs"].pop(), lambda r: r["jobs"].__setitem__(0, r["jobs"][1]),
    lambda r: r["jobs"][0].update(fold=True), lambda r: r["jobs"][0].update(run_sha256="0" * 64),
    lambda r: r["jobs"][0].update(evaluation_state_key="model"),
    lambda r: r["jobs"][0].update(best_epoch_inner_only=4),
    lambda r: r["jobs"][0].update(config_file_sha256="0" * 64),
    lambda r: r["jobs"][0].update(selected_epoch_checkpoint_sha256="0" * 64),
    lambda r: r.update(durable_optimizer_updates=239),
])
def test_resealed_invalid_receipt_refused(ready, monkeypatch, edit):
    options, receipt, _, _ = ready
    edit(receipt)
    seal(options, receipt, monkeypatch)
    with pytest.raises(ValueError):
        study.admit_completed_training(**options)


@pytest.mark.parametrize("edit", [
    lambda s: s.update(state="partial"), lambda s: s.update(full_training_completed=False),
    lambda s: s.update(outer_test_evaluated=True), lambda s: s.update(advice=["unsupported"]),
    lambda s: s["immutable"].update(initialization="warm-start"),
    lambda s: s["immutable"]["runtime"].update(python="different"),
    lambda s: s["immutable"]["implementation_sha256"].update(unrelated="0" * 64),
    lambda s: s["immutable"]["admission"].update(manifest_sha256="0" * 64),
    lambda s: s["immutable"]["config"].update(seed=0),
    lambda s: s["immutable"].update(total_optimizer_batches=39),
    lambda s: s["class_weights_train_only"].__setitem__(0, 1.),
    lambda s: s.update(initial_model_tensor_sha256="0" * 64),
    lambda s: s["summary"].update(completed_epochs=19),
    lambda s: s["summary"].update(best_epoch=4),
    lambda s: s["summary"]["history"][0]["validation"].update(clip_count=1),
    lambda s: s["summary"]["history"][0]["validation"].update(partition="outer-test"),
    lambda s: s["checkpoints"].pop(),
    lambda s: s["checkpoints"][0].update(optimizer_steps=True),
    lambda s: s["checkpoints"][-1].update(summary_sha256="0" * 64),
])
def test_resealed_invalid_training_metadata_refused(ready, monkeypatch, edit):
    options, receipt, _, _ = ready
    reseal_run(options, receipt, monkeypatch, edit)
    with pytest.raises(ValueError):
        study.admit_completed_training(**options)


@pytest.mark.parametrize("job_index", range(6))
@pytest.mark.parametrize("kind", ["missing-run", "wrong-run", "checkpoint", "missing-checkpoint", "extra-checkpoint", "noop"])
def test_each_head_requires_complete_raw_bytes(ready, job_index, kind):
    options, receipt, _, _ = ready
    job = receipt["jobs"][job_index]
    path = Path(job["run_path"])
    if kind in ("checkpoint", "missing-checkpoint"):
        path = Path(audit._read(path)["checkpoints"][0]["path"])
    elif kind == "noop":
        path = Path(job["completed_noop_witness_path"])
    elif kind == "extra-checkpoint":
        path = path.with_name("step-9999999.pt")
    if kind.startswith("missing"):
        path.unlink()
    else:
        path.write_bytes(b"SYNTHETIC changed or unreferenced software bytes")
    with pytest.raises((ValueError, OSError)):
        study.admit_completed_training(**options)


def test_current_runtime_drift_refused(ready, monkeypatch):
    options, _, _, _ = ready
    original = study.full._runtime_identity
    monkeypatch.setattr(study.full, "_runtime_identity", lambda: {**original(), "python_executable": "/changed"})
    with pytest.raises(ValueError, match="runtime"):
        study.admit_completed_training(**options)


@pytest.mark.parametrize("kind", ["receipt", "run", "checkpoint", "noop", "watched", "ledger-metadata", "protocol", "audit", "groups"])
def test_late_byte_and_metadata_drift_refused(ready, kind):
    options, receipt, _, _ = ready
    admitted = study.admit_completed_training(**options)
    if kind == "watched":
        admitted.watched.pop(next(iter(admitted.watched)))
    elif kind == "ledger-metadata":
        admitted.training[2]["frame"]["ledger"]["summary"]["best_epoch"] = 4
    elif kind == "protocol":
        admitted.protocol["synthetic"] = False
    elif kind == "audit":
        admitted.audit_report["clips"].pop("0")
    elif kind == "groups":
        admitted.folds[2].manifest["partitions"]["train"]["source_ids"].pop()
    else:
        job = receipt["jobs"][0]
        path = {"receipt": options["training_receipt"], "run": Path(job["run_path"]),
                "checkpoint": Path(job["final_checkpoint_path"]), "noop": Path(job["completed_noop_witness_path"])}[kind]
        path.write_bytes(path.read_bytes() + b" changed after admission")
    with pytest.raises(ValueError, match="changed"):
        study.recheck_study(admitted)


def test_late_runtime_drift_refused(ready, monkeypatch):
    options, _, _, _ = ready
    admitted = study.admit_completed_training(**options)
    original = study.full._runtime_identity
    monkeypatch.setattr(study.full, "_runtime_identity", lambda: {**original(), "torch": "changed"})
    with pytest.raises(ValueError, match="runtime"):
        study.recheck_study(admitted)
