"""Authenticate reviewed six-head completion bytes without loading model tensors.

The 018 receipt supersedes the preserved, stale 017 controller. This gate admits
research evaluation inputs only; it supplies no outer score or serving claim.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import fold_admission as prior
from . import fold_cache_audit as audit
from . import full_fold_admission as full
from .experiment import ExperimentConfig
from .fold_protocol import DEFAULT_PROTOCOL, ROOT
from .full_fold_training import _sources, _summary

KNOWN_TRAINING_RECEIPT_SHA256 = "7c37a7d1b04e618dda41673f2a0b5e252b23f1822e6d3fc394590cbc0b06b407"
KNOWN_TRAINING_RUN_SHA256 = {
    (2, "frame"): "9ad9f31f7e5b516b186ee797ba62477eca340d8b2baf208175793ce75cd34041",
    (2, "temporal"): "746fb1519c75861a9fbdba6787904647106499e7cbf6de6c55845755fcbfcec5",
    (3, "frame"): "91718ef607e81568bc3399852fa6ab9100295416492c442375dc0359789c72bb",
    (3, "temporal"): "fabc1a4003114aa5debd748772e07178e0c710ae87e57a23ddeb93e2d75392af",
    (4, "frame"): "c57fb2e619af5d82dacd4be51cc821dda5e86ec17403a483f843cec699f4af85",
    (4, "temporal"): "08f0ce094785a80e9e331b3f2a5172c39295b917e5d5c5cd48e003a29ec2a33b",
}
CLOSED_GATES = {
    "outer_test_evaluated": False, "additional_fold_accuracy_computed": False,
    "latency_benchmark_completed": False, "normal_model_available": False,
    "validated_for_serving": False, "advice": [],
}


@dataclass(frozen=True)
class StudyAdmission:
    dataset: object
    protocol: dict
    folds: dict
    training: dict
    audit_report: dict
    watched: dict
    immutable: dict


def _require(value, expected, message):
    if type(value) is not dict or any(not prior._same(value.get(k), v) for k, v in expected.items()):
        raise ValueError(message)


def _path(value):
    if type(value) is not str:
        raise ValueError("Witness path must be a string")
    path = ROOT / value
    if path.is_symlink():
        raise ValueError("Study witness must not be a symlink")
    return path.resolve()


def _summary_hash(value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def _admit_training(job, base, directory, watched):
    fold, model = job["fold"], job["model"]
    root = directory / f"fold{fold}-{model}-v1"
    if root.is_symlink():
        raise ValueError("Training directory must not be a symlink")
    ledger = root / "run.json"
    if _path(job.get("run_path")) != ledger or job.get("run_sha256") != KNOWN_TRAINING_RUN_SHA256[fold, model]:
        raise ValueError("Training run differs from independently known completion")
    state = prior._known_json(ledger, KNOWN_TRAINING_RUN_SHA256[fold, model])
    _require(state, {"schema_version": 1, "state": "completed", "error": None,
                     "full_training_completed": True, "pending_advance": None,
                     "evidence_type": "fixed-additional-fold-inner-selection-not-outer-benchmark",
                     **{k: CLOSED_GATES[k] for k in ("outer_test_evaluated", "normal_model_available",
                                                    "validated_for_serving", "advice")}},
             "Full training must be completed with closed outer/serving gates")
    config_path = ROOT / f"experiments/{model}-v1.json"
    config = ExperimentConfig.from_dict(audit._read(config_path))
    if config.epochs != 20 or config.model != model:
        raise ValueError("Training must use the original full twenty epoch configuration")
    counts = base.class_counts_train_only
    n = len(base.manifest["partitions"]["train"]["clip_ids"])
    inner = len(base.manifest["partitions"]["validation"]["clip_ids"])
    total = math.ceil(n / config.batch_size) * config.epochs
    sources = _sources(config_path)
    immutable = state.get("immutable")
    _require(immutable, {
        "admission": base.immutable, "config": config.to_dict(), "fold": fold, "model": model,
        "mode": "full", "total_optimizer_batches": total, "checkpoint_batches": 500,
        "initialization": "fresh-seeded-head-never-smoke-or-prior-fold-warm-start",
        "runtime": full._runtime_identity(), "implementation_sha256": sources,
        "run_dir": str(root), "lock_path": str((ROOT / "artifacts/training/active-training.lock").resolve()),
    }, "Completed training data, source, configuration, grouping or runtime differs")
    initial = base.immutable["smoke_witnesses"][model]["initial_model_tensor_sha256"]
    inverse = np.reciprocal(np.asarray(counts, dtype=np.float64))
    weights = (inverse / inverse.mean()).astype(np.float32).tolist()
    _require(state, {"class_counts_train_only": counts, "class_weights_train_only": weights,
                     "initial_model_tensor_sha256": initial}, "Train-only weights or fresh initialization differs")
    if not full._hex(state.get("final_model_tensor_sha256")) or state["final_model_tensor_sha256"] == initial:
        raise ValueError("Completed head must retain changed final tensor identity")
    summary = state.get("summary")
    if _summary(summary, config, n, counts) != total or summary["completed_epochs"] != 20:
        raise ValueError("Completed full training has incomplete epoch history")
    for record in summary["history"]:
        _require(record["validation"], {"clip_count": inner}, "Inner selection clip coverage differs")
    _require(job, {
        "state": "completed", "full_training_completed": True, "job_id": state.get("job_id"),
        "config": config.to_dict(), "config_file_sha256": audit.sha(config_path),
        "manifest_sha256": base.immutable["manifest_sha256"], "class_counts_train_only": counts,
        "initial_model_tensor_sha256": initial, "final_model_tensor_sha256": state["final_model_tensor_sha256"],
        "completed_epochs": 20, "optimizer_steps": total,
        "best_epoch_inner_only": summary["best_epoch"],
        "best_pce_inner_selection_only": summary["best_validation_pce"],
        "checkpoint_count": 20, "evaluation_state_key": "best_model",
        "attempt_count": len(state.get("attempts", [])), "attempts_recorded": state.get("attempts"),
        "git_base_commit": state.get("git_base_commit"), "completed_resume_byte_identical": True,
    }, "Completion receipt differs from full training selection or configuration")
    rows = state.get("checkpoints")
    if type(rows) is not list or len(rows) != 20:
        raise ValueError("Twenty immutable epoch checkpoints are required")
    paths = set()
    for epoch, row in enumerate(rows, 1):
        steps = epoch * math.ceil(n / config.batch_size)
        path = root / f"step-{steps:07d}.pt"
        _require(row, {"optimizer_steps": steps, "path": str(path)}, "Epoch checkpoint progress differs")
        if not full._hex(row.get("sha256")) or not full._hex(row.get("summary_sha256")):
            raise ValueError("Invalid checkpoint byte or summary fingerprint")
        full._watch(watched, path, row["sha256"])
        paths.add(path)
    if set(root.glob("step-*.pt")) != paths or rows[-1]["summary_sha256"] != _summary_hash(summary):
        raise ValueError("Checkpoint coverage or final inner-selection summary differs")
    selected = rows[summary["best_epoch"] - 1]
    if (_path(job.get("final_checkpoint_path")) != Path(rows[-1]["path"])
            or job.get("final_checkpoint_sha256") != rows[-1]["sha256"]
            or _path(job.get("selected_epoch_checkpoint_path")) != Path(selected["path"])
            or job.get("selected_epoch_checkpoint_sha256") != selected["sha256"]):
        raise ValueError("Final or selected earliest-best checkpoint differs")
    noop_path = _path(job.get("completed_noop_witness_path"))
    noop = prior._known_json(noop_path, job.get("completed_noop_witness_sha256"))
    expected = {str(ledger): job["run_sha256"], **{r["path"]: r["sha256"] for r in rows}}
    _require(noop, {"job_id": state["job_id"], "completed_resume_byte_identical": True,
                    "attempt_count_unchanged": len(state["attempts"]), "optimizer_updates_unchanged": total,
                    "before_sha256": expected, "after_sha256": expected}, "Completed resume witness differs")
    for path, digest in {str(ledger): job["run_sha256"], str(noop_path): job["completed_noop_witness_sha256"], **sources}.items():
        full._watch(watched, path, digest)
    return {"run_path": str(ledger), "run_sha256": job["run_sha256"],
            "checkpoint_path": rows[-1]["path"], "checkpoint_sha256": rows[-1]["sha256"],
            "config": config, "ledger": state}


def _metadata(study):
    return {"protocol": study.protocol, "audit_report": study.audit_report,
            "dataset_sha256": study.dataset.sha256,
            "folds": {str(f): {"manifest": a.manifest, "immutable": a.immutable,
                               "cache_paths": {str(k): str(v) for k, v in a.cache_paths.items()},
                               "class_counts_train_only": a.class_counts_train_only,
                               "feature_pipeline": a.feature_pipeline} for f, a in study.folds.items()},
            "training": {str(f): {m: {**r, "config": r["config"].to_dict()} for m, r in models.items()}
                         for f, models in study.training.items()}}


def admit_completed_training(*, annotations, manifests_dir, audit_run, smokes_dir,
                             training_runs_dir, training_receipt, expected_receipt_sha256,
                             protocol_path=DEFAULT_PROTOCOL):
    """Bind 018 completion, 015 audit, BOTH 016 smokes and all six full heads.

    Expected SHA is an assertion against the fixed production witness, never a
    caller-controlled replacement trust anchor. No model or checkpoint is loaded.
    """
    if expected_receipt_sha256 != KNOWN_TRAINING_RECEIPT_SHA256:
        raise ValueError("Expected receipt SHA256 must match the independently known 018 completion witness")
    receipt_path = Path(training_receipt)
    receipt = prior._known_json(receipt_path, KNOWN_TRAINING_RECEIPT_SHA256)
    _require(receipt, {"schema_version": 1,
                       "artifact_type": "independently-reviewed-additional-fold-training-completion-witness",
                       "execution": "018", "training_execution": "017", "study_id": "additional-official-folds-v1",
                       "state": "completed", "training_performed": True, "completed_head_epochs": 120,
                       "known_audit_run_sha256": prior.KNOWN_AUDIT_RUN_SHA256,
                       "known_audit_report_sha256": prior.KNOWN_AUDIT_REPORT_SHA256,
                       "known_smoke_run_sha256": full.KNOWN_SMOKE_RUN_SHA256, **CLOSED_GATES},
             "Only the reviewed 018 completion receipt admits six heads; the stale controller is not completion")
    jobs = receipt.get("jobs")
    if (type(jobs) is not list or len(jobs) != 6
            or any(type(j) is not dict or type(j.get("fold")) is not int
                   or type(j.get("model")) is not str for j in jobs)
            or {(j["fold"], j["model"]) for j in jobs} != set(KNOWN_TRAINING_RUN_SHA256)):
        raise ValueError("Completion requires exactly all six registered fold/head jobs")
    directory = Path(training_runs_dir)
    if directory.is_symlink():
        raise ValueError("Training root must not be a symlink")
    directory = directory.resolve()
    watched, folds = {}, {}
    full._watch(watched, receipt_path, KNOWN_TRAINING_RECEIPT_SHA256)
    for fold in (2, 3, 4):
        folds[fold] = full.admit_full_fold(annotations=annotations, manifests_dir=manifests_dir,
                                          audit_run=audit_run, smokes_dir=smokes_dir, fold=fold,
                                          protocol_path=protocol_path)
        for path, digest in folds[fold].watched.items():
            full._watch(watched, path, digest)
    dataset = folds[2].dataset
    if (receipt.get("dataset_sha256") != dataset.sha256
            or receipt.get("protocol_sha256") != audit.sha(protocol_path)
            or _path(receipt.get("protocol_path")) != Path(protocol_path).resolve()):
        raise ValueError("Receipt dataset or protocol differs from audited study")
    sources = receipt.get("reviewed_training_source_sha256")
    if type(sources) is not dict or not sources:
        raise ValueError("Reviewed training source bindings are required")
    for path, digest in sources.items():
        full._watch(watched, _path(path), digest)
    training = {f: {} for f in folds}
    for job in jobs:
        training[job["fold"]][job["model"]] = _admit_training(job, folds[job["fold"]], directory, watched)
    total = sum(r["ledger"]["summary"]["optimizer_steps"] for models in training.values() for r in models.values())
    if not prior._same(receipt.get("durable_optimizer_updates"), total):
        raise ValueError("Completion receipt update total differs from all six jobs")
    protocol = audit._read(protocol_path)
    report = prior._known_json(Path(audit_run).resolve().with_name("audit.json"), prior.KNOWN_AUDIT_REPORT_SHA256)
    full._watch(watched, __file__, audit.sha(__file__))
    immutable = {"schema_version": 1, "completed_training_admitted": True,
                 "training_receipt_sha256": KNOWN_TRAINING_RECEIPT_SHA256,
                 "evaluation_state_key": "best_model", **CLOSED_GATES,
                 "watched_sha256": audit.json_hash(watched)}
    study = StudyAdmission(dataset, protocol, folds, training, report, watched, immutable)
    immutable["metadata_sha256"] = audit.json_hash(_metadata(study))
    recheck_study(study)
    return study


def recheck_study(study):
    """Rehash all bound bytes before model construction and result publication."""
    _require(study.immutable, {"schema_version": 1, "completed_training_admitted": True,
                             "training_receipt_sha256": KNOWN_TRAINING_RECEIPT_SHA256,
                             "evaluation_state_key": "best_model", **CLOSED_GATES}, "Study evidence gates changed")
    if (set(study.folds) != {2, 3, 4} or set(study.training) != {2, 3, 4}
            or any(set(models) != {"frame", "temporal"} for models in study.training.values())
            or audit.json_hash(study.watched) != study.immutable.get("watched_sha256")
            or audit.json_hash(_metadata(study)) != study.immutable.get("metadata_sha256")):
        raise ValueError("Study metadata or dependency bindings changed")
    # The combined map contains every fold binding. Validate each original map
    # without rehashing the same shared caches three additional times.
    for base in study.folds.values():
        if (audit.json_hash(base.watched) != base.immutable.get("watched_sha256")
                or any(study.watched.get(path) != digest for path, digest in base.watched.items())):
            raise ValueError("Study fold dependency bindings changed")
    runtime = full._runtime_identity()
    if any(not prior._same(row["ledger"]["immutable"]["runtime"], runtime)
           for heads in study.training.values() for row in heads.values()):
        raise ValueError("Study current runtime changed")
    for path, digest in study.watched.items():
        if Path(path).is_symlink() or audit.sha(path) != digest:
            raise ValueError("An admitted study input or checkpoint changed: " + path)
