"""Admit fixed additional-fold training only after both known real smoke witnesses.

This authenticates local bytes and metadata; it never constructs a model, loads
checkpoint tensors, fits, forwards, evaluates outer data or proves a remote push.
The caller owns the shared expensive-work lock and durable launch identity.
"""
from dataclasses import replace
import importlib.metadata
import os
from pathlib import Path
import platform
import sys

from . import fold_admission as prior
from . import fold_cache_audit as audit
from .fold_protocol import DEFAULT_PROTOCOL, ROOT

KNOWN_SMOKE_RUN_SHA256 = {
    "frame": "eb0728446686ad3db71960da17a131722f103cff0cb575589152795c70294657",
    "temporal": "9b0bc9acc8c65a5a1898e9bebc415bdbed8b8c9c49e9267ec78e6d8b2ed41736",
}
SMOKE_CLOSED_GATES = {
    "full_training_completed": False, "outer_test_evaluated": False,
    "inner_validation_evaluated": False, "accuracy_metrics_computed": False,
    "normal_model_available": False, "validated_for_serving": False, "advice": [],
}


def _runtime_identity():
    # Thread/deterministic values describe the witnessed CPU policy. The caller
    # enforces current Torch settings before constructing its fresh full head.
    return {
        "python": platform.python_version(), "python_executable": os.path.realpath(sys.executable),
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
        "numpy": importlib.metadata.version("numpy"), "pyav": importlib.metadata.version("av"),
        "machine": platform.machine(), "platform": platform.platform(),
        "device": "cpu", "torch_num_threads": 1, "deterministic_algorithms": True,
    }


def _hex(value, length=64):
    return (type(value) is str and len(value) == length
            and all(c in "0123456789abcdef" for c in value))


def _implementation(model):
    paths = [ROOT / "backend/swingsync/research" / name for name in (
        "fold_admission.py", "fold_protocol.py", "fold_training.py")]
    paths += [ROOT / "scripts/train_additional_fold.py", ROOT / "uv.lock",
              ROOT / f"experiments/{model}-v1.json"]
    return {str(path.resolve()): audit.sha(path) for path in paths}


def _watch(watched, path, digest):
    if Path(path).is_symlink():
        raise ValueError("Smoke witness, source or checkpoint must not be a symlink")
    prior._watch(watched, path, digest)


def _admit_smoke(model, root, base, manifests_dir, watched, counts):
    ledger = root / "run.json"
    state = prior._known_json(ledger, KNOWN_SMOKE_RUN_SHA256[model])
    if (type(state) is not dict or type(state.get("schema_version")) is not int
            or state["schema_version"] != 1 or state.get("state") != "completed"
            or state.get("smoke_completed") is not True or state.get("error") is not None
            or state.get("evidence_type") != "five-update-training-only-smoke-not-completed-benchmark-training"
            or not _hex(state.get("job_id"), 32)
            or any(not prior._same(state.get(k), v) for k, v in SMOKE_CLOSED_GATES.items())):
        raise ValueError("Both smoke witnesses must be completed and preserve closed evidence gates")
    immutable = state.get("immutable")
    config = audit._read(ROOT / f"experiments/{model}-v1.json")
    sources = _implementation(model)
    if (type(immutable) is not dict
            or any(not prior._same(immutable.get(k), v) for k, v in {
                "mode": "smoke", "fold": 2, "model": model,
                "config": config, "total_optimizer_batches": 5,
                "initialization": "fresh-seeded-head-no-warm-start",
                "runtime": _runtime_identity(), "implementation_sha256": sources,
                "run_dir": str(root),
                "lock_path": str((ROOT / "artifacts/training/active-training.lock").resolve()),
            }.items())):
        raise ValueError("Smoke model, configuration, source, runtime or initialization differs")
    old = immutable.get("admission")
    # Old fold2 cache membership is not the new fold3/4 membership. Authenticate
    # its known metadata without opening those old fold2 arrays or outer caches.
    common = {k: base.immutable[k] for k in (
        "schema_version", "dataset_sha256", "protocol_sha256", "audit_run_sha256",
        "audit_report_sha256", "feature_pipeline", "verification_runtime",
        "outer_test_evaluated", "validated_for_serving")}
    common.update(fold=2, manifest_sha256=audit.sha(Path(manifests_dir) / "fold-2.json"),
                  class_counts_train_only=counts)
    if (type(old) is not dict or any(not prior._same(old.get(k), v) for k, v in common.items())
            or not _hex(old.get("watched_sha256"))):
        raise ValueError("Smoke data, audit, fold2 source grouping or pipeline differs")
    summary = state.get("summary")
    expected_summary = {
        "evidence_type": "inner-validation-head-training-not-outer-benchmark",
        "state": "partial", "optimizer_steps": 5, "completed_epochs": 0,
        "clips_processed_in_current_epoch": 20, "history": [],
        "best_epoch": None, "best_validation_pce": None,
        "class_counts_train_only": counts, "outer_test_evaluated": False,
        "validated_for_serving": False,
    }
    attempts = state.get("attempts")
    if (not prior._same(summary, expected_summary)
            or not prior._same(state.get("class_counts_train_only"), counts)
            or type(attempts) is not list or len(attempts) != 2
            or [a.get("state") if type(a) is dict else None for a in attempts] != ["partial", "completed"]
            or not _hex(state.get("initial_model_tensor_sha256"))
            or not _hex(state.get("final_model_tensor_sha256"))
            or state["initial_model_tensor_sha256"] == state["final_model_tensor_sha256"]):
        raise ValueError("Smoke progress, selection, attempt history or fresh tensor evidence differs")
    rows = state.get("checkpoints")
    if type(rows) is not list or len(rows) != 5:
        raise ValueError("Each smoke must retain exactly five durable checkpoints")
    expected_paths = set()
    for step, row in enumerate(rows, 1):
        path = root / f"step-{step:07d}.pt"
        if (type(row) is not dict or type(row.get("optimizer_steps")) is not int
                or row["optimizer_steps"] != step or row.get("path") != str(path)
                or not _hex(row.get("sha256"))):
            raise ValueError("Smoke checkpoint path, count or cumulative update differs")
        _watch(watched, path, row["sha256"])
        expected_paths.add(path)
    if set(root.glob("step-*.pt")) != expected_paths:
        raise ValueError("Unreferenced smoke checkpoints require inspection")
    _watch(watched, ledger, KNOWN_SMOKE_RUN_SHA256[model])
    for path, digest in sources.items():
        _watch(watched, path, digest)
    return {"job_id": state["job_id"], "run_path": str(ledger),
            "run_sha256": KNOWN_SMOKE_RUN_SHA256[model],
            "initial_model_tensor_sha256": state["initial_model_tensor_sha256"],
            "checkpoint_sha256": [r["sha256"] for r in rows]}


def admit_full_fold(*, annotations, manifests_dir, audit_run, smokes_dir, fold,
                    protocol_path=DEFAULT_PROTOCOL):
    """Authenticate both smoke witnesses and return only this fold's train/inner.

    No weights are loaded. Passing this metadata gate is permission for the
    separately reviewed full launcher to initialize fresh, never a warm start.
    Remote publication/review remain coordinator workflow requirements.
    """
    base = prior.admit_fold(annotations=annotations, manifests_dir=manifests_dir,
                            audit_run=audit_run, fold=fold, protocol_path=protocol_path)
    directory = Path(smokes_dir)
    if directory.is_symlink():
        raise ValueError("Smoke witness directory must not be a symlink")
    directory = directory.resolve()
    watched = dict(base.watched)
    report = prior._known_json(Path(audit_run).resolve().with_name("audit.json"),
                               prior.KNOWN_AUDIT_REPORT_SHA256)
    counts = report["folds"]["2"]["class_counts_train_only"]
    smokes = {}
    for model in ("frame", "temporal"):
        root = directory / f"{model}-smoke-v1"
        if root.is_symlink():
            raise ValueError("Smoke witness directory must not be a symlink")
        smokes[model] = _admit_smoke(model, root, base, manifests_dir, watched, counts)
    _watch(watched, __file__, audit.sha(__file__))
    immutable = {**base.immutable, "full_training_input_admitted": True,
                 "full_initialization": "fresh-per-fold-and-model-never-smoke-warm-start",
                 "smoke_witnesses": smokes, "watched_sha256": audit.json_hash(watched)}
    result = replace(base, immutable=immutable, watched=watched)
    recheck_full(result)
    return result


def recheck_full(admission):
    """Recheck selected inputs, both smoke witnesses/checkpoints and source bytes."""
    if (admission.immutable.get("full_training_input_admitted") is not True
            or admission.immutable.get("full_initialization") != "fresh-per-fold-and-model-never-smoke-warm-start"
            or set(admission.immutable.get("smoke_witnesses", {})) != {"frame", "temporal"}):
        raise ValueError("Full training admission metadata changed")
    prior.recheck(admission)
