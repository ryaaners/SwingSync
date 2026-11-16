"""Hypothetical synthetic byte witnesses prove admission wiring, never real ML.

The fixture explicitly substitutes its own known witness hashes. Its checkpoint
files are labeled arbitrary software bytes, not models or pretrained extraction.
Production continues to pin the independently reviewed real 016 ledger hashes.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from test_fold_admission import toy as _toy, witnessed as _witnessed  # noqa: E402
from backend.swingsync.research import full_fold_admission as full  # noqa: E402
from backend.swingsync.research import fold_admission as prior  # noqa: E402
from backend.swingsync.research import fold_cache_audit as audit  # noqa: E402

toy = _toy
witnessed = _witnessed


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@pytest.fixture
def ready(witnessed, monkeypatch, tmp_path):
    options, *_ = witnessed
    base = prior.admit_fold(**options)
    smokes_dir = tmp_path / "explicit-synthetic-smoke-witnesses"
    smokes_dir.mkdir()
    hashes = {}
    for model in ("frame", "temporal"):
        root = smokes_dir / f"{model}-smoke-v1"
        root.mkdir()
        rows = []
        for step in range(1, 6):
            path = root / f"step-{step:07d}.pt"
            path.write_bytes(f"SYNTHETIC software bytes, never Torch checkpoint {model} {step}".encode())
            rows.append({"optimizer_steps": step, "path": str(path), "sha256": audit.sha(path)})
        config = audit._read(full.ROOT / f"experiments/{model}-v1.json")
        counts = base.class_counts_train_only
        state = {
            "schema_version": 1, "state": "completed", "smoke_completed": True,
            "error": None, "job_id": ("f" if model == "frame" else "e") * 32,
            "evidence_type": "five-update-training-only-smoke-not-completed-benchmark-training",
            "immutable": {
                "admission": deepcopy(base.immutable), "config": config,
                "mode": "smoke", "fold": 2, "model": model, "total_optimizer_batches": 5,
                "initialization": "fresh-seeded-head-no-warm-start",
                "implementation_sha256": full._implementation(model),
                "runtime": full._runtime_identity(), "run_dir": str(root),
                "lock_path": str((full.ROOT / "artifacts/training/active-training.lock").resolve()),
            },
            "summary": {"evidence_type": "inner-validation-head-training-not-outer-benchmark",
                        "state": "partial", "optimizer_steps": 5, "completed_epochs": 0,
                        "clips_processed_in_current_epoch": 20, "history": [],
                        "best_epoch": None, "best_validation_pce": None,
                        "class_counts_train_only": counts, "outer_test_evaluated": False,
                        "validated_for_serving": False},
            "class_counts_train_only": counts, "attempts": [{"state": "partial"}, {"state": "completed"}],
            "initial_model_tensor_sha256": "a" * 64, "final_model_tensor_sha256": "b" * 64,
            "checkpoints": rows, **full.SMOKE_CLOSED_GATES,
        }
        write(root / "run.json", state)
        hashes[model] = audit.sha(root / "run.json")
    monkeypatch.setattr(full, "KNOWN_SMOKE_RUN_SHA256", hashes)
    return {**options, "smokes_dir": smokes_dir}, hashes


def reseal(options, hashes, model, edit):
    path = options["smokes_dir"] / f"{model}-smoke-v1/run.json"
    state = audit._read(path)
    edit(state)
    write(path, state)
    hashes[model] = audit.sha(path)


def test_production_known_smoke_hashes_are_independent_fixed_values():
    assert full.KNOWN_SMOKE_RUN_SHA256 == {
        "frame": "eb0728446686ad3db71960da17a131722f103cff0cb575589152795c70294657",
        "temporal": "9b0bc9acc8c65a5a1898e9bebc415bdbed8b8c9c49e9267ec78e6d8b2ed41736",
    }


@pytest.mark.parametrize("fold", [2, 3, 4])
def test_complete_witnesses_admit_new_fold_without_reusing_smoke_partition(ready, fold):
    options, hashes = ready
    result = full.admit_full_fold(**{**options, "fold": fold})
    expected = set(result.manifest["partitions"]["train"]["clip_ids"]
                   + result.manifest["partitions"]["validation"]["clip_ids"])
    assert set(result.cache_paths) == expected
    assert not expected & set(result.manifest["partitions"]["test"]["clip_ids"])
    assert result.immutable["full_training_input_admitted"] is True
    assert result.immutable["validated_for_serving"] is False
    assert {k: v["run_sha256"] for k, v in result.immutable["smoke_witnesses"].items()} == hashes
    assert len([p for p in result.watched if p.endswith(".pt")]) == 10
    full.recheck_full(result)


@pytest.mark.parametrize("model", ["frame", "temporal"])
def test_unknown_or_missing_smoke_is_not_an_admission_token(ready, model):
    options, _ = ready
    path = options["smokes_dir"] / f"{model}-smoke-v1/run.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="independently known SHA256"):
        full.admit_full_fold(**options)
    path.unlink()
    with pytest.raises(OSError):
        full.admit_full_fold(**options)


@pytest.mark.parametrize("edit", [
    lambda s: s.update(state="running"),
    lambda s: s.update(smoke_completed=False),
    lambda s: s.update(schema_version=True),
    lambda s: s.update(validated_for_serving=True),
    lambda s: s.update(advice=["not validated"]),
    lambda s: s.update(error="interrupted"),
    lambda s: s["summary"].update(optimizer_steps=True),
    lambda s: s["summary"].update(completed_epochs=1),
    lambda s: s["summary"].update(history=[{"not": "training-only"}]),
    lambda s: s["summary"].update(best_epoch=1),
    lambda s: s["summary"].update(best_validation_pce=.2),
    lambda s: s["immutable"].update(model="other"),
    lambda s: s["immutable"].update(fold=3),
    lambda s: s["immutable"].update(total_optimizer_batches=6),
    lambda s: s["immutable"].update(initialization="warm-start"),
    lambda s: s["immutable"]["config"].update(context="reference-64-reset"),
    lambda s: s["immutable"]["runtime"].update(torch="different-build"),
    lambda s: s["immutable"]["admission"].update(dataset_sha256="0" * 64),
    lambda s: s["immutable"]["admission"].update(fold=3),
    lambda s: s["immutable"]["admission"]["class_counts_train_only"].__setitem__(0, 99),
    lambda s: s["immutable"]["implementation_sha256"].__setitem__(str(full.ROOT / "uv.lock"), "0" * 64),
    lambda s: s.update(attempts=[{"state": "completed"}]),
    lambda s: s.update(initial_model_tensor_sha256=s["final_model_tensor_sha256"]),
    lambda s: s["checkpoints"].pop(),
    lambda s: s["checkpoints"][0].update(optimizer_steps=True),
    lambda s: s["checkpoints"][0].update(path="/private/tmp/unrelated.pt"),
])
def test_resealed_hypothetical_invalid_witness_is_semantically_refused(ready, edit):
    options, hashes = ready
    reseal(options, hashes, "temporal", edit)
    with pytest.raises(ValueError):
        full.admit_full_fold(**options)


@pytest.mark.parametrize("model", ["frame", "temporal"])
def test_changed_or_unreferenced_checkpoint_is_refused(ready, model):
    options, _ = ready
    root = options["smokes_dir"] / f"{model}-smoke-v1"
    path = root / "step-0000005.pt"
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="selected cache changed"):
        full.admit_full_fold(**options)
    path.write_bytes(original)
    (root / "step-0000006.pt").write_bytes(b"unreferenced synthetic bytes")
    with pytest.raises(ValueError, match="Unreferenced smoke"):
        full.admit_full_fold(**options)


@pytest.mark.parametrize("kind", ["checkpoint", "ledger", "watch-map"])
def test_late_witness_or_dependency_drift_refused(ready, kind):
    options, _ = ready
    result = full.admit_full_fold(**options)
    if kind == "watch-map":
        result.watched.pop(next(iter(result.watched)))
    else:
        path = options["smokes_dir"] / "temporal-smoke-v1" / ("run.json" if kind == "ledger" else "step-0000001.pt")
        path.write_bytes(path.read_bytes() + b"changed after admission")
    with pytest.raises(ValueError, match="changed"):
        full.recheck_full(result)


def test_current_runtime_drift_rejected_without_model_or_tensor_load(ready, monkeypatch):
    options, _ = ready
    original = full._runtime_identity
    monkeypatch.setattr(full, "_runtime_identity", lambda: {**original(), "python_executable": "/different/python"})
    with pytest.raises(ValueError, match="runtime"):
        full.admit_full_fold(**options)


def test_no_outer_array_video_model_or_checkpoint_tensor_is_opened(ready, monkeypatch):
    options, _ = ready
    import torch
    from backend.swingsync.research.golfdb import make_split_manifest, load_annotations
    manifest = make_split_manifest(load_annotations(options["annotations"]), 3, 20261104, .2)
    report = audit._read(Path(options["audit_run"]).with_name("audit.json"))
    forbidden = {Path(report["clips"][str(i)]["cache_path"]) for i in manifest["partitions"]["test"]["clip_ids"]}
    original = audit.sha
    def safe_sha(path):
        assert Path(path).resolve() not in forbidden
        assert Path(path).suffix != ".mp4"
        return original(path)
    monkeypatch.setattr(audit, "sha", safe_sha)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Smoke tensors opened"))
    monkeypatch.setattr(audit, "_load_cache", lambda *a, **k: pytest.fail("Cache arrays opened"))
    result = full.admit_full_fold(**{**options, "fold": 3})
    full.recheck_full(result)
