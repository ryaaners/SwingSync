"""Synthetic cache identities prove software behavior, never learned results.

The fixture replaces prospective protocol admission with a synthetic registration;
production CLI still requires the separately pinned real protocol validator.
"""

from copy import deepcopy
import fcntl
import json
import hashlib
import importlib.metadata
import inspect
import marshal
import platform
from pathlib import Path

import numpy as np
import pytest
from test_golfdb import write_mat

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from backend.swingsync.research import fold_cache_audit as audit  # noqa: E402
from backend.swingsync.research.features import FeatureClip, _array_hash, cache_key, save_feature_cache  # noqa: E402
from backend.swingsync.research.golfdb import load_annotations, make_split_manifest, write_split_manifest  # noqa: E402


def synthetic_runtime_pipeline():
    # These are declared fixture identities for zero-valued synthetic arrays,
    # not assertions that ImageNet feature extraction actually ran in this test.
    from backend.swingsync.research.models import FrozenMobileNetV2

    pipeline = json.loads((audit.ROOT / "experiments/event-v1.json").read_text())["feature_pipeline"]
    pipeline.update(
        python=platform.python_version(),
        numpy=np.__version__,
        pyav=audit.av.__version__,
        torch=importlib.metadata.version("torch"),
        torchvision=importlib.metadata.version("torchvision"),
        ffmpeg_libraries=json.loads(json.dumps(audit.av.library_versions)),
    )
    pipeline["backbone_provenance"].update(torch=pipeline["torch"], torchvision=pipeline["torchvision"])
    implementations = {
        name: audit.sha(audit.ROOT / "backend/swingsync/research" / name)
        for name in ("features.py", "models.py", "timeline.py", "golfdb.py")
    }
    implementations["backbone_module"] = audit.sha(inspect.getfile(FrozenMobileNetV2))
    implementations["backbone_forward_code"] = hashlib.sha256(
        marshal.dumps(FrozenMobileNetV2.forward.__code__)
    ).hexdigest()
    pipeline["implementation_sha256"] = implementations
    return pipeline


@pytest.fixture
def toy(tmp_path, monkeypatch):
    annotation = write_mat(
        tmp_path / "synthetic.mat", lambda i: {"events": np.array([1, 1, 2, 3, 4, 5, 7, 6, 8, 9 + i])}
    )
    dataset = load_annotations(annotation)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for fold in (1, 2, 3, 4):
        write_split_manifest(
            make_split_manifest(dataset, fold, 20261104, 0.2), manifests / f"fold-{fold}.json"
        )
    pipeline = synthetic_runtime_pipeline()
    runtime = audit._runtime_pipeline(pipeline)
    registration = tmp_path / "synthetic-feature-registration.json"
    registration.write_text(json.dumps({"feature_pipeline": pipeline}))
    clips = tmp_path / "videos"
    clips.mkdir()
    caches = tmp_path / "caches"
    caches.mkdir()
    entries = {}
    for record in dataset.records:
        video = clips / f"{record.clip_id}.mp4"
        video.write_bytes(f"SYNTHETIC bytes for checksum only {record.clip_id}".encode())
        n = record.expected_frame_count
        identity = {
            "schema_version": 1,
            "video_sha256": audit.sha(video),
            "annotation_sha256": dataset.sha256,
            "clip_id": record.clip_id,
            "source_id": record.source_id,
            "outer_split": record.outer_split,
            "original_events": list(record.original_events),
            "expected_frame_count": n,
            "historical_source_indices": [record.source_frame_index(i) for i in range(n)],
            "annotation_event_indices": list(record.event_indices),
            "timestamp_provenance": "reencoded-clip-pts",
            **pipeline,
        }
        labels = np.full(n, 8, np.int64)
        labels[list(record.event_indices)] = np.arange(8)
        arrays = {
            "features": np.full((n, 1280), record.clip_id, np.float32),
            "labels": labels,
            "pts_seconds": np.array([5 + i * 0.04 for i in range(n)], np.float64),
            "historical_source_indices": np.array(identity["historical_source_indices"], np.int64),
        }
        clip = FeatureClip(
            **arrays,
            metadata={
                "identity": identity,
                "cache_key": cache_key(identity),
                "array_sha256": {name: _array_hash(value) for name, value in arrays.items()},
            },
        )
        path = caches / (cache_key(identity) + ".npz")
        save_feature_cache(clip, path)
        entries[record.clip_id] = {
            "clip_id": record.clip_id,
            "source_id": record.source_id,
            "cache_key": cache_key(identity),
            "cache_path": str(path),
            "cache_sha256": audit.sha(path),
            "frames": n,
            "completed_at": "synthetic",
            "reused": False,
        }
    fold1 = json.loads((manifests / "fold-1.json").read_text())
    protocol = {
        "split": {"seed": 20261104, "validation_fraction_sources": 0.2},
        "feature_registration": {
            "path": str(registration),
            "sha256": audit.sha(registration),
        },
        "dependency_lock_sha256": audit.sha(audit.ROOT / "uv.lock"),
        "cache_catalogs": {},
    }
    catalogs = {}
    for role, partitions in (("train_validation", ("train", "validation")), ("test", ("test",))):
        membership = {i: part for part in partitions for i in fold1["partitions"][part]["clip_ids"]}
        cfg = {
            "annotations_sha256": dataset.sha256,
            "manifest_sha256": audit.sha(manifests / "fold-1.json"),
            "partitions": list(partitions),
            "allow_test": "test" in partitions,
            "device": "cpu",
            "batch_size": 16,
            "torch_threads": 1,
            "deterministic_algorithms": True,
            "weights_request": "IMAGENET1K_V2",
            "dependency_lock_sha256": audit.sha(audit.ROOT / "uv.lock"),
            "clips_dir": str(clips),
            "cache_dir": str(caches),
            "job_implementation_sha256": audit.sha(audit.ROOT / "backend/swingsync/research/cache_job.py"),
            **runtime,
            "inventory": [
                {
                    "clip_id": i,
                    "source_id": entries[i]["source_id"],
                    "partition": membership[i],
                    "video_sha256": audit.sha(clips / f"{i}.mp4"),
                }
                for i in sorted(membership)
            ],
        }
        cfg.pop("versions")
        catalog = {
            "schema_version": 1,
            "state": "completed",
            "error": None,
            "validated_for_serving": False,
            "outer_test_selected": "test" in partitions,
            "completed_clip_count": len(membership),
            "target_clip_count": len(membership),
            "model_identity": pipeline,
            "config": cfg,
            "config_sha256": audit.json_hash(cfg),
            "completed": {str(i): {**entries[i], "partition": membership[i]} for i in membership},
        }
        catalog["job_fingerprint"] = audit.json_hash(
            {"config_sha256": catalog["config_sha256"], "model_identity": pipeline}
        )
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(catalog))
        protocol["cache_catalogs"][role] = {"path": str(path), "sha256": audit.sha(path)}
        catalogs[role] = path
    config = tmp_path / "fixed-config.json"
    config.write_text("synthetic registered config")
    protocol["original_configs"] = {"frame": {"path": str(config), "sha256": audit.sha(config)}}
    protocol["frozen_source_sha256"] = {}
    protocol_path = tmp_path / "synthetic-protocol.json"
    protocol_path.write_text('{"synthetic_registration_only":true}')
    monkeypatch.setattr(audit, "_protocol", lambda *args: deepcopy(protocol))
    monkeypatch.setattr(audit, "LOCK_PATH", tmp_path / "shared.lock")
    return (
        {
            "annotations": annotation,
            "train_catalog": catalogs["train_validation"],
            "test_catalog": catalogs["test"],
            "manifests_dir": manifests,
            "clips_dir": clips,
            "output": tmp_path / "audit",
            "protocol_path": protocol_path,
        },
        protocol,
        entries,
        dataset,
        config,
    )


def refresh(path, protocol, role):
    catalog = json.loads(path.read_text())
    catalog["config_sha256"] = audit.json_hash(catalog["config"])
    catalog["job_fingerprint"] = audit.json_hash(
        {"config_sha256": catalog["config_sha256"], "model_identity": catalog["model_identity"]}
    )
    path.write_text(json.dumps(catalog))
    protocol["cache_catalogs"][role]["sha256"] = audit.sha(path)


def test_complete_audit_reuses_bytes_without_relabeling_or_training(toy):
    options, _, entries, dataset, _ = toy
    before = {path: audit.sha(path) for path in [Path(row["cache_path"]) for row in entries.values()]}
    result = audit.run_fold_cache_audit(**options)
    report = json.loads((options["output"] / "audit.json").read_text())
    assert result["state"] == "completed" and result["completed_clip_count"] == 16
    assert result["audit_sha256"] == audit.sha(options["output"] / "audit.json")
    assert report["artifact_type"] == "read-only-cache-reuse-audit-not-training-catalog"
    assert report["fresh_video_decode_performed"] is False
    for key, value in audit.FALSE_GATES.items():
        assert report[key] == value
    for fold in (2, 3, 4):
        manifest = make_split_manifest(dataset, fold, 20261104, 0.2)
        assert report["folds"][str(fold)]["partitions"] == manifest["partitions"]
        ids = manifest["partitions"]["train"]["clip_ids"]
        assert report["folds"][str(fold)]["class_counts_train_only"] == [len(ids)] * 8 + [
            sum(dataset.records[i].expected_frame_count - 8 for i in ids)
        ]
    assert report["clips"]["0"]["official_outer_split"] == 1
    assert before == {path: audit.sha(path) for path in before}
    with pytest.raises(FileExistsError):
        audit.run_fold_cache_audit(**options)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("state", "partial"),
        ("completed_clip_count", True),
        ("target_clip_count", 1),
        ("validated_for_serving", True),
        ("outer_test_selected", True),
    ],
)
def test_catalog_completion_and_coverage_fail_closed(toy, field, value):
    options, protocol, *_ = toy
    path = options["train_catalog"]
    catalog = json.loads(path.read_text())
    catalog[field] = value
    path.write_text(json.dumps(catalog))
    refresh(path, protocol, "train_validation")
    with pytest.raises(ValueError, match="catalog"):
        audit.run_fold_cache_audit(**options)
    assert json.loads((options["output"] / "run.json").read_text())["state"] == "failed"
    assert not (options["output"] / "audit.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["completed"].pop(next(iter(c["completed"]))),
        lambda c: c["config"]["inventory"].append(c["config"]["inventory"][0]),
        lambda c: c["config"]["inventory"][0].update(partition="test"),
        lambda c: c["completed"][next(iter(c["completed"]))].update(source_id="other"),
        lambda c: c["completed"][next(iter(c["completed"]))].update(frames=999),
        lambda c: c["config"].update(deterministic_algorithms=False),
        lambda c: c["config"].update(weights_request="trained-backbone"),
    ],
)
def test_catalog_semantics_reject_wrong_lineage_or_fitted_pipeline(toy, mutation):
    options, protocol, *_ = toy
    path = options["train_catalog"]
    catalog = json.loads(path.read_text())
    mutation(catalog)
    path.write_text(json.dumps(catalog))
    refresh(path, protocol, "train_validation")
    with pytest.raises(ValueError):
        audit.run_fold_cache_audit(**options)


def test_unregistered_catalog_bytes_refused(toy):
    options, *_ = toy
    options["train_catalog"].write_bytes(options["train_catalog"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="dependency|bytes"):
        audit.run_fold_cache_audit(**options)


def test_source_media_and_cache_corruption_refused(toy):
    options, _, entries, *_ = toy
    Path(entries[0]["cache_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes changed"):
        audit.run_fold_cache_audit(**options)


def test_changed_video_refused_before_cache_load(toy, monkeypatch):
    options, *_ = toy
    (options["clips_dir"] / "0.mp4").write_bytes(b"changed")
    monkeypatch.setattr(
        audit, "_load_cache", lambda _: pytest.fail("Changed media must fail before array loading")
    )
    with pytest.raises(ValueError, match="bytes changed"):
        audit.run_fold_cache_audit(**options)


def test_manifest_source_group_or_membership_tamper_refused(toy):
    options, *_ = toy
    p = options["manifests_dir"] / "fold-2.json"
    m = json.loads(p.read_text())
    m["partitions"]["train"]["clip_ids"][0] = m["partitions"]["test"]["clip_ids"][0]
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="source-grouped"):
        audit.run_fold_cache_audit(**options)


def test_shared_lock_refuses_second_output_without_starting(toy):
    options, *_ = toy
    with audit.LOCK_PATH.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="shared expensive-work lock"):
            audit.run_fold_cache_audit(**options)
    assert not options["output"].exists()


def test_loader_semantic_mismatch_refused_even_with_known_file_hash(toy, monkeypatch):
    options, *_ = toy
    original = audit._load_cache

    def wrong(path):
        clip = original(path)
        clip.metadata["identity"]["annotation_event_indices"] = sorted(
            clip.metadata["identity"]["annotation_event_indices"]
        )
        return clip

    monkeypatch.setattr(audit, "_load_cache", wrong)
    with pytest.raises(ValueError, match="annotation/pipeline"):
        audit.run_fold_cache_audit(**options)


def test_protocol_declared_config_drift_after_admission_refused(toy, monkeypatch):
    options, _, _, _, config = toy
    original = audit._check_clip
    changed = []

    def drift(*args):
        if not changed:
            config.write_text("changed registered config")
            changed.append(True)
        return original(*args)

    monkeypatch.setattr(audit, "_check_clip", drift)
    with pytest.raises(ValueError, match="inputs or implementation"):
        audit.run_fold_cache_audit(**options)
    assert not (options["output"] / "audit.json").exists()


def test_source_drift_during_final_completed_ledger_refused(toy, monkeypatch):
    options, *_ = toy
    original_write = audit._write
    original_sources = audit._source_hashes
    changed = []

    def publish(path, value):
        original_write(path, value)
        if path.name == "run.json" and value.get("state") == "completed":
            changed.append(True)

    def sources():
        value = original_sources()
        return {**value, "synthetic-drift": "0" * 64} if changed else value

    monkeypatch.setattr(audit, "_write", publish)
    monkeypatch.setattr(audit, "_source_hashes", sources)
    with pytest.raises(ValueError, match="inputs or implementation"):
        audit.run_fold_cache_audit(**options)
    assert json.loads((options["output"] / "run.json").read_text())["state"] == "failed"
    assert not (options["output"] / "audit.json").exists()


def test_protocol_cannot_promote_fitted_or_stochastic_features(toy, monkeypatch):
    options, *_ = toy
    original = audit._runtime_pipeline

    def incompatible(pipeline):
        different = deepcopy(pipeline)
        different["torch"] = "unregistered"
        return original(different)

    monkeypatch.setattr(audit, "_runtime_pipeline", incompatible)
    with pytest.raises(ValueError, match="runtime differs"):
        audit.run_fold_cache_audit(**options)


@pytest.mark.parametrize("target", ["audit.json", "run.json"])
def test_modified_published_output_cannot_be_bound_as_valid(toy, monkeypatch, target):
    options, *_ = toy
    original = audit._write

    def change_after_write(path, value):
        original(path, value)
        if path.name == target and (target == "audit.json" or value.get("state") == "completed"):
            changed = json.loads(path.read_text())
            changed["tampered_after_publication"] = True
            path.write_text(json.dumps(changed))

    monkeypatch.setattr(audit, "_write", change_after_write)
    with pytest.raises(ValueError, match="output changed|ledger changed"):
        audit.run_fold_cache_audit(**options)
    assert json.loads((options["output"] / "run.json").read_text())["state"] == "failed"
    assert not (options["output"] / "audit.json").exists()


def test_exact_runtime_build_metadata_is_not_normalized():
    pipeline = synthetic_runtime_pipeline()
    version = pipeline["torch"]
    pipeline["torch"] = version.split("+")[0] if "+" in version else version + "+cpu"
    with pytest.raises(ValueError, match="runtime differs"):
        audit._runtime_pipeline(pipeline)


def test_strict_production_runtime_rejects_another_source_path():
    from backend.swingsync.research.models import FrozenMobileNetV2

    pipeline = synthetic_runtime_pipeline()
    code = FrozenMobileNetV2.forward.__code__.replace(co_filename="/synthetic/different-root/models.py")
    pipeline["implementation_sha256"]["backbone_forward_code"] = hashlib.sha256(
        marshal.dumps(code)
    ).hexdigest()
    with pytest.raises(ValueError, match="implementation differs"):
        audit._runtime_pipeline(pipeline)
