"""Read-only reuse audit of frozen GolfDB pixels, never training admission.

Known completed catalog bytes bind historical cache arrays and reencoded media.
This audit re-hashes media and cache bytes but does not decode video again, load
head checkpoints, execute a backbone, fit preprocessing, or compute predictions.
"""

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import marshal
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

import av
import numpy as np

from .golfdb import load_annotations, make_split_manifest

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "experiments/additional-folds-v1.json"
LOCK_PATH = ROOT / "artifacts/training/active-training.lock"
PIPELINE_KEYS = (
    "backbone_class",
    "backbone_state_sha256",
    "backbone_provenance",
    "transform",
    "numpy",
    "torch",
    "torchvision",
    "pyav",
    "ffmpeg_libraries",
    "extraction_device",
    "implementation_sha256",
    "python",
)
FALSE_GATES = {
    "training_admitted": False,
    "outer_evaluation_admitted": False,
    "normal_model_available": False,
    "validated_for_serving": False,
    "training_performed": False,
    "inference_performed": False,
    "accuracy_metrics_computed": False,
    "advice": [],
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _serialized_sha(value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".part")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _process_start():
    try:
        result = subprocess.run(
            ["ps", "-p", str(os.getpid()), "-o", "lstart="],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _read(path):
    def unique(pairs):
        value = {}
        for key, child in pairs:
            if key in value:
                raise ValueError("Duplicate audit-input JSON field")
            value[key] = child
        return value

    try:
        return json.loads(
            Path(path).read_text(),
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON input")),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Malformed audit-input JSON") from exc


def _protocol(path, dataset, manifests_dir):
    from .fold_protocol import validate_protocol

    return validate_protocol(path, dataset=dataset, manifests_dir=manifests_dir)


def _runtime_pipeline(pipeline):
    """Inspect the original runtime/code only; do not construct any model."""
    try:
        from . import features
        from .models import FrozenMobileNetV2
    except ImportError as exc:
        raise ValueError("Audit requires the original locked ML runtime, without model execution.") from exc
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyav": av.__version__,
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
        "ffmpeg_libraries": json.loads(json.dumps(av.library_versions)),
    }
    if any(pipeline[key] != value for key, value in versions.items()):
        raise ValueError("Audit runtime differs from the historical frozen feature pipeline")
    expected = {
        name: sha(ROOT / "backend/swingsync/research" / name)
        for name in ("features.py", "models.py", "timeline.py", "golfdb.py")
    }
    expected["backbone_module"] = sha(inspect.getfile(FrozenMobileNetV2))
    expected["backbone_forward_code"] = hashlib.sha256(
        marshal.dumps(FrozenMobileNetV2.forward.__code__)
    ).hexdigest()
    if expected != pipeline["implementation_sha256"]:
        raise ValueError("Frozen feature implementation differs from its registered identity")
    return {
        "versions": versions,
        "extractor_module_sha256": sha(inspect.getfile(features)),
        "extractor_implementation_sha256": hashlib.sha256(
            marshal.dumps(features.extract_clip_features.__code__)
        ).hexdigest(),
    }


def _load_cache(path):
    from .features import load_feature_cache

    return load_feature_cache(path)


def _source_hashes():
    paths = [
        Path(__file__),
        ROOT / "scripts/audit_fold_caches.py",
        ROOT / "backend/swingsync/research/fold_protocol.py",
        ROOT / "uv.lock",
    ]
    paths += [
        ROOT / "backend/swingsync/research" / name
        for name in (
            "features.py",
            "models.py",
            "timeline.py",
            "golfdb.py",
            "cache_job.py",
            "experiment.py",
            "training.py",
            "evaluation.py",
        )
    ]
    return {str(p.resolve()): sha(p) for p in paths}


def _protocol_dependencies(protocol):
    """Bind every declared registration and frozen source, not just the cache inputs."""
    watched = {}

    def add(path, expected):
        path = (ROOT / Path(path)).resolve()
        if sha(path) != expected:
            raise ValueError("A protocol dependency changed after admission")
        watched[str(path)] = expected

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                add(value["path"], value["sha256"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(protocol)
    for path, digest in protocol.get("frozen_source_sha256", {}).items():
        add(path, digest)
    add("uv.lock", protocol["dependency_lock_sha256"])
    return watched


def _manifest(dataset, directory, fold, seed, fraction):
    path = directory / f"fold-{fold}.json"
    value = _read(path)
    expected = make_split_manifest(dataset, fold, seed, fraction)
    if value != expected:
        raise ValueError(f"Fold{fold} manifest differs from regenerated source-grouped partitions")
    return value


def _admit_catalog(
    catalog, expected_sha, path, dataset, fold1, manifest_path, pipeline, runtime, partitions, clips_dir
):
    if sha(path) != expected_sha:
        raise ValueError("Historical completed catalog bytes differ from registration")
    ids = {i for part in partitions for i in fold1["partitions"][part]["clip_ids"]}
    members = {i: part for part in partitions for i in fold1["partitions"][part]["clip_ids"]}
    if (
        not isinstance(catalog, dict)
        or catalog.get("state") != "completed"
        or type(catalog.get("schema_version")) is not int
        or catalog.get("schema_version") != 1
        or catalog.get("validated_for_serving") is not False
        or any(
            type(catalog.get(key)) is not int or catalog[key] != len(ids)
            for key in ("completed_clip_count", "target_clip_count")
        )
        or catalog.get("error") is not None
        or catalog.get("outer_test_selected") is not ("test" in partitions)
        or catalog.get("model_identity") != pipeline
    ):
        raise ValueError("Historical catalog completion, model or coverage is invalid")
    cfg = catalog.get("config", {})
    if not isinstance(cfg, dict):
        raise ValueError("Historical catalog configuration is invalid")
    if (
        catalog.get("config_sha256") != json_hash(cfg)
        or catalog.get("job_fingerprint")
        != json_hash({"config_sha256": json_hash(cfg), "model_identity": pipeline})
        or cfg.get("annotations_sha256") != dataset.sha256
        or cfg.get("manifest_sha256") != sha(manifest_path)
        or cfg.get("partitions") != list(partitions)
        or cfg.get("allow_test") is not ("test" in partitions)
        or cfg.get("device") != "cpu"
        or type(cfg.get("batch_size")) is not int
        or cfg.get("batch_size") != 16
        or type(cfg.get("torch_threads")) is not int
        or cfg.get("torch_threads") != 1
        or cfg.get("deterministic_algorithms") is not True
        or cfg.get("weights_request") != "IMAGENET1K_V2"
        or cfg.get("dependency_lock_sha256") != sha(ROOT / "uv.lock")
        or cfg.get("job_implementation_sha256") != sha(ROOT / "backend/swingsync/research/cache_job.py")
        or any(
            cfg.get(key) != runtime[key]
            for key in ("extractor_module_sha256", "extractor_implementation_sha256")
        )
        or Path(cfg.get("clips_dir", "")).resolve() != clips_dir
    ):
        raise ValueError("Historical catalog configuration/fingerprint differs")
    entries = catalog.get("completed")
    inventory = cfg.get("inventory")
    if (
        not isinstance(entries, dict)
        or set(entries) != {str(i) for i in ids}
        or not isinstance(inventory, list)
    ):
        raise ValueError("Historical catalog coverage differs")
    records = {r.clip_id: r for r in dataset.records}
    videos = {}
    for row in inventory:
        if (
            type(row) is not dict
            or set(row) != {"clip_id", "source_id", "partition", "video_sha256"}
            or type(row["clip_id"]) is not int
            or row["clip_id"] not in ids
            or row["clip_id"] in videos
            or row["source_id"] != records[row["clip_id"]].source_id
            or row["partition"] != members[row["clip_id"]]
            or type(row["video_sha256"]) is not str
            or len(row["video_sha256"]) != 64
        ):
            raise ValueError("Historical source-video inventory is invalid")
        videos[row["clip_id"]] = row["video_sha256"]
    if set(videos) != ids:
        raise ValueError("Historical source-video inventory coverage differs")
    result = {}
    cache_dir = Path(cfg.get("cache_dir", "")).resolve()
    for clip_id in sorted(ids):
        row = entries[str(clip_id)]
        record = records[clip_id]
        if (
            not isinstance(row, dict)
            or type(row.get("clip_id")) is not int
            or row.get("clip_id") != clip_id
            or row.get("source_id") != record.source_id
            or row.get("partition") != members[clip_id]
            or type(row.get("frames")) is not int
            or row.get("frames") != record.expected_frame_count
            or type(row.get("cache_key")) is not str
            or len(row["cache_key"]) != 64
            or type(row.get("cache_sha256")) is not str
            or len(row["cache_sha256"]) != 64
        ):
            raise ValueError("Historical completed cache entry differs from annotations")
        cache = Path(row["cache_path"])
        if cache.is_symlink() or cache.resolve() != cache_dir / (row["cache_key"] + ".npz"):
            raise ValueError("Historical cache path is not its immutable cache-key path")
        result[clip_id] = {
            "entry": row,
            "cache": cache.resolve(),
            "video": clips_dir / f"{clip_id}.mp4",
            "video_sha256": videos[clip_id],
            "prior_partition": members[clip_id],
        }
    return result


def _check_clip(record, dataset_sha, row, pipeline):
    cache, video = row["cache"], row["video"]
    if video.is_symlink() or sha(video) != row["video_sha256"] or sha(cache) != row["entry"]["cache_sha256"]:
        raise ValueError(f"Cache or source-video bytes changed for clip{record.clip_id}")
    clip = _load_cache(cache)
    identity = {
        "schema_version": 1,
        "video_sha256": row["video_sha256"],
        "annotation_sha256": dataset_sha,
        "clip_id": record.clip_id,
        "source_id": record.source_id,
        "outer_split": record.outer_split,
        "original_events": list(record.original_events),
        "expected_frame_count": record.expected_frame_count,
        "historical_source_indices": [
            record.source_frame_index(i) for i in range(record.expected_frame_count)
        ],
        "annotation_event_indices": list(record.event_indices),
        "timestamp_provenance": "reencoded-clip-pts",
        **pipeline,
    }
    n = record.expected_frame_count
    labels = np.full(n, 8, dtype=np.int64)
    labels[list(record.event_indices)] = np.arange(8)
    if (
        clip.metadata.get("identity") != identity
        or clip.metadata.get("cache_key") != json_hash(identity)
        or clip.metadata["cache_key"] != row["entry"]["cache_key"]
        or clip.features.shape != (n, 1280)
        or not np.array_equal(clip.labels, labels)
        or not np.array_equal(clip.historical_source_indices, identity["historical_source_indices"])
    ):
        raise ValueError(f"Cached annotation/pipeline/shape identity differs for clip{record.clip_id}")
    if sha(cache) != row["entry"]["cache_sha256"] or sha(video) != row["video_sha256"]:
        raise ValueError("Cache/source changed during validation")
    return clip


def _unchanged(inputs, sources, rows, output=None, expected_output_sha=None):
    if any(sha(path) != digest for path, digest in inputs.items()) or _source_hashes() != sources:
        raise ValueError("Audit inputs or implementation changed during publication")
    for row in rows.values():
        if sha(row["cache"]) != row["entry"]["cache_sha256"] or sha(row["video"]) != row["video_sha256"]:
            raise ValueError("Audited media/cache bytes changed during publication")
    if output is not None and sha(output) != expected_output_sha:
        raise ValueError("Audit output changed during publication")


def run_fold_cache_audit(
    annotations, train_catalog, test_catalog, manifests_dir, clips_dir, output, *, protocol_path=None
):
    """Exclusive read-only audit. There is deliberately no resume or fitting path."""
    protocol_path = Path(protocol_path or DEFAULT_PROTOCOL).resolve()
    annotations, train_catalog, test_catalog, manifests_dir, clips_dir, output = (
        Path(p).resolve()
        for p in (annotations, train_catalog, test_catalog, manifests_dir, clips_dir, output)
    )
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another job owns the shared expensive-work lock") from exc
        output.mkdir(parents=True, exist_ok=False)
        state = {
            "schema_version": 1,
            "state": "initializing",
            "job_id": uuid.uuid4().hex,
            "pid": os.getpid(),
            "process_start_identity": _process_start(),
            "created_at": _now(),
            "completed_clip_count": 0,
            "evidence_scope": "Read-only frozen-cache reuse audit; not a training catalog, fit or evaluation",
            "fresh_video_decode_performed": False,
            **FALSE_GATES,
        }
        _write(output / "run.json", state)
        report_path = output / "audit.json"
        started = time.monotonic()
        try:
            input_paths = [annotations, train_catalog, test_catalog, protocol_path]
            input_paths += [manifests_dir / f"fold-{fold}.json" for fold in (1, 2, 3, 4)]
            state["inputs_sha256"] = {str(p): sha(p) for p in input_paths}
            state["implementation_sha256"] = _source_hashes()
            state["verification_runtime"] = {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyav": av.__version__,
                "machine": platform.machine(),
            }
            _write(output / "run.json", state)
            dataset = load_annotations(annotations)
            protocol = _protocol(protocol_path, dataset, manifests_dir)
            state["inputs_sha256"].update(_protocol_dependencies(protocol))
            _write(output / "run.json", state)
            registration_path = ROOT / protocol["feature_registration"]["path"]
            if sha(registration_path) != protocol["feature_registration"]["sha256"]:
                raise ValueError("Frozen feature registration changed")
            state["inputs_sha256"][str(registration_path)] = sha(registration_path)
            registration = _read(registration_path)
            pipeline = registration["feature_pipeline"]
            if (
                set(pipeline) != set(PIPELINE_KEYS)
                or pipeline["backbone_provenance"].get("swingsync_event_training") is not False
                or pipeline["backbone_provenance"].get("weights") != "IMAGENET1K_V2"
                or pipeline["transform"].get("stochastic") is not False
            ):
                raise ValueError("Only the registered unfitted ImageNet frozen pipeline may be reused")
            runtime = _runtime_pipeline(pipeline)
            seed, fraction = protocol["split"]["seed"], protocol["split"]["validation_fraction_sources"]
            manifests = {
                fold: _manifest(dataset, manifests_dir, fold, seed, fraction) for fold in (1, 2, 3, 4)
            }
            rows = {}
            for role, catalog_path, partitions in (
                ("train_validation", train_catalog, ("train", "validation")),
                ("test", test_catalog, ("test",)),
            ):
                admitted = _admit_catalog(
                    _read(catalog_path),
                    protocol["cache_catalogs"][role]["sha256"],
                    catalog_path,
                    dataset,
                    manifests[1],
                    manifests_dir / "fold-1.json",
                    pipeline,
                    runtime,
                    partitions,
                    clips_dir,
                )
                if set(rows) & set(admitted):
                    raise ValueError("Historical catalogs overlap")
                rows.update(admitted)
            if set(rows) != {r.clip_id for r in dataset.records}:
                raise ValueError("Historical catalogs do not cover the complete annotated dataset")
            state.update(state="running", target_clip_count=len(rows), feature_pipeline=pipeline)
            _write(output / "run.json", state)
            memberships = {
                fold: set(manifests[fold]["partitions"]["train"]["clip_ids"]) for fold in (2, 3, 4)
            }
            counts = {fold: np.zeros(9, np.int64) for fold in memberships}
            clips = {}
            for record in dataset.records:
                row = rows[record.clip_id]
                clip = _check_clip(record, dataset.sha256, row, pipeline)
                for fold in memberships:
                    if record.clip_id in memberships[fold]:
                        counts[fold] += np.bincount(clip.labels, minlength=9)
                clips[str(record.clip_id)] = {
                    "clip_id": record.clip_id,
                    "source_id": record.source_id,
                    "official_outer_split": record.outer_split,
                    "cache_path": str(row["cache"]),
                    "cache_sha256": row["entry"]["cache_sha256"],
                    "cache_key": row["entry"]["cache_key"],
                    "source_video_sha256": row["video_sha256"],
                    "prior_catalog_partition": row["prior_partition"],
                    "frames": len(clip.labels),
                    "native_pts_sha256": hashlib.sha256(clip.pts_seconds.tobytes()).hexdigest(),
                }
                state.update(completed_clip_count=len(clips), last_clip_id=record.clip_id)
                _write(output / "run.json", state)
            folds = {}
            for fold in (2, 3, 4):
                manifest = manifests[fold]
                folds[str(fold)] = {
                    "manifest_sha256": sha(manifests_dir / f"fold-{fold}.json"),
                    "partitions": manifest["partitions"],
                    "class_counts_train_only": counts[fold].tolist(),
                }
            report = {
                "schema_version": 1,
                "artifact_type": "read-only-cache-reuse-audit-not-training-catalog",
                "dataset_sha256": dataset.sha256,
                "clip_count": len(clips),
                "frame_count": sum(x["frames"] for x in clips.values()),
                "feature_pipeline": pipeline,
                "protocol_sha256": sha(protocol_path),
                "clips": clips,
                "folds": folds,
                "fresh_video_decode_performed": False,
                "timestamp_scope": "Cached float64 native reencoded PTS validated against known catalog/cache bytes; no fresh video decode or original-source equivalence.",
                **FALSE_GATES,
            }
            _unchanged(state["inputs_sha256"], state["implementation_sha256"], rows)
            report_sha = _serialized_sha(report)
            _write(report_path, report)
            _unchanged(state["inputs_sha256"], state["implementation_sha256"], rows, report_path, report_sha)
            state.update(
                state="completed",
                completed_at=_now(),
                elapsed_seconds=time.monotonic() - started,
                audit_sha256=report_sha,
                frame_count=report["frame_count"],
            )
            final_state_sha = _serialized_sha(state)
            _write(output / "run.json", state)
            _unchanged(state["inputs_sha256"], state["implementation_sha256"], rows, report_path, report_sha)
            if sha(output / "run.json") != final_state_sha:
                raise ValueError("Completed audit ledger changed during publication")
            return state
        except BaseException as exc:
            report_path.unlink(missing_ok=True)
            state.update(state="failed", failed_at=_now(), error=f"{type(exc).__name__}: {exc}")
            _write(output / "run.json", state)
            raise
