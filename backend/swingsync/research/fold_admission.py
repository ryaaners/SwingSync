"""Authenticate additional-fold training inputs, without constructing a head.

The independently reviewed read-only audit is an input witness, never a trained
model or serving token. Only the requested fold's complete train/inner caches
are returned. Outer arrays, video decoders and old trained heads are not opened.
The caller must hold the shared expensive-work lock and persist launch identity.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import fold_cache_audit as audit
from .fold_protocol import DEFAULT_PROTOCOL
from .golfdb import load_annotations

KNOWN_AUDIT_RUN_SHA256 = "fedd0ab2d2e51cf3e22c6c4eb453cd0629e4fb6fc5fc29db03b677070109951f"
KNOWN_AUDIT_REPORT_SHA256 = "282f540880ca111cae2723c971951f7e577d67858ce6762beea1301f49b9666e"


@dataclass(frozen=True)
class Admission:
    dataset: object
    manifest: dict
    cache_paths: dict
    feature_pipeline: dict
    immutable: dict
    watched: dict
    class_counts_train_only: list


def _known_json(path, expected):
    if Path(path).is_symlink() or audit.sha(path) != expected:
        raise ValueError("Completed cache-audit witness differs from its independently known SHA256")
    value = audit._read(path)
    if audit.sha(path) != expected:
        raise ValueError("Completed cache-audit witness changed while reading")
    return value


def _watch(watched, path, expected):
    path = Path(path).resolve()
    if str(path) in watched and watched[str(path)] != expected:
        raise ValueError("Conflicting immutable training-input bindings")
    if audit.sha(path) != expected:
        raise ValueError("An audited input, implementation or selected cache changed: " + str(path))
    watched[str(path)] = expected


def _same(actual, expected):
    # In particular reject booleans where a count, fold or clip ID was declared.
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(_same(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def admit_fold(*, annotations, manifests_dir, audit_run, fold, protocol_path=DEFAULT_PROTOCOL):
    """Bind known witnesses, frozen code/runtime and exactly one regenerated fold.

    Cache bytes are re-hashed before model construction. Cache array parsing and
    per-batch integrity remain the unchanged HeadExperiment driver's job. Raw
    video bytes are not needed for training on already authenticated features.
    This function does not establish remote Git publication or authorize fitting.
    """
    if type(fold) is not int or fold not in (2, 3, 4):
        raise ValueError("Only registered additional official folds 2, 3 and 4 are admitted")
    annotations, manifests_dir, audit_run, protocol_path = (
        Path(p).resolve() for p in (annotations, manifests_dir, audit_run, protocol_path)
    )
    report_path = audit_run.with_name("audit.json")
    state = _known_json(audit_run, KNOWN_AUDIT_RUN_SHA256)
    report = _known_json(report_path, KNOWN_AUDIT_REPORT_SHA256)
    if (not isinstance(state, dict) or not isinstance(report, dict)
            or state.get("state") != "completed"
            or state.get("audit_sha256") != KNOWN_AUDIT_REPORT_SHA256
            or report.get("artifact_type") != "read-only-cache-reuse-audit-not-training-catalog"
            or any(not _same(value.get(k), v) for value in (state, report)
                   for k, v in {"schema_version": 1, "fresh_video_decode_performed": False,
                                **audit.FALSE_GATES}.items())):
        raise ValueError("Cache-audit witness is incomplete or promotes unsupported evidence")
    dataset = load_annotations(annotations)
    protocol = audit._protocol(protocol_path, dataset, manifests_dir)
    if report.get("dataset_sha256") != dataset.sha256 or report.get("protocol_sha256") != audit.sha(protocol_path):
        raise ValueError("Cache audit is bound to different annotations or registration")
    count = len(dataset.records)
    if any(type(value) is not int or value != count for value in (
            state.get("completed_clip_count"), state.get("target_clip_count"), report.get("clip_count"))):
        raise ValueError("Cache-audit completion coverage differs from annotations")
    watched = {}
    _watch(watched, audit_run, KNOWN_AUDIT_RUN_SHA256)
    _watch(watched, report_path, KNOWN_AUDIT_REPORT_SHA256)
    for mapping in (state.get("inputs_sha256"), state.get("implementation_sha256")):
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Cache-audit dependency identities are missing")
        for path, digest in mapping.items():
            _watch(watched, path, digest)
    # Every declared binding is checked, including old configs not used by this
    # particular head; new code may not silently weaken the fixed registration.
    for path, digest in audit._protocol_dependencies(protocol).items():
        if state["inputs_sha256"].get(path) != digest:
            raise ValueError("Cache-audit registration dependency coverage differs")
        _watch(watched, path, digest)
    if state["implementation_sha256"] != audit._source_hashes():
        raise ValueError("Cache-audit implementation differs from the reviewed witness")
    for path in [annotations, protocol_path, *(manifests_dir / f"fold-{f}.json" for f in (1, 2, 3, 4))]:
        if state["inputs_sha256"].get(str(path)) != audit.sha(path):
            raise ValueError("Supplied annotations, registration or manifests differ from audited paths")
    registration = audit._read(audit.ROOT / protocol["feature_registration"]["path"])
    pipeline = registration["feature_pipeline"]
    if (not _same(report.get("feature_pipeline"), pipeline)
            or not _same(state.get("feature_pipeline"), pipeline)):
        raise ValueError("Cache-audit frozen feature pipeline differs from registration")
    runtime = audit._runtime_pipeline(pipeline)  # Inspects code/packages, never builds a model.
    manifest_path = manifests_dir / f"fold-{fold}.json"
    manifest = audit._manifest(dataset, manifests_dir, fold,
                               protocol["split"]["seed"], protocol["split"]["validation_fraction_sources"])
    fold_report = report.get("folds", {}).get(str(fold), {})
    if (fold_report.get("manifest_sha256") != audit.sha(manifest_path)
            or not _same(fold_report.get("partitions"), manifest["partitions"])):
        raise ValueError("Audited fold partitions differ from regenerated source-grouped manifest")
    clips = report.get("clips")
    records = {r.clip_id: r for r in dataset.records}
    if not isinstance(clips, dict) or set(clips) != {str(i) for i in records}:
        raise ValueError("Audited clip mapping coverage differs from annotations")
    selected_ids = set(manifest["partitions"]["train"]["clip_ids"]) | set(
        manifest["partitions"]["validation"]["clip_ids"])
    train_ids = set(manifest["partitions"]["train"]["clip_ids"])
    old_catalogs = {}
    for binding in protocol["cache_catalogs"].values():
        catalog = audit._read(audit.ROOT / binding["path"])
        for key, row in catalog["completed"].items():
            if key in old_catalogs:
                raise ValueError("Historical cache catalogs overlap")
            old_catalogs[key] = (row, catalog["config"])
    if set(old_catalogs) != set(clips):
        raise ValueError("Audit and historical catalog cache coverage differ")
    paths = {}
    counts = np.zeros(9, dtype=np.int64)
    total_frames = 0
    for key, clip in clips.items():
        clip_id = int(key)
        record = records[clip_id]
        row, old_config = old_catalogs[key]
        expected_metadata = {"clip_id": clip_id, "source_id": record.source_id,
                             "official_outer_split": record.outer_split, "frames": record.expected_frame_count,
                             "cache_key": row["cache_key"], "cache_sha256": row["cache_sha256"],
                             "cache_path": str(Path(row["cache_path"]).resolve()),
                             "prior_catalog_partition": row["partition"]}
        if (not isinstance(clip, dict) or any(not _same(clip.get(k), v) for k, v in expected_metadata.items())
                or Path(clip["cache_path"]).is_symlink()
                or Path(clip["cache_path"]).resolve() != Path(old_config["cache_dir"]).resolve() / (
                    clip["cache_key"] + ".npz")):
            raise ValueError("Audited cache/source mapping differs from historical catalog or annotations")
        total_frames += record.expected_frame_count
        if clip_id in selected_ids:
            path = Path(clip["cache_path"]).resolve()
            _watch(watched, path, clip["cache_sha256"])
            paths[clip_id] = path
        if clip_id in train_ids:
            labels = np.full(record.expected_frame_count, 8, np.int64)
            labels[list(record.event_indices)] = np.arange(8)
            counts += np.bincount(labels, minlength=9)
    if (not _same(report.get("frame_count"), total_frames) or not _same(state.get("frame_count"), total_frames)
            or not _same(fold_report.get("class_counts_train_only"), counts.tolist())):
        raise ValueError("Audited frame totals or train-only class counts differ from annotations")
    _watch(watched, __file__, audit.sha(__file__))
    immutable = {
        "schema_version": 1, "fold": fold, "dataset_sha256": dataset.sha256,
        "protocol_sha256": audit.sha(protocol_path), "audit_run_sha256": KNOWN_AUDIT_RUN_SHA256,
        "audit_report_sha256": KNOWN_AUDIT_REPORT_SHA256,
        "manifest_sha256": audit.sha(manifest_path), "feature_pipeline": pipeline,
        "verification_runtime": runtime, "watched_sha256": audit.json_hash(watched),
        "class_counts_train_only": counts.tolist(), "outer_test_evaluated": False,
        "validated_for_serving": False,
    }
    result = Admission(dataset, manifest, paths, pipeline, immutable, watched, counts.tolist())
    recheck(result)
    return result


def recheck(admission):
    """Refuse changed inputs/selected cache bytes before checkpoint publication."""
    if audit.json_hash(admission.watched) != admission.immutable["watched_sha256"]:
        raise ValueError("Training admission dependency bindings changed")
    for path, digest in admission.watched.items():
        if Path(path).is_symlink() or audit.sha(path) != digest:
            raise ValueError("An admitted training input or implementation changed: " + path)
