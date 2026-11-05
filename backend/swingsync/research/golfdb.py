"""Validated GolfDB annotations and source-grouped, reproducible split manifests.

Schema and index conventions audited against wmcnally/golfdb revision
 a63b4ce8c0900d09abbe92519223efe16810f5f4. This module does not load pickle,
 acquire videos, run models or establish image-to-label alignment.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy.io import loadmat

UPSTREAM_REVISION = "a63b4ce8c0900d09abbe92519223efe16810f5f4"
FIELDS = {"id", "youtube_id", "player", "sex", "club", "view", "slow", "events", "bbox", "split"}


def _integer(value, name: str) -> int:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite integer")
    result = int(array.item())
    if array.item() != result:
        raise ValueError(f"{name} must be an integer")
    return result


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class GolfDBRecord:
    clip_id: int
    source_id: str
    player: str
    sex: str
    club: str
    view: str
    slow: bool
    original_events: tuple[int, ...]
    bbox: tuple[float, ...]
    outer_split: int

    @property
    def trim_start(self) -> int:
        return self.original_events[0]

    @property
    def trim_end(self) -> int:
        return self.original_events[-1]

    @property
    def event_indices(self) -> tuple[int, ...]:
        """Reference-loader labels in the preprocessed clip, without mutation."""
        return tuple(event - self.trim_start for event in self.original_events[1:-1])

    @property
    def event_order_valid(self) -> bool:
        """Audit flag only: preserve official anomalies rather than repair labels."""
        return all(a < b for a, b in zip(self.original_events[1:-2], self.original_events[2:-1]))

    @property
    def expected_frame_count(self) -> int:
        """Historical inclusive preprocessing count, assuming source is long enough."""
        return self.trim_end - max(1, self.trim_start) + 1

    def source_frame_index(self, local_index: int) -> int:
        """Historical preprocessing's zero-based source index, not a timestamp.

        Its counter starts at one; a zero trim boundary still begins at source
        index zero. Original source availability/alignment must be checked later.
        """
        if isinstance(local_index, bool) or not isinstance(local_index, Integral):
            raise ValueError("local_index must be an integer")
        if not 0 <= local_index < self.expected_frame_count:
            raise ValueError("local_index is outside the historical trimmed clip")
        return max(1, self.trim_start) - 1 + int(local_index)


@dataclass(frozen=True)
class GolfDBAnnotations:
    records: tuple[GolfDBRecord, ...]
    sha256: str


def load_annotations(path: Path | str) -> GolfDBAnnotations:
    """Read the official structured MAT schema; reject ambiguous/broken labels.

    The fingerprint identifies exact bytes, not a guessed dataset version.
    Frozen tuples prevent an in-place offset subtraction corrupting future reads.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    # Parse the exact bytes hashed, even if another process replaces the source.
    from io import BytesIO
    raw = loadmat(BytesIO(raw_bytes), squeeze_me=True, struct_as_record=False)
    if "golfDB" not in raw:
        raise ValueError("MAT file is missing golfDB")
    rows = np.atleast_1d(raw["golfDB"]).reshape(-1)
    if not len(rows):
        raise ValueError("golfDB contains no records")
    records = []
    ids = set()
    source_folds = {}
    for row_number, row in enumerate(rows):
        if set(getattr(row, "_fieldnames", [])) != FIELDS:
            raise ValueError(f"Row {row_number}: unexpected GolfDB annotation schema")
        clip_id = _integer(row.id, "id")
        if clip_id < 0 or clip_id in ids:
            raise ValueError("Clip IDs must be unique nonnegative integers")
        ids.add(clip_id)
        source = _text(row.youtube_id, "youtube_id")
        fold = _integer(row.split, "split")
        if fold not in {1, 2, 3, 4}:
            raise ValueError("Official split must be 1, 2, 3 or 4")
        if source in source_folds and source_folds[source] != fold:
            raise ValueError(f"Source {source} crosses official outer splits")
        source_folds[source] = fold
        events_array = np.asarray(row.events).reshape(-1)
        if len(events_array) != 10:
            raise ValueError("Events must contain trim start, eight targets and trim end")
        events = tuple(_integer(e, "event") for e in events_array)
        if events[0] < 0 or events[-1] <= events[0] or any(
            event < events[0] or event > events[-1] for event in events[1:-1]
        ):
            raise ValueError("Target events must lie within nonnegative ordered trim boundaries")
        if len(set(events[1:-1])) != 8:
            raise ValueError("Target events must occupy eight distinct frames")
        frame_count = events[-1] - max(1, events[0]) + 1
        if frame_count < 8 or any(e - events[0] >= frame_count for e in events[1:-1]):
            raise ValueError("Event label falls outside historical preprocessed clip bounds")
        bbox_array = np.asarray(row.bbox, dtype=float).reshape(-1)
        if bbox_array.shape != (4,) or not np.isfinite(bbox_array).all():
            raise ValueError("bbox must have four finite normalized x,y,width,height values")
        if np.any(bbox_array < 0) or np.any(bbox_array > 1) or np.any(bbox_array[2:] <= 0):
            raise ValueError("bbox coordinates must be normalized with positive width/height")
        slow = _integer(row.slow, "slow")
        if slow not in {0, 1}:
            raise ValueError("slow must be zero or one")
        records.append(GolfDBRecord(
            clip_id, source, _text(row.player, "player"), _text(row.sex, "sex"),
            _text(row.club, "club"), _text(row.view, "view"), bool(slow), events,
            tuple(float(v) for v in bbox_array), fold,
        ))
    return GolfDBAnnotations(tuple(sorted(records, key=lambda r: r.clip_id)), hashlib.sha256(raw_bytes).hexdigest())


def make_split_manifest(dataset: GolfDBAnnotations, outer_fold: int, seed: int = 42,
                        validation_fraction: float = 0.2) -> dict:
    """Hold out official fold; assign whole candidate source groups internally.

    SHA256 ranking of seed/source is deterministic across NumPy/Python versions.
    Fraction controls number of sources, not frame or clip count. Neither event
    timing nor test labels are consulted to choose internal validation sources.
    """
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, Integral) or outer_fold not in {1, 2, 3, 4}:
        raise ValueError("outer_fold must be 1, 2, 3 or 4")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if not dataset.records:
        raise ValueError("Cannot split empty annotations")
    records = sorted(dataset.records, key=lambda r: r.clip_id)
    if len({r.clip_id for r in records}) != len(records):
        raise ValueError("Clip IDs must be unique")
    test = [r for r in records if r.outer_split == outer_fold]
    candidates = [r for r in records if r.outer_split != outer_fold]
    sources = sorted({r.source_id for r in candidates})
    if not test or len(sources) < 2:
        raise ValueError("Need a nonempty official test fold and at least two training sources")
    if {r.source_id for r in test} & set(sources):
        raise ValueError("Source recordings cross official outer split")
    ranked = sorted(sources, key=lambda source: hashlib.sha256(
        json.dumps([int(seed), source], ensure_ascii=True, separators=(",", ":")).encode()).hexdigest())
    n_val = min(len(sources) - 1, max(1, math.ceil(len(sources) * validation_fraction)))
    validation_sources = set(ranked[:n_val])
    partitions = {
        "train": [r for r in candidates if r.source_id not in validation_sources],
        "validation": [r for r in candidates if r.source_id in validation_sources], "test": test,
    }
    return {
        "schema_version": 1, "dataset_sha256": dataset.sha256,
        "upstream_revision": UPSTREAM_REVISION, "outer_fold": int(outer_fold), "seed": int(seed),
        "validation_fraction_sources": validation_fraction,
        "split_algorithm": "sha256-seed-source-rank-v1-ceil",
        "partitions": {name: {
            "clip_ids": [r.clip_id for r in rows], "source_ids": sorted({r.source_id for r in rows}),
            "clips": [{"clip_id": r.clip_id, "source_id": r.source_id} for r in rows],
        } for name, rows in partitions.items()},
        "evidence_scope": "Annotation partition only; no trained model or verified media alignment",
    }


def write_split_manifest(manifest: dict, path: Path | str) -> None:
    """Write stable reviewable JSON; do not overwrite an existing experiment file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
