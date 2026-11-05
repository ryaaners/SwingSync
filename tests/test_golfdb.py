"""Synthetic annotation contracts, not model-performance or media-alignment evidence."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import numpy as np
import pytest
from scipy.io import savemat

from backend.swingsync.research.golfdb import (
    FIELDS, GolfDBAnnotations, load_annotations, make_split_manifest, write_split_manifest,
)


def write_mat(path, edits=None):
    """Minimal official structured schema with source-repeated clips in each fold."""
    rows = np.empty((1, 16), dtype=[(key, "O") for key in sorted(FIELDS)])
    for i in range(16):
        values = {
            "id": i, "youtube_id": f"source-{i // 2}", "player": "test-player",
            "sex": "test-sex", "club": "iron", "view": "face-on", "slow": 0,
            "events": np.array([408, 455, 473, 476, 490, 495, 498, 501, 514, 545]),
            "bbox": np.array([0.1, 0.1, 0.8, 0.8]), "split": i // 4 + 1,
        }
        if edits:
            values.update(edits(i))
        for key, value in values.items():
            rows[key][0, i] = value
    savemat(path, {"golfDB": rows})
    return path


def test_reference_trim_indices_and_immutability(tmp_path):
    path = write_mat(tmp_path / "fixture.mat")
    original_bytes = path.read_bytes()
    dataset = load_annotations(path)
    record = dataset.records[0]
    assert record.event_indices == (47, 65, 68, 82, 87, 90, 93, 106)
    assert record.original_events == (408, 455, 473, 476, 490, 495, 498, 501, 514, 545)
    assert record.expected_frame_count == 138
    assert record.source_frame_index(0) == 407
    assert record.source_frame_index(137) == 544
    assert record.event_order_valid
    for bad in (-1, 138, True, 0.5):
        with pytest.raises(ValueError):
            record.source_frame_index(bad)
    with pytest.raises(FrozenInstanceError):
        record.original_events = ()
    assert load_annotations(path) == dataset
    assert path.read_bytes() == original_bytes
    assert dataset.sha256 == hashlib.sha256(original_bytes).hexdigest()


def test_zero_trim_boundary_reproduces_counter_start(tmp_path):
    path = write_mat(tmp_path / "fixture.mat", lambda i: {"events": np.arange(10)})
    record = load_annotations(path).records[0]
    assert record.expected_frame_count == 9
    assert record.source_frame_index(0) == 0
    assert record.source_frame_index(8) == 8
    assert record.event_indices == tuple(range(1, 9))


def test_preserve_official_nonordered_event_anomaly(tmp_path):
    events = np.array([794, 886, 908, 925, 940, 952, 962, 961, 1036, 1071])
    path = write_mat(tmp_path / "fixture.mat", lambda i: {"events": events})
    record = load_annotations(path).records[0]
    assert not record.event_order_valid
    assert record.original_events == tuple(events)
    assert record.event_indices[5:7] == (168, 167)


@pytest.mark.parametrize("change,match", [
    ({"events": np.arange(9)}, "Events must contain"),
    ({"events": [0, 1, 2, 3, 4, 5, 6, 6, 8, 9]}, "distinct"),
    ({"events": [0, 1, 2, 3, 4, 5, 6, 7, 99, 9]}, "trim boundaries"),
    ({"events": [0, 1, 2, 3, 4, 5, 6, 7, 8.5, 9]}, "integer"),
    ({"bbox": [0, 0, 0, 1]}, "positive"),
    ({"bbox": [0, 0, float("nan"), 1]}, "finite"),
    ({"split": 5}, "Official split"),
    ({"slow": 2}, "slow"),
    ({"id": -1}, "nonnegative"),
    ({"youtube_id": " "}, "nonempty"),
])
def test_reject_malformed_annotations(tmp_path, change, match):
    path = write_mat(tmp_path / "fixture.mat", lambda i: change if i == 0 else {})
    with pytest.raises(ValueError, match=match):
        load_annotations(path)


def test_reject_duplicate_id_and_outer_source_leakage(tmp_path):
    path = write_mat(tmp_path / "duplicate.mat", lambda i: {"id": 0} if i == 1 else {})
    with pytest.raises(ValueError, match="unique"):
        load_annotations(path)
    path = write_mat(tmp_path / "leaked.mat", lambda i: {"youtube_id": "source-0"} if i == 4 else {})
    with pytest.raises(ValueError, match="crosses official"):
        load_annotations(path)
    savemat(tmp_path / "wrong.mat", {"notGolfDB": [1]})
    with pytest.raises(ValueError, match="missing golfDB"):
        load_annotations(tmp_path / "wrong.mat")


def test_saved_manifests_preserve_outer_folds_and_source_groups(tmp_path):
    dataset = load_annotations(write_mat(tmp_path / "fixture.mat"))
    for fold in range(1, 5):
        manifest = make_split_manifest(dataset, fold, seed=17, validation_fraction=0.25)
        reverse = GolfDBAnnotations(tuple(reversed(dataset.records)), dataset.sha256)
        assert make_split_manifest(reverse, fold, seed=17, validation_fraction=0.25) == manifest
        assert manifest["dataset_sha256"] == dataset.sha256
        partitions = manifest["partitions"]
        assert set(partitions["test"]["clip_ids"]) == {r.clip_id for r in dataset.records if r.outer_split == fold}
        a, b, c = (set(partitions[name]["source_ids"]) for name in ["train", "validation", "test"])
        assert not (a & b or a & c or b & c)
        ids = [clip_id for part in partitions.values() for clip_id in part["clip_ids"]]
        assert sorted(ids) == sorted(r.clip_id for r in dataset.records)
        assert len(ids) == len(set(ids))
        for part in partitions.values():
            for source in part["source_ids"]:
                assert {r.clip_id for r in dataset.records if r.source_id == source} <= set(part["clip_ids"])
        output = tmp_path / f"fold-{fold}.json"
        write_split_manifest(manifest, output)
        assert json.loads(output.read_text()) == manifest
        with pytest.raises(FileExistsError):
            write_split_manifest(manifest, output)


def test_split_rejects_bad_parameters_and_insufficient_groups(tmp_path):
    dataset = load_annotations(write_mat(tmp_path / "fixture.mat"))
    for bad in (0, 5, True, 1.0):
        with pytest.raises(ValueError):
            make_split_manifest(dataset, bad)
    for bad in (0, 1, -0.1, float("nan")):
        with pytest.raises(ValueError):
            make_split_manifest(dataset, 1, validation_fraction=bad)
    with pytest.raises(ValueError, match="seed"):
        make_split_manifest(dataset, 1, seed=True)
    small = replace(dataset, records=dataset.records[:4])
    with pytest.raises(ValueError, match="at least two"):
        make_split_manifest(small, 1)
