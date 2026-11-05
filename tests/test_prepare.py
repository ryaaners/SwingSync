"""End-to-end preparation artifacts use synthetic schema, not model predictions."""
import json
from zipfile import ZipFile

import pytest
from test_golfdb import write_mat

from scripts.golfdb_prepare import prepare


def test_prepare_writes_all_folds_and_refuses_overwrite(tmp_path):
    annotations = write_mat(tmp_path / 'fixture.mat')
    output = tmp_path / 'run'
    result = prepare(annotations, output)
    assert result['annotations'] == 16
    assert result['source_recordings'] == 8
    assert set(result['manifest_sha256']) == {f'fold-{i}.json' for i in range(1, 5)}
    assert not result['original_source_image_alignment_verified']
    assert result['evidence_type'] == 'data-preparation-not-model-performance'
    assert json.loads((output / 'summary.json').read_text()) == result
    original = (output / 'summary.json').read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        prepare(annotations, output)
    assert (output / 'summary.json').read_bytes() == original
    second = prepare(annotations, tmp_path / 'second')
    assert second == result


def test_archive_inventory_and_no_partial_output_on_bad_inputs(tmp_path):
    annotations = write_mat(tmp_path / 'fixture.mat')
    archive = tmp_path / 'fixture.zip'
    with ZipFile(archive, 'w') as z:
        for i in range(16):
            z.writestr(f'videos_160/{i}.mp4', b'software inventory only, not decoded video')
    result = prepare(annotations, tmp_path / 'good', archive=archive)
    assert result['archive']['clips'] == 16
    with ZipFile(archive, 'a') as z:
        z.writestr('../unexpected.mp4', b'not extracted')
    with pytest.raises(ValueError, match='exactly one'):
        prepare(annotations, tmp_path / 'bad', archive=archive)
    assert not (tmp_path / 'bad').exists()
    with pytest.raises(ValueError, match='requires clips_dir'):
        prepare(annotations, tmp_path / 'no-clips', clip_ids=[0])
    assert not (tmp_path / 'no-clips').exists()


def test_all_clips_requires_directory_and_exclusive_selection(tmp_path):
    annotations = write_mat(tmp_path / 'fixture.mat')
    with pytest.raises(ValueError, match='requires clips_dir'):
        prepare(annotations, tmp_path / 'missing', all_clips=True)
    with pytest.raises(ValueError, match='not both'):
        prepare(annotations, tmp_path / 'both', all_clips=True, clip_ids=[0])
    assert not (tmp_path / 'missing').exists()
    assert not (tmp_path / 'both').exists()
