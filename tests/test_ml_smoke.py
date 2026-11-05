"""Smoke runner preflight rejects split tampering and duplicate job state offline."""
from argparse import Namespace
import fcntl
import json

import pytest
from test_golfdb import write_mat

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from backend.swingsync.research.golfdb import load_annotations, make_split_manifest  # noqa: E402
from scripts.run_ml_smoke import run  # noqa: E402


def args_for(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.run_ml_smoke.DEFAULT_LOCK_PATH', tmp_path / 'shared' / 'active-training.lock')
    annotations = write_mat(tmp_path / 'fixture.mat')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(make_split_manifest(load_annotations(annotations), 1)))
    return Namespace(annotations=annotations, manifest=manifest, run_dir=tmp_path / 'runs' / 'smoke',
                     clips=2, seed=7, clips_dir=tmp_path / 'clips', cache_dir=tmp_path / 'cache')


def test_manifest_tampering_rejected_before_training(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    data = json.loads(args.manifest.read_text())
    data['partitions']['train']['clip_ids'][0] = data['partitions']['test']['clip_ids'][0]
    args.manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='manifest does not match'):
        run(args)
    assert not args.run_dir.exists()


def test_existing_state_and_active_lock_block_duplicate_smoke(tmp_path, monkeypatch):
    args = args_for(tmp_path, monkeypatch)
    args.run_dir.mkdir(parents=True)
    (args.run_dir / 'run.json').write_text('{"state":"running"}')
    with pytest.raises(ValueError, match='Run directory exists'):
        run(args)
    assert (args.run_dir / 'run.json').read_text() == '{"state":"running"}'
    with (tmp_path / 'shared' / 'active-training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='Another training'):
            run(args)
