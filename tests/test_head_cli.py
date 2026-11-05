"""Synthetic schema/tensor fixtures test process/checkpoint wiring, not ML results.

Expected pipeline metadata is a mocked contract in these fixtures; features are
random software tensors and are never presented as pretrained extraction.
"""
from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from test_experiment import inputs as experiment_inputs  # noqa: E402,F401
from backend.swingsync.research.features import cache_key  # noqa: E402
from scripts import train_event_head  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def cli_inputs(tmp_path, monkeypatch, experiment_inputs):  # noqa: F811 - pytest fixture injection
    dataset, manifest, paths = experiment_inputs
    monkeypatch.setattr(train_event_head, 'load_annotations', lambda _: dataset)
    monkeypatch.setattr(train_event_head, '_git_base_commit', lambda: 'synthetic-software-revision')
    monkeypatch.setattr(train_event_head, 'DEFAULT_LOCK_PATH', tmp_path / 'shared' / 'lock')
    model = {'schema_version': 1, 'seed': 7, 'model': 'frame', 'epochs': 2, 'batch_size': 3,
             'learning_rate': .001, 'hidden_size': 4, 'selection_metric': 'validation_pce',
             'context': 'full-clip', 'augmentation': 'none', 'device': 'cpu'}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(model))
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    pipeline = deepcopy(json.loads(Path('experiments/event-v1.json').read_text())['feature_pipeline'])
    for path in paths.values():
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        meta = json.loads(arrays['metadata'].item())
        meta['identity'].update(pipeline)
        meta['cache_key'] = cache_key(meta['identity'])
        arrays['metadata'] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
    registration_path = tmp_path / 'registration.json'
    registration_path.write_text(json.dumps({'config_sha256': {'frame': sha(config_path)},
                                             'dataset_sha256': dataset.sha256,
                                             'manifest_sha256': sha(manifest_path),
                                             'feature_pipeline': pipeline}))
    job = {'state': 'completed', 'outer_test_selected': False, 'model_identity': pipeline,
           'config': {'partitions': ['train', 'validation'], 'manifest_sha256': sha(manifest_path),
                      'annotations_sha256': dataset.sha256, 'weights_request': 'IMAGENET1K_V2'},
           'completed': {str(i): {'cache_path': str(path), 'cache_sha256': sha(path)} for i, path in paths.items()},
           'completed_clip_count': len(paths), 'target_clip_count': len(paths)}
    job_path = tmp_path / 'job.json'
    job_path.write_text(json.dumps(job))
    args = Namespace(config=config_path, annotations=tmp_path / 'annotations.mat', manifest=manifest_path,
                     registration=registration_path, cache_job=job_path, run_dir=tmp_path / 'run',
                     max_batches=1, checkpoint_batches=1, resume=False)
    return args, job


def test_cli_partial_resume_and_orphan_checkpoint_recovery(cli_inputs):
    args, _ = cli_inputs
    first = train_event_head.run(args)
    assert first['state'] == 'partial' and first['summary']['optimizer_steps'] == 1
    assert first['git_base_commit'] == 'synthetic-software-revision'
    assert first['runtime']['torch_num_threads'] == 1
    checkpoint = deepcopy(first['checkpoints'][0])
    # Simulate a crash after checkpoint publish but before its JSON pointer.
    first['checkpoints'] = []
    (args.run_dir / 'run.json').write_text(json.dumps(first))
    args.resume = True
    second = train_event_head.run(args)
    assert second['summary']['optimizer_steps'] == 2
    assert second['checkpoints'][0]['recovered'] is True
    assert second['checkpoints'][0]['sha256'] == checkpoint['sha256']
    assert second['summary']['outer_test_evaluated'] is False
    assert len(second['attempts']) == 2
    args.resume = False
    with pytest.raises(ValueError, match='Run directory exists'):
        train_event_head.run(args)


@pytest.mark.parametrize('problem', ['partial', 'missing_pipeline', 'wrong_weights', 'wrong_transform'])
def test_cli_rejects_incomplete_or_unregistered_pipeline_before_run_creation(cli_inputs, problem):
    args, job = cli_inputs
    if problem == 'partial':
        job['state'] = 'partial'
    elif problem == 'missing_pipeline':
        del job['model_identity']
    elif problem == 'wrong_weights':
        job['config']['weights_request'] = 'injected-backbone'
    else:
        job['model_identity']['transform']['center_crop'] = 160
    args.cache_job.write_text(json.dumps(job))
    with pytest.raises(ValueError, match='requires a complete|registered frozen'):
        train_event_head.run(args)
    assert not args.run_dir.exists()
