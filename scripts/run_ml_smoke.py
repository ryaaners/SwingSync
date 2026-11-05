"""Local CPU pipeline smoke on real training clips, never a benchmark result."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import torchvision

from backend.swingsync.research.cache_job import DEFAULT_LOCK_PATH
from backend.swingsync.research.features import (
    build_cache_identity, cache_key, extract_clip_features, load_feature_cache, save_feature_cache,
)
from backend.swingsync.research.golfdb import load_annotations, make_split_manifest
from backend.swingsync.research.models import (
    FrameClassifier, FrozenMobileNetV2, TemporalClassifier, class_weights_from_counts, detect_devices,
)
from backend.swingsync.research.training import (
    collate_features, restore_checkpoint, save_checkpoint, seed_everything, train_steps,
)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args):
    run_dir = args.run_dir.resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_LOCK_PATH.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another training/smoke job owns this artifact-root lock') from exc
        if run_dir.exists():
            raise ValueError('Run directory exists; inspect its state rather than launching duplicate work')
        dataset = load_annotations(args.annotations)
        manifest = json.loads(args.manifest.read_text())
        expected = make_split_manifest(dataset, manifest['outer_fold'], manifest['seed'],
                                       manifest['validation_fraction_sources'])
        if manifest != expected:
            raise ValueError('Split manifest does not match audited dataset/grouped partition algorithm')
        ids = manifest['partitions']['train']['clip_ids'][:args.clips]
        if not 1 <= args.clips <= 4 or len(ids) != args.clips:
            raise ValueError('Smoke requires one to four available training clips')
        records = {r.clip_id: r for r in dataset.records}
        run_dir.mkdir()
        source_files = [Path(__file__), Path('pyproject.toml'), Path('uv.lock')]
        source_files += sorted(Path('backend/swingsync/research').glob('*.py'))
        state = {
            'evidence_type': 'real-training-partition-pipeline-smoke-not-benchmark',
            'state': 'running', 'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
            'process_start_identity': subprocess.check_output(
                ['ps', '-p', str(os.getpid()), '-o', 'lstart='], text=True).strip(),
            'config': {'seed': args.seed, 'device': 'cpu', 'threads': 1, 'steps_per_model': 5,
                       'checkpoint_step': 2, 'learning_rate': .001, 'clips': ids,
                       'batch_size': 16, 'hidden_size': 64, 'augmentation': 'none'},
            'dataset_sha256': dataset.sha256, 'manifest_sha256': digest(args.manifest),
            'git_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'code_sha256': {str(p): digest(p) for p in source_files},
            'hardware': {'platform': platform.platform(), 'machine': platform.machine(),
                         'processor': platform.processor()},
            'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': np.__version__,
            'devices': detect_devices(), 'models': {},
            'outer_test_evaluated': False, 'validated_for_serving': False,
        }
        write_json(run_dir / 'run.json', state)  # Durable identity/config BEFORE feature extraction/training.
        started = time.monotonic()
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            seed_everything(args.seed)
            os.environ.setdefault('TORCH_HOME', str(Path('data/torch').resolve()))
            backbone = FrozenMobileNetV2(weights='IMAGENET1K_V2')
            state['backbone'] = backbone.provenance
            clips, identities = [], []
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            feature_started = time.monotonic()
            for clip_id in ids:
                path = args.clips_dir / f'{clip_id}.mp4'
                identity = build_cache_identity(path, records[clip_id], dataset.sha256, backbone)
                cache = args.cache_dir / f'{cache_key(identity)}.npz'
                if cache.exists():
                    clip = load_feature_cache(cache, expected_identity=identity)
                else:
                    clip = extract_clip_features(path, records[clip_id], annotation_sha256=dataset.sha256,
                                                 backbone=backbone, batch_size=16, device='cpu')
                    save_feature_cache(clip, cache)
                clips.append(clip)
                identities.append({'clip_id': clip_id, 'cache_key': cache_key(identity),
                                   'cache_sha256': digest(cache), 'cache_path': str(cache.resolve())})
            state['feature_cache'] = identities
            state['feature_extraction_or_cache_load_seconds'] = time.monotonic() - feature_started
            batch = collate_features(clips)
            counts = np.bincount(np.concatenate([c.labels for c in clips]), minlength=9)
            weights = class_weights_from_counts(torch.tensor(counts))
            state['training_partition_smoke_class_counts'] = counts.tolist()
            state['class_weights'] = weights.tolist()
            state['frames'] = int(sum(len(c.labels) for c in clips))
            write_json(run_dir / 'run.json', state)
            for name, constructor in [('frame', FrameClassifier), ('temporal', TemporalClassifier)]:
                seed_everything(args.seed)
                model = constructor()
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                provenance = {'model': name, 'config': state['config'], 'dataset': dataset.sha256,
                              'manifest': state['manifest_sha256'], 'caches': identities,
                              'code_sha256': state['code_sha256']}
                model_started = time.monotonic()
                first_losses = train_steps(model, optimizer, batch, weights, 2)
                initial_hash = save_checkpoint(run_dir / f'{name}-step2.pt', model, optimizer,
                                               step=2, provenance=provenance)
                final_losses = train_steps(model, optimizer, batch, weights, 3)
                resumed = constructor()
                resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.001)
                step = restore_checkpoint(run_dir / f'{name}-step2.pt', resumed, resumed_optimizer,
                                          expected_provenance=provenance)
                resumed_losses = train_steps(resumed, resumed_optimizer, batch, weights, 3)
                parity = step == 2 and final_losses == resumed_losses and all(
                    torch.equal(v, resumed.state_dict()[k]) for k, v in model.state_dict().items())
                if not parity:
                    raise ValueError(f'{name} checkpoint resume does not exactly match CPU continuation')
                final_hash = save_checkpoint(run_dir / f'{name}-step5.pt', resumed, resumed_optimizer,
                                             step=5, provenance=provenance)
                state['models'][name] = {'completed_optimizer_steps': 5,
                                         'training_loss': first_losses + final_losses,
                                         'exact_cpu_resume_verified': True,
                                         'step2_sha256': initial_hash, 'step5_sha256': final_hash,
                                         'training_and_resume_check_seconds': time.monotonic() - model_started}
                write_json(run_dir / 'run.json', state)
            state.update(state='completed', elapsed_seconds=time.monotonic() - started,
                         completed_at=datetime.now(timezone.utc).isoformat())
            write_json(run_dir / 'run.json', state)
        except BaseException as exc:
            state.update(state='failed', error=str(exc), elapsed_seconds=time.monotonic() - started)
            write_json(run_dir / 'run.json', state)
            raise
        return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--clips-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, default=Path('artifacts/features'))
    p.add_argument('--clips', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261105)
    args = p.parse_args()
    state = run(args)
    print(json.dumps({'run_dir': str(args.run_dir), 'state': state['state'], 'frames': state['frames'],
                      'evidence_type': state['evidence_type'], 'models': state['models']}, indent=2))


if __name__ == '__main__':
    main()
