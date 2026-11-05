"""Resume one preregistered CPU head experiment; no outer-test evaluation."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

from backend.swingsync.research.cache_job import DEFAULT_LOCK_PATH, _publish_json, _process_start, _git_base_commit
from backend.swingsync.research.experiment import ExperimentConfig, HeadExperiment
from backend.swingsync.research.features import DEFAULT_TRANSFORM
from backend.swingsync.research.golfdb import load_annotations


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def run(args):
    DEFAULT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_LOCK_PATH.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another feature/training job owns the shared lock') from exc
        if args.max_batches is not None and args.max_batches < 1:
            raise ValueError('max_batches must be positive')
        if args.checkpoint_batches < 1:
            raise ValueError('checkpoint_batches must be positive')
        config = ExperimentConfig.from_dict(json.loads(args.config.read_text()))
        registration = json.loads(args.registration.read_text())
        if registration['config_sha256'][config.model] != digest(args.config):
            raise ValueError('Config differs from the registered comparison')
        dataset = load_annotations(args.annotations)
        if dataset.sha256 != registration['dataset_sha256'] or digest(args.manifest) != registration['manifest_sha256']:
            raise ValueError('Data/split differs from the registered comparison')
        manifest = json.loads(args.manifest.read_text())
        job = json.loads(args.cache_job.read_text())
        expected_ids = set(manifest['partitions']['train']['clip_ids'] + manifest['partitions']['validation']['clip_ids'])
        if (job.get('state') != 'completed' or job.get('outer_test_selected') is not False
                or job.get('config', {}).get('partitions') != ['train', 'validation']
                or job['config'].get('manifest_sha256') != digest(args.manifest)
                or job['config'].get('annotations_sha256') != dataset.sha256
                or set(job.get('completed', {})) != {str(i) for i in expected_ids}
                or job.get('completed_clip_count') != len(expected_ids)
                or job.get('target_clip_count') != len(expected_ids)):
            raise ValueError('Head training requires a complete matching train+validation cache job')
        expected_pipeline = registration.get('feature_pipeline')
        if (not isinstance(expected_pipeline, dict)
                or expected_pipeline.get('backbone_class') != 'backend.swingsync.research.models.FrozenMobileNetV2'
                or expected_pipeline.get('backbone_provenance', {}).get('weights') != 'IMAGENET1K_V2'
                or expected_pipeline.get('transform') != DEFAULT_TRANSFORM
                or expected_pipeline.get('extraction_device') != 'cpu'
                or job['config'].get('weights_request') != 'IMAGENET1K_V2'
                or job.get('model_identity') != expected_pipeline):
            raise ValueError('Cache job differs from registered frozen ImageNet feature pipeline')
        paths = {}
        for key, row in job['completed'].items():
            if digest(row['cache_path']) != row['cache_sha256']:
                raise ValueError('A completed cache differs from its job ledger')
            paths[int(key)] = Path(row['cache_path'])
        immutable = {'config_sha256': digest(args.config), 'registration_sha256': digest(args.registration),
                     'cache_job_sha256': digest(args.cache_job), 'manifest_sha256': digest(args.manifest),
                     'dependency_lock_sha256': digest(Path(__file__).resolve().parents[1] / 'uv.lock'),
                     'cli_sha256': digest(__file__)}
        root = args.run_dir.resolve()
        state_path = root / 'run.json'
        if root.exists():
            if not args.resume:
                raise ValueError('Run directory exists; inspect state and explicitly resume')
            state = json.loads(state_path.read_text())
            if state['immutable'] != immutable:
                raise ValueError('Run provenance changed; do not silently restart or retune')
        else:
            if args.resume:
                raise ValueError('Run directory does not exist')
            root.mkdir(parents=True)
            state = {'schema_version': 1, 'created_at': now(), 'immutable': immutable,
                     'config': config.to_dict(), 'attempts': [], 'checkpoints': [],
                     'git_base_commit': _git_base_commit(),
                     'evidence_type': 'preregistered-inner-validation-head-training',
                     'outer_test_evaluated': False, 'validated_for_serving': False}
        attempt = {'pid': os.getpid(), 'process_start_identity': _process_start(os.getpid()), 'started_at': now()}
        state['attempts'].append(attempt)
        started = time.monotonic()
        state.update(state='running', pid=os.getpid(), process_start_identity=attempt['process_start_identity'], error=None)
        _publish_json(state_path, state)  # Process identity persists before model construction/training.
        try:
            driver = HeadExperiment(config, dataset, manifest, paths)
            if any(driver.provenance['feature_pipeline'].get(k) != v for k, v in expected_pipeline.items()):
                raise ValueError('Actual cache signature differs from the registered job pipeline')
            state['runtime'] = driver.provenance['runtime']
            if state['checkpoints']:
                last = state['checkpoints'][-1]
                if digest(last['path']) != last['sha256']:
                    raise ValueError('Latest checkpoint differs from recorded bytes')
                driver.restore(last['path'])
            known = {Path(row['path']).resolve() for row in state['checkpoints']}
            orphans = [p for p in root.glob('step-*.pt') if p.resolve() not in known]
            if len(orphans) > 1:
                raise ValueError('Multiple unreferenced checkpoints require inspection')
            if orphans:
                prior_steps = driver.optimizer_steps
                driver.restore(orphans[0])  # Validates exact config/data/runtime before adoption.
                if (driver.optimizer_steps <= prior_steps
                        or orphans[0].name != f'step-{driver.optimizer_steps:07d}.pt'):
                    raise ValueError('Unreferenced checkpoint is inconsistent with durable progress')
                state['checkpoints'].append({'path': str(orphans[0]), 'sha256': digest(orphans[0]),
                                              'optimizer_steps': driver.optimizer_steps, 'recovered': True})
                state.update(summary=driver.summary(), updated_at=now())
                _publish_json(state_path, state)
            start_steps = driver.optimizer_steps
            while driver.summary()['state'] != 'completed':
                remaining = None if args.max_batches is None else args.max_batches - (driver.optimizer_steps - start_steps)
                if remaining is not None and remaining <= 0:
                    break
                budget = args.checkpoint_batches if remaining is None else min(remaining, args.checkpoint_batches)
                driver.advance(max_batches=budget)
                path = root / f'step-{driver.optimizer_steps:07d}.pt'
                if path.exists():
                    raise ValueError('Unreferenced checkpoint exists; inspect crash state before resuming')
                sha = driver.save(path)
                state['checkpoints'].append({'path': str(path), 'sha256': sha,
                                              'optimizer_steps': driver.optimizer_steps})
                state.update(summary=driver.summary(), updated_at=now())
                _publish_json(state_path, state)
            state.update(state=driver.summary()['state'], summary=driver.summary(), updated_at=now())
            attempt.update(state=state['state'], ended_at=now(), elapsed_seconds=time.monotonic() - started)
            _publish_json(state_path, state)
            return state
        except BaseException as exc:
            state.update(state='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                         error=str(exc), updated_at=now())
            attempt.update(state=state['state'], ended_at=now(), elapsed_seconds=time.monotonic() - started)
            _publish_json(state_path, state)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'annotations', 'manifest', 'cache-job', 'run-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--registration', type=Path, default=Path('experiments/event-v1.json'))
    p.add_argument('--max-batches', type=int)
    p.add_argument('--checkpoint-batches', type=int, default=25)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    try:
        state = run(args)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(2, f'Head training stopped: {exc}\n')
    print(json.dumps({'state': state['state'], 'summary': state['summary']}, indent=2))


if __name__ == '__main__':
    main()
