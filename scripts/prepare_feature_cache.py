"""Prepare resumable frozen features; no event training or outer-test evaluation."""
import argparse
import json
from pathlib import Path

from backend.swingsync.research.cache_job import run_cache_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('artifacts/features'))
    parser.add_argument('--partitions', nargs='+', choices=['train', 'validation', 'test'],
                        default=['train', 'validation'])
    parser.add_argument('--allow-test', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-new-clips', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        state = run_cache_job(args.annotations, args.manifest, args.clips_dir, args.job_dir, args.cache_dir,
                              partitions=args.partitions, allow_test=args.allow_test, device=args.device,
                              batch_size=args.batch_size, max_new_clips=args.max_new_clips, resume=args.resume)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Feature preparation stopped: {exc}\n')
    print(json.dumps({'state': state['state'], 'job_id': state['job_id'],
                      'completed_clips': state['completed_clip_count'],
                      'target_clips': state['target_clip_count'], 'latest_attempt': state['attempts'][-1]}, indent=2))


if __name__ == '__main__':
    main()
