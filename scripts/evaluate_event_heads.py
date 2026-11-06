"""Evaluate fixed selected heads on a complete registered outer-test feature job."""
import argparse
import json
from pathlib import Path

from backend.swingsync.research.heldout import run_heldout_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'annotations', 'manifest', 'cache-job', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-new-clips', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        result = run_heldout_evaluation(args.contract, args.annotations, args.manifest,
                                       args.cache_job, args.run_dir, resume=args.resume,
                                       max_new_clips=args.max_new_clips)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Outer evaluation stopped: {exc}\n')
    print(json.dumps({'state': result['state'], 'completed_clips': len(result['completed']),
                      'summary': result['summary'], 'validated_for_serving': False}, indent=2))


if __name__ == '__main__':
    main()
