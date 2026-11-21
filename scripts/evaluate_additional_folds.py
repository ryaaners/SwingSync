"""Evaluate fixed additional outer folds using authenticated completed selected heads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('annotations', 'manifests-dir', 'audit-run', 'smokes-dir', 'training-runs-dir',
                 'training-receipt', 'clips-dir', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-clips', type=int)
    args = parser.parse_args()
    if args.max_new_clips is not None and args.max_new_clips < 1:
        parser.error('--max-new-clips must be positive')
    options = vars(args)
    protocol = options.pop('protocol')
    if protocol is not None:
        options['protocol_path'] = protocol
    try:
        from backend.swingsync.research.additional_evaluation import run_additional_evaluation
        state = run_additional_evaluation(**options)
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f'Additional-fold evaluation unavailable: {exc}\n')
    print(json.dumps({'state': state['state'], 'job_id': state['job_id'],
                      'completed_clips': len(state['clips']), 'validated_for_serving': False}, indent=2))


if __name__ == '__main__':
    main()
