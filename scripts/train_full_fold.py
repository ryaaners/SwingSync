"""Train or explicitly resume one registered additional-fold head; no outer evaluation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('annotations', 'manifests-dir', 'audit-run', 'smokes-dir', 'run-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--model', choices=('frame', 'temporal'), required=True)
    parser.add_argument('--fold', type=int, choices=(2, 3, 4), required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-batches', type=int)
    parser.add_argument('--checkpoint-batches', type=int, default=500)
    args = parser.parse_args()
    from backend.swingsync.research.full_fold_training import run_full_fold
    options = vars(args)
    if options['protocol'] is None:
        del options['protocol']
    else:
        options['protocol_path'] = options.pop('protocol')
    try:
        state = run_full_fold(**options)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f'Additional-fold full training stopped: {exc}\n')
    summary = state['summary']
    print(json.dumps({'job_id': state['job_id'], 'state': state['state'],
                      'full_training_completed': state['full_training_completed'],
                      'optimizer_steps': summary['optimizer_steps'],
                      'completed_epochs': summary['completed_epochs'], 'best_epoch': summary['best_epoch'],
                      'selection_scope': 'inner-only-not-outer-results', 'outer_test_evaluated': False,
                      'validated_for_serving': False}, indent=2))


if __name__ == '__main__':
    main()
