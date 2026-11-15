"""Run one registered additional-fold training-only smoke; no outer evaluation."""
import argparse
import json
from pathlib import Path

from backend.swingsync.research.fold_protocol import DEFAULT_PROTOCOL
from backend.swingsync.research.fold_training import run_fold_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--manifests-dir', required=True, type=Path)
    parser.add_argument('--audit-run', required=True, type=Path)
    parser.add_argument('--model', choices=('frame', 'temporal'), required=True)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--max-new-batches', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        state = run_fold_smoke(annotations=args.annotations, manifests_dir=args.manifests_dir,
                               audit_run=args.audit_run, model=args.model, run_dir=args.run_dir,
                               protocol_path=args.protocol, fold=args.fold, mode=args.mode,
                               max_new_batches=args.max_new_batches, resume=args.resume)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Additional-fold smoke stopped: {exc}\n')
    print(json.dumps({'job_id': state['job_id'], 'state': state['state'],
                      'smoke_completed': state['smoke_completed'], 'summary': state['summary'],
                      'full_training_completed': False, 'validated_for_serving': False}, indent=2))


if __name__ == '__main__':
    main()
