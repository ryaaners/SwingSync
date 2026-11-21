"""Measure the registered CPU pipeline on fixed TRAIN clips after all six heads finish."""
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
    parser.add_argument('--expected-receipt-sha256', required=True,
                        help='Independently known 018 completion receipt SHA256; arbitrary hashes are refused')
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    from backend.swingsync.research.additional_latency import run_additional_latency
    options = vars(args)
    if options['protocol'] is None:
        del options['protocol']
    else:
        options['protocol_path'] = options.pop('protocol')
    try:
        result = run_additional_latency(**options)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f'Additional-fold CPU timing unavailable: {exc}\n')
    print(json.dumps({'state': result['state'], 'job_id': result['job_id'],
                      'completed_clip_observations': result['completed_clip_observations'],
                      'accuracy_metrics_computed': False, 'validated_for_serving': False}, indent=2))


if __name__ == '__main__':
    main()
