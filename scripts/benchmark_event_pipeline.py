"""Measure fixed CPU video inference stages on registered training clips only."""
import argparse
import json
from pathlib import Path

from backend.swingsync.research.latency import run_latency_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'annotations', 'manifest', 'clips-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_latency_benchmark(args.contract, args.annotations, args.manifest, args.clips_dir, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f'Latency benchmark stopped: {exc}\n')
    print(json.dumps({'state': result['state'], 'frame_count': result['frame_count'],
                      'output': str(args.output), 'accuracy_metrics_computed': False}, indent=2))


if __name__ == '__main__':
    main()
