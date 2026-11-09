"""Package a hash-bound completed fixed event evaluation without inference."""
import argparse
import json
from pathlib import Path

from backend.swingsync.research.event_report import build_event_report, write_event_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'contract', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-sha256', required=True, help='Independently recorded expected run byte SHA256')
    parser.add_argument('--include-order-diagnostics', action='store_true',
                        help='Add post-hoc descriptive counts from saved predictions only')
    args = parser.parse_args()
    try:
        report = build_event_report(args.run, args.run_sha256, args.contract, args.manifest,
                                    include_order_diagnostics=args.include_order_diagnostics)
        write_event_report(report, args.output)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        parser.exit(2, f'Event report refused: {exc}\n')
    print(json.dumps({'output': str(args.output), 'input_sha256': report['input_sha256'],
                      'test_clips': report['coverage']['test_clip_count'],
                      'validated_for_serving': False}, indent=2))


if __name__ == '__main__':
    main()
