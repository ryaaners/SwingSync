"""Run the fixed, separate CaddieSet research comparison; no coaching serving."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from backend.swingsync.research.caddie_experiment import run_experiment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-folds', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        ledger = run_experiment(**vars(args))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'CaddieSet experiment unavailable: {exc}\n')
    print(f"Run {ledger['job_id']}: {ledger['state']}, {len(ledger['folds'])}/16 outer folds; "
          'video outcome integration remains disabled.')


if __name__ == '__main__':
    main()
