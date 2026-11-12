"""Audit immutable frozen feature reuse for additional GolfDB folds; no fitting."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from backend.swingsync.research.fold_cache_audit import run_fold_cache_audit

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "train-catalog", "test-catalog", "manifests-dir", "clips-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    try:
        state = run_fold_cache_audit(
            args.annotations,
            args.train_catalog,
            args.test_catalog,
            args.manifests_dir,
            args.clips_dir,
            args.output,
            protocol_path=args.protocol,
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f"Frozen-cache audit unavailable: {exc}\n")
    print(
        f"Audit {state['job_id']} completed {state['completed_clip_count']} clips; training/evaluation/serving admission remains false."
    )


if __name__ == "__main__":
    main()
