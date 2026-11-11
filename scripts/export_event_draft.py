"""Export authenticated saved research proposals for explicit manual review only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from backend.swingsync.proposal_handoff import export_draft
    from backend.swingsync.research.event_proposal_report import load_event_evidence
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'video', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--run-sha256', required=True)
    args = parser.parse_args()
    original = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    published = False
    try:
        evidence = load_event_evidence(args.run, expected_run_sha256=args.run_sha256, video=args.video)
        result = export_draft(evidence, args.output)
        published = True
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != original:
            raise ValueError('Export CLI changed during publication.')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if published:
            args.output.unlink(missing_ok=True)
        parser.exit(2, f'Research draft unavailable: {exc}\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
