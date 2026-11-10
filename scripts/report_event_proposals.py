"""Render authenticated local event observations without model inference."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from backend.swingsync.research.event_proposal_report import build_report,load_event_evidence
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('run','video','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--run-sha256',required=True)
    args=parser.parse_args()
    try:
        evidence=load_event_evidence(args.run,expected_run_sha256=args.run_sha256,video=args.video)
        report=build_report(evidence,args.output)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.exit(2,f'Event evidence unavailable: {exc}\n')
    print(f"Saved {report['event_count']} unvalidated proposal cards; no serving or coaching enabled.")


if __name__=='__main__':
    main()
