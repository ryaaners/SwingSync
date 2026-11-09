"""Build local diagnostic overlays and a separate clean manual reference page."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from backend.swingsync.movement.pose_artifacts import load_pose_artifact
    from backend.swingsync.research.pose_evidence import build_pose_evidence
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--run-sha256', required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-index', type=int, action='append', required=True)
    parser.add_argument('--manual-events', type=Path)
    args = parser.parse_args()
    try:
        artifact = load_pose_artifact(args.run, expected_run_sha256=args.run_sha256, video_path=args.video)
        document = json.loads(args.manual_events.read_text()) if args.manual_events else None
        result = build_pose_evidence(artifact, args.frame_index, args.output, manual_events=document)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Pose review refused: {exc}\n')
    print(json.dumps({'output': str(args.output), 'state': result['state'],
                      'selected_frames': result['selected_frame_indices'],
                      'accuracy_evaluation_completed': False, 'validated_for_coaching': False}, indent=2))


if __name__ == '__main__':
    main()
