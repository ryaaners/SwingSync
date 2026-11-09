"""Extract unvalidated local pose observations using the isolated pose interpreter."""
import argparse
import json
from pathlib import Path
import sys

# An explicit standalone invocation must not depend on root package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.swingsync.pose_runtime.extraction import extract_pose  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'model', 'model-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--camera-view', choices=('face-on', 'down-the-line'), required=True)
    parser.add_argument('--handedness', choices=('right', 'left'), required=True)
    parser.add_argument('--rights', required=True, help='Local-use permission/source provenance; not a redistribution grant')
    parser.add_argument('--timestamp-basis', choices=('original', 'reencoded'), required=True)
    parser.add_argument('--unmirrored', action='store_true', required=True)
    parser.add_argument('--single-golfer-confirmed', action='store_true', required=True)
    args = parser.parse_args()
    try:
        result = extract_pose(**vars(args))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f'Pose extraction refused: {exc}\n')
    print(json.dumps({'output': str(args.output), 'frames': len(result['frames']),
                      'status_counts': result['status_counts'], 'validated_for_coaching': False}, indent=2))


if __name__ == '__main__':
    main()
