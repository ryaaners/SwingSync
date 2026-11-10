"""Offline selected event proposals; never enables normal app model availability."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('video', 'output', 'contract', 'annotations', 'manifest'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'experiments/event-input-v1.json')
    parser.add_argument('--camera-view', choices=('face-on', 'down-the-line'), required=True)
    parser.add_argument('--handedness', choices=('left', 'right'), required=True)
    parser.add_argument('--club', required=True)
    parser.add_argument('--rights', required=True)
    parser.add_argument('--timestamp-basis', choices=('source', 'reencoded'), required=True)
    parser.add_argument('--unmirrored', action='store_true', required=True)
    parser.add_argument('--single-swing-confirmed', action='store_true', required=True)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    metadata_names = ('camera_view', 'handedness', 'club', 'rights', 'timestamp_basis',
                      'unmirrored', 'single_swing_confirmed')
    values = vars(args).copy()
    metadata = {key: values.pop(key) for key in metadata_names}
    payload = {key: str(value.resolve()) for key, value in values.items()}
    payload['metadata'] = metadata
    try:
        # Separate native process; no shell, no automatic retries or downloads.
        code = ('import json,sys; from backend.swingsync.research.event_input import run_event_input; '
                'r=run_event_input(**json.load(sys.stdin)); '
                'print(json.dumps({"state":r["state"],"frames":len(r["frames"]),'
                '"normal_model_available":False,"accuracy_metrics_computed":False}))')
        child = subprocess.run([sys.executable, '-c', code], input=json.dumps(payload), cwd=ROOT,
                               capture_output=True, text=True, timeout=600, check=True)
    except subprocess.TimeoutExpired:
        parser.exit(2, 'Event proposals exceeded600seconds. Inspect the interrupted run; do not retry automatically.\n')
    except subprocess.CalledProcessError as exc:
        parser.exit(2, 'Event model unavailable or input refused: ' + exc.stderr[-3000:] + '\n')
    print(child.stdout.strip())


if __name__ == '__main__':
    main()
