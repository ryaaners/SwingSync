"""Launch the isolated pose CLI inside a verified macOS network-denial profile.

MediaPipe's wheel notice mentions utilization telemetry. This launcher denies
native network access, rather than patching Python sockets or assuming an opt-out.
Linux extraction is deliberately unavailable until its isolation is verified.
"""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROFILE = '(version 1)(allow default)(deny network*)'


def command(arguments: list[str]) -> list[str]:
    if sys.platform != 'darwin':
        raise ValueError('Pose extraction needs verified OS network isolation; only macOS is supported yet.')
    sandbox = Path('/usr/bin/sandbox-exec')
    python = ROOT / '.tools/pose-runtime/bin/python'
    if not sandbox.is_file() or not python.is_file():
        raise ValueError('Install the isolated pose environment and ensure macOS sandbox-exec is available.')
    return [str(sandbox), '-p', PROFILE, str(python), str(ROOT / 'scripts/extract_pose.py'), *arguments]


def main() -> int:
    try:
        # No shell interpolation. Child checks both IPv4/IPv6 denial before importing MediaPipe.
        return subprocess.run(command(sys.argv[1:]), cwd=ROOT, check=False, timeout=600).returncode
    except subprocess.TimeoutExpired:
        print('Pose unavailable: 600-second process limit exceeded. Inspect the interrupted run; do not resume it automatically.',
              file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f'Pose unavailable: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
