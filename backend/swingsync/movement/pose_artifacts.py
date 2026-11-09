"""Authenticate completed pose observations for offline review, never model inference.

An independently recorded ledger SHA is mandatory. This verifies correspondence
and saved engineering policies, not landmark accuracy or a golfer's identity.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

from backend.swingsync.pose_runtime.contracts import pose_observation, validate_context, video_clock
from scripts.acquire_pose_asset import LICENSE_EVIDENCE, MEMBERS, MODEL_SHA256, MODEL_SIZE, MODEL_URL

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / 'artifacts/training/active-training.lock'
POSE_LOCK = ROOT / 'pose_runtime/uv.lock'
MAX_JSON_BYTES = 100 * 1024 * 1024


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _digest(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Expected independently recorded lowercase SHA256')
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON keys are not admitted')
        result[key] = value
    return result


def _json(path, expected=None):
    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ValueError('Pose evidence JSON size budget exceeded')
    raw = path.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != _digest(expected):
        raise ValueError('Pose evidence SHA256 mismatch')

    def nonfinite(value):
        raise ValueError('Nonfinite JSON values are not admitted')

    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError('Pose evidence must be a JSON object')
    return value


@dataclass(frozen=True)
class VerifiedPoseArtifact:
    run: dict
    result: dict
    run_sha256: str
    result_sha256: str
    video_path: Path
    run_path: Path


def _source_paths():
    return sorted([*(ROOT / 'backend/swingsync/pose_runtime').glob('*.py'),
                   *(ROOT / 'scripts' / name for name in (
                       'extract_pose.py', 'run_pose_offline.py', 'acquire_pose_asset.py'))])


def _validate_provenance(run):
    model = run['model']
    manifest = model['manifest']
    expected = dict(schema_version=1, url=MODEL_URL, sha256=MODEL_SHA256,
                    size_bytes=MODEL_SIZE, members=MEMBERS, license_evidence=LICENSE_EVIDENCE)
    if model['bundle_sha256'] != MODEL_SHA256 or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Pose model registration differs from pinned upstream asset/policy')
    _digest(model['manifest_sha256'])
    runtime = run['runtime']
    if runtime['pose_lock_sha256'] != sha256(POSE_LOCK) or runtime['delegate'] != 'CPU':
        raise ValueError('Pose runtime lock/delegate drift')
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in _source_paths()}
    if runtime['adapter_source_sha256'] != sources:
        raise ValueError('Pose extraction/launcher source fingerprint drift')
    locked = {p['name']: p['version'] for p in tomllib.loads(POSE_LOCK.read_text())['package']}
    installed = runtime['installed_distributions']
    if not isinstance(installed, dict) or any(locked.get(k) != v for k, v in installed.items()):
        raise ValueError('Recorded pose packages differ from complete lock')
    direct = {'mediapipe': '1.2.2', 'numpy': '2.2.5', 'opencv-contrib-python': '4.12.0.88', 'av': '16.0.1'}
    if any(installed.get(k) != v for k, v in direct.items()):
        raise ValueError('Recorded pose dependency pins differ')
    expected_config = {
        'running_mode': 'VIDEO', 'num_poses': 2, 'delegate': 'CPU',
        'min_pose_detection_confidence': 0.5, 'min_pose_presence_confidence': 0.5,
        'min_tracking_confidence': 0.5, 'presence_filter_floor': 0.5, 'output_segmentation_masks': False,
        'clock': 'exact-rational-relative-half-up-ms-v1',
        'limits': {'bytes': 150 * 1024 * 1024, 'frames': 3600, 'seconds': 30},
    }
    if run['config'] != expected_config:
        raise ValueError('Pose extraction configuration differs from registered engineering policy')
    if run.get('network_policy') != 'OS deny-all network via supported launcher; IPv4/IPv6 denial guard passed':
        raise ValueError('Recorded offline extraction policy required')
    context = run['context']
    expected_context = validate_context(
        context['camera_view'], context['handedness'], context['mirrored'] is False,
        context['single_golfer_confirmed_by_operator'], context['rights_provenance'], context['timestamp_basis'])
    if context != expected_context:
        raise ValueError('Pose metadata/identity declaration differs')


def _inspect_bounded(video_path):
    try:
        proc = subprocess.run([sys.executable, '-m', __name__, '--inspect', str(video_path)],
                              cwd=ROOT, capture_output=True, text=True, timeout=45, check=True)
        return json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError('Bounded pose source inspection failed; no review artifact admitted') from exc


def assert_artifact_unchanged(artifact):
    """Recheck disk, snapshot, source and environment before derived publication."""
    if (_json(artifact.run_path, artifact.run_sha256) != artifact.run
            or _json(artifact.run_path.with_name('result.json'), artifact.result_sha256) != artifact.result):
        raise ValueError('Pose artifact snapshot drift')
    if sha256(artifact.video_path) != artifact.run['video_sha256']:
        raise ValueError('Source video bytes changed')
    _validate_provenance(artifact.run)


def load_pose_artifact(run_path, *, expected_run_sha256, video_path):
    """Admit a SHA-bound completed run and independently inspect every source PTS.

    No task initialization, new pose inference, interpolation or event prediction.
    Missing/multi-person observations remain explicit for diagnostic rendering.
    """
    run_path, video_path = Path(run_path).resolve(), Path(video_path).resolve()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another expensive job owns the shared lock') from exc
        run = _json(run_path, expected_run_sha256)
        if run.get('schema_version') != 1 or run.get('state') != 'completed':
            raise ValueError('Completed version1 pose ledger required')
        result_sha = _digest(run['result_sha256'])
        result = _json(run_path.with_name('result.json'), result_sha)
        if result.get('schema_version') != 1 or result.get('state') != 'completed':
            raise ValueError('Completed version1 pose result required')
        for key in ('job_id', 'created_at', 'completed_at', 'pid', 'context', 'video_sha256', 'video',
                    'model', 'runtime', 'config', 'network_policy', 'advice', 'validated_for_coaching'):
            if result.get(key) != run.get(key):
                raise ValueError(f'Pose result/ledger disagree: {key}')
        if (run.get('advice') != [] or run.get('validated_for_coaching') is not False
                or run.get('resume_supported') is not False):
            raise ValueError('Unvalidated offline observations must not claim advice/admission')
        _validate_provenance(run)
        if sha256(video_path) != _digest(run['video_sha256']):
            raise ValueError('Source video SHA256 mismatch')
        inspected = _inspect_bounded(video_path)
        if inspected['geometry'] != run['video']:
            raise ValueError('Pose source raster/transform correspondence differs')
        frames, clocks = result['frames'], inspected['clocks']
        if not isinstance(frames, list) or len(frames) != len(clocks) or not 1 <= len(frames) <= 3600:
            raise ValueError('Pose frame coverage differs from source')
        for frame, clock in zip(frames, clocks):
            for key, value in clock.items():
                expected_type = float if key == 'pts_seconds' else int
                if frame.get(key) != value or type(frame.get(key)) is not expected_type:
                    raise ValueError('Pose native PTS/API clock correspondence differs')
            if not isinstance(frame['raw_poses'], list) or len(frame['raw_poses']) > 2:
                raise ValueError('Raw pose count exceeds registered detector cap')
            if any(not isinstance(pose, list) or any(
                not isinstance(p, dict) or set(p) != {'x', 'y', 'z', 'visibility', 'presence'} for p in pose
            ) for pose in frame['raw_poses']):
                raise ValueError('Malformed raw landmark schema')
            poses = [[SimpleNamespace(**p) for p in pose] for pose in frame['raw_poses']]
            recomputed = pose_observation(poses, presence_floor=run['config']['presence_filter_floor'])
            if any(frame.get(k) != v for k, v in recomputed.items()):
                raise ValueError('Pose raw/filtered observation correspondence differs')
        # Revalidate the exact rational map rather than trusting saved API clocks.
        exact = video_clock([(f['pts'], f['time_base_numerator'], f['time_base_denominator']) for f in frames])
        if [asdict(row) for row in exact] != clocks:
            raise ValueError('Pose clock derivation differs')
        counts = dict(Counter(f['status'] for f in frames))
        ambiguous = any(f['pose_count'] > 1 for f in frames)
        if (counts != run['status_counts'] or counts != result['status_counts']
                or result.get('identity_ambiguous') is not ambiguous
                or result.get('clock_origin_seconds') != frames[0]['pts_seconds']
                or result.get('api_clock_is_measurement_coordinate') is not False):
            raise ValueError('Pose status/identity/clock summary differs')
        admitted = VerifiedPoseArtifact(run, result, expected_run_sha256, result_sha, video_path, run_path)
        assert_artifact_unchanged(admitted)
        return admitted


def _main():
    """Private bounded child: decoding only; never imports a MediaPipe task."""
    if len(sys.argv) != 3 or sys.argv[1] != '--inspect':
        raise SystemExit('Use the offline evidence CLI')
    import av
    from backend.swingsync.pose_runtime.extraction import inspect_video
    clocks, geometry = inspect_video(av, Path(sys.argv[2]))
    print(json.dumps({'clocks': [asdict(c) for c in clocks], 'geometry': geometry}, allow_nan=False))


if __name__ == '__main__':
    _main()
