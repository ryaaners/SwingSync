"""Explicit isolated CPU VIDEO extraction. Product inference remains unavailable."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import fcntl
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import tomllib
import uuid
import zipfile

from .contracts import pose_observation, validate_context, video_clock

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / 'artifacts/training/active-training.lock'
POSE_LOCK = ROOT / 'pose_runtime/uv.lock'
MAX_BYTES = 150 * 1024 * 1024
MAX_FRAMES = 3600
MAX_SECONDS = 30
# Pure registry import; acquisition has no import-time network or filesystem action.
from scripts.acquire_pose_asset import (  # noqa: E402
    MODEL_URL, MODEL_SHA256, MODEL_SIZE as MODEL_SIZE_BYTES,
    MEMBERS as MODEL_MEMBERS, LICENSE_EVIDENCE,
)


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def publish_json(path, value, *, replace=False):
    """Publish a complete JSON file; exclusive destination unless replacing ledger."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.part')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # atomic no-clobber on this same filesystem
    finally:
        temporary.unlink(missing_ok=True)


def validate_model(model_path, manifest_path):
    model_path, manifest_path = Path(model_path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('url') != MODEL_URL:
        raise ValueError('Unregistered model URL/schema')
    if manifest.get('sha256') != MODEL_SHA256 or manifest.get('size_bytes') != MODEL_SIZE_BYTES:
        raise ValueError('Manifest differs from independently pinned model bytes')
    if sorted(manifest.get('members', []), key=lambda member: member['name']) != MODEL_MEMBERS:
        raise ValueError('Manifest differs from independently pinned model members')
    if manifest.get('license_evidence') != LICENSE_EVIDENCE:
        raise ValueError('Exact separate model/code license and offline-privacy policy required')
    if (model_path.stat().st_size != manifest['size_bytes']
            or file_sha256(model_path) != manifest['sha256']):
        raise ValueError('Model bundle byte hash/size mismatch')
    with zipfile.ZipFile(model_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {'pose_detector.tflite', 'pose_landmarks_detector.tflite'}:
            raise ValueError('Unexpected or duplicate model bundle members')
        actual = sorted([{'name': info.filename, 'size_bytes': info.file_size,
                          'sha256': hashlib.sha256(archive.read(info)).hexdigest()}
                         for info in archive.infolist()], key=lambda member: member['name'])
    if actual != sorted(manifest['members'], key=lambda member: member['name']):
        raise ValueError('Model member manifest mismatch')
    return {'manifest': manifest, 'manifest_sha256': file_sha256(manifest_path),
            'bundle_sha256': manifest['sha256']}


def _source_hashes():
    files = [*Path(__file__).parent.glob('*.py'), ROOT / 'scripts/extract_pose.py',
             ROOT / 'scripts/run_pose_offline.py', ROOT / 'scripts/acquire_pose_asset.py']
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(files)}


def require_network_denial():
    """Fail closed unless IPv4/IPv6 native socket access is denied by OS policy.

    The supported launcher installs a deny-all-network sandbox. These numeric
    loopback checks detect an absent policy without DNS or remote traffic. They
    are a guard, not a portable proof that arbitrary firewall rules deny egress.
    """
    for family, address in ((socket.AF_INET, ('127.0.0.1', 9)),
                            (socket.AF_INET6, ('::1', 9, 0, 0))):
        connection = None
        try:
            connection = socket.socket(family, socket.SOCK_STREAM)
            connection.settimeout(0.25)
            connection.connect(address)
        except OSError as exc:
            if exc.errno in {errno.EPERM, errno.EACCES}:
                continue
            raise ValueError('OS network denial is unverified; use the offline launcher') from exc
        else:
            raise ValueError('Network access remains available; native pose inference refused')
        finally:
            if connection is not None:
                connection.close()


def runtime_identity():
    """Require all installed distribution versions to belong to this complete lock."""
    if sys.version_info[:2] != (3, 11) or sys.prefix == sys.base_prefix:
        raise ValueError('Use the separate Python3.11 pose virtual environment')
    if Path(sys.prefix).resolve() == (ROOT / '.venv').resolve():
        raise ValueError('Root environment must not host pose extraction')
    lock = tomllib.loads(POSE_LOCK.read_text())
    locked = {}
    for package in lock['package']:
        locked.setdefault(package['name'].lower().replace('_', '-'), set()).add(package['version'])
    installed = {d.metadata['Name'].lower().replace('_', '-'): d.version
                 for d in importlib.metadata.distributions()}
    for name, version in installed.items():
        if version not in locked.get(name, set()):
            raise ValueError(f'Installed distribution outside pose lock: {name}=={version}')
    expected = {'mediapipe': '1.2.2', 'numpy': '2.2.5', 'opencv-contrib-python': '4.12.0.88', 'av': '16.0.1'}
    if any(installed.get(name) != version for name, version in expected.items()):
        raise ValueError('Pose runtime direct dependency versions differ from registration')
    return {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
            'python_executable_sha256': file_sha256(sys.executable), 'prefix': sys.prefix,
            'platform': platform.platform(), 'machine': platform.machine(),
            'pose_lock_sha256': file_sha256(POSE_LOCK), 'installed_distributions': installed,
            'adapter_source_sha256': _source_hashes(), 'delegate': 'CPU'}


def _check_geometry(stream, frame=None):
    for metadata in (stream.metadata,):
        rotate = metadata.get('rotate', '0')
        try:
            if float(rotate) != 0:
                raise ValueError('Rotated video is not admitted')
        except (ValueError, TypeError):
            raise ValueError('Unknown/nonzero display rotation is not admitted') from None
    aspect = stream.sample_aspect_ratio
    if aspect is None or Fraction(aspect) != 1:
        raise ValueError('Explicit square pixel aspect is required')
    if frame is not None:
        if getattr(frame, 'rotation', 0) != 0:
            raise ValueError('Nonzero decoded display rotation is not admitted')
        if any('DISPLAYMATRIX' in str(data.type).upper().replace('_', '') for data in frame.side_data):
            raise ValueError('Display matrix needs a separately verified upright transform')
        if not 1 <= frame.width <= 3840 or not 1 <= frame.height <= 2160:
            raise ValueError('Frame dimensions exceed pose decode budget')


def _open_video(av, path):
    formats = {'.mp4': 'mov', '.mov': 'mov', '.webm': 'matroska', '.avi': 'avi'}
    suffix = Path(path).suffix.lower()
    if suffix not in formats:
        raise ValueError('Unsupported local video extension')
    return av.open(str(path), format=formats[suffix], options={'protocol_whitelist': 'file'})


def inspect_video(av, path):
    """Decode all bounded frames before constructing the native VIDEO detector."""
    if not 0 < Path(path).stat().st_size <= MAX_BYTES:
        raise ValueError('Video file size exceeds budget or is empty')
    ticks, dimensions = [], None
    with _open_video(av, path) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Exactly one video stream required')
        stream = container.streams.video[0]
        _check_geometry(stream)
        for index, frame in enumerate(container.decode(video=0)):
            if index >= MAX_FRAMES:
                raise ValueError('Pose frame budget exceeded')
            _check_geometry(stream, frame)
            current_dimensions = (frame.width, frame.height)
            if dimensions is not None and current_dimensions != dimensions:
                raise ValueError('Changing raster dimensions are not admitted')
            dimensions = current_dimensions
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Missing native PTS/time base')
            ticks.append((frame.pts, frame.time_base.numerator, frame.time_base.denominator))
            if len(ticks) > 1:
                elapsed = Fraction(ticks[-1][0] * ticks[-1][1], ticks[-1][2]) - Fraction(
                    ticks[0][0] * ticks[0][1], ticks[0][2])
                if elapsed > MAX_SECONDS:
                    raise ValueError('Pose recording time span exceeds30seconds')
    clocks = video_clock(ticks)
    return clocks, {'width': dimensions[0], 'height': dimensions[1], 'pixel_aspect': '1/1',
                    'display_transform': 'coded-upright-unrotated', 'crop': None, 'resize': None,
                    'frame_count': len(clocks)}


def _infer_frames(av, mp, detector, path, clocks, geometry):
    frames = []
    started = time.perf_counter()
    with _open_video(av, path) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(video=0)):
            if index >= len(clocks):
                raise ValueError('Decoded frame count drifted')
            _check_geometry(stream, frame)
            clock = clocks[index]
            if ((frame.pts, frame.time_base.numerator, frame.time_base.denominator)
                    != (clock.pts, clock.time_base_numerator, clock.time_base_denominator)
                    or (frame.width, frame.height) != (geometry['width'], geometry['height'])):
                raise ValueError('Decoded PTS/raster drifted')
            rgb = frame.to_ndarray(format='rgb24')
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(image, clock.api_timestamp_ms)
            frames.append({**asdict(clock), **pose_observation(result.pose_landmarks)})
    if len(frames) != len(clocks):
        raise ValueError('Decoded frame count drifted')
    return frames, time.perf_counter() - started


def extract_pose(*, video, model, model_manifest, output, camera_view, handedness,
                 rights, timestamp_basis, unmirrored=False, single_golfer_confirmed=False):
    """No resume/overwrite: interrupted identities must be inspected by the caller."""
    context = validate_context(camera_view, handedness, unmirrored, single_golfer_confirmed,
                               rights, timestamp_basis)
    video, model, model_manifest, output = map(Path, (video, model, model_manifest, output))
    if output.exists():
        raise ValueError('Output already exists; inspect its saved job before choosing a new identity')
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another expensive job owns the shared lock; no duplicate work started') from exc
        environment = runtime_identity()
        registered_model = validate_model(model, model_manifest)
        video_hash = file_sha256(video)
        output.mkdir(parents=True, exist_ok=False)
        ledger_path = output / 'run.json'
        ledger = {'schema_version': 1, 'state': 'initializing', 'job_id': uuid.uuid4().hex,
                  'created_at': _now(), 'pid': os.getpid(), 'context': context,
                  'video_path': str(video.resolve()), 'video_sha256': video_hash,
                  'model': registered_model, 'runtime': dict(environment),
                  'config': {'running_mode': 'VIDEO', 'num_poses': 2, 'delegate': 'CPU',
                             'min_pose_detection_confidence': 0.5, 'min_pose_presence_confidence': 0.5,
                             'min_tracking_confidence': 0.5, 'presence_filter_floor': 0.5,
                             'output_segmentation_masks': False,
                             'clock': 'exact-rational-relative-half-up-ms-v1',
                             'limits': {'bytes': MAX_BYTES, 'frames': MAX_FRAMES, 'seconds': MAX_SECONDS}},
                  'advice': [], 'validated_for_coaching': False, 'resume_supported': False}
        publish_json(ledger_path, ledger)
        try:
            require_network_denial()
            ledger['network_policy'] = 'OS deny-all network via supported launcher; IPv4/IPv6 denial guard passed'
            import av
            import cv2
            import mediapipe as mp
            import numpy as np
            ledger['runtime']['native_libraries'] = {
                'av_library_versions': av.library_versions,
                'opencv_version': cv2.__version__, 'numpy_version': np.__version__,
                'mediapipe_version': mp.__version__,
                'opencv_build_information_sha256': hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()}
            clocks, geometry = inspect_video(av, video)
            if file_sha256(video) != video_hash:
                raise ValueError('Source video changed during preflight')
            ledger.update(state='running', inference_started_at=_now(), video=geometry)
            publish_json(ledger_path, ledger, replace=True)
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model),
                                                 delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=2, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5, output_segmentation_masks=False)
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
                frames, elapsed = _infer_frames(av, mp, detector, video, clocks, geometry)
            if (file_sha256(video) != video_hash or runtime_identity() != environment
                    or validate_model(model, model_manifest) != registered_model):
                raise ValueError('Source/model/runtime/adapter drift during extraction')
            result = {**ledger, 'state': 'completed', 'completed_at': _now(), 'frames': frames,
                      'clock_origin_seconds': clocks[0].pts_seconds,
                      'api_clock_is_measurement_coordinate': False,
                      'inference_decode_elapsed_seconds': elapsed,
                      'status_counts': dict(Counter(frame['status'] for frame in frames)),
                      'identity_ambiguous': any(f['status'] == 'multiple_poses' for f in frames),
                      'identity_policy': 'Any multiple-pose frame suppresses the whole spatial sequence; raw rows retained',
                      'limitations': ['Unvalidated golf landmark and identity accuracy',
                                      'Operator identity/view/mirror claims are not calibrated reliability',
                                      'Presence/visibility thresholds are engineering filters only',
                                      'Single returned pose is not proof of one person; misses remain possible',
                                      'No calibrated3D, outcome prediction, advice or app integration']}
            publish_json(output / 'result.json', result)
            ledger.update(state='completed', completed_at=result['completed_at'],
                          result_sha256=file_sha256(output / 'result.json'),
                          status_counts=result['status_counts'], inference_decode_elapsed_seconds=elapsed)
            publish_json(ledger_path, ledger, replace=True)
            return result
        except BaseException as exc:
            (output / 'result.json').unlink(missing_ok=True)
            ledger.update(state='failed', failed_at=_now(), error_type=type(exc).__name__, error=str(exc)[:2000])
            publish_json(ledger_path, ledger, replace=True)
            raise
