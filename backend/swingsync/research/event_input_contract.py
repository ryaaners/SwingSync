"""Generic local video admission for research event proposals, not serving/accuracy.

Only presentation timestamps and coded upright pixels are trusted here. Metadata
is an operator declaration; no swing/identity/camera distribution is inferred.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

MAX_BYTES = 150 * 1024 * 1024
MAX_FRAMES = 3600
MAX_SECONDS = 30
MAX_WIDTH = 3840
MAX_HEIGHT = 2160
FORMATS = {'.mp4': 'mov', '.mov': 'mov', '.webm': 'matroska', '.avi': 'avi'}
INPUT_PROTOCOL_SHA256 = '30b270ab29362a7645598e65a953108a1b2582c06acec48fc9b68ab3f76f8caa'
MODEL_CONTRACT_SHA256 = '4b4339a469dd5fe09f97b57ae4b650ce378ce1c3a3863f86b46249ad8d9d0008'
ROOT_LOCK_SHA256 = '8ce1d6674a07bf122541f555b90ad81532029b7ce343af54a78cd057fc2b26fc'


def validate_protocol(path):
    """Admit only this pinned research protocol and its actual operational policy."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_PROTOCOL_SHA256:
        raise ValueError('Input protocol byte fingerprint differs from registered v1')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate input protocol key')
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError('Nonfinite protocol value')

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    inference = {'device': 'cpu', 'torch_num_threads': 1, 'batch_size': 16,
                 'context': 'reference-64-reset', 'heads': ['frame', 'temporal'],
                 'selection': 'best_model', 'order_constrained_decoding': False,
                 'accuracy_metrics_computed': False}
    if (type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or value.get('study_id') != 'event-input-v1'
            or value.get('model_contract') != {'path': 'experiments/heldout-v1.json',
                                              'sha256': MODEL_CONTRACT_SHA256}
            or value.get('dependency_lock_sha256') != ROOT_LOCK_SHA256
            or json.dumps(value.get('inference'), sort_keys=True) != json.dumps(inference, sort_keys=True)):
        raise ValueError('Input protocol operational model/runtime policy differs')
    policy = value['input_policy']
    limits = {'max_bytes': MAX_BYTES, 'max_frames': MAX_FRAMES, 'max_seconds': MAX_SECONDS,
              'max_width': MAX_WIDTH, 'max_height': MAX_HEIGHT}
    if (any(type(policy.get(k)) is not int or policy[k] != expected for k, expected in limits.items())
            or policy.get('pre_backbone_roi_crop') is not None
            or policy.get('pre_backbone_padding') is not None
            or policy.get('unmirrored_required') is not True
            or policy.get('single_swing_confirmation_required') is not True
            or policy.get('single_video_stream') is not True):
        raise ValueError('Input protocol video admission policy differs')
    gates = value['gates']
    expected_gates = {'input_distribution_validated', 'uploaded_video_accuracy_validated',
                      'normal_app_model_available', 'pose_measurements_permitted',
                      'outcome_predictions_permitted', 'coaching_permitted'}
    if set(gates) != expected_gates or any(v is not False for v in gates.values()):
        raise ValueError('Input protocol cannot promote unvalidated integration gates')
    return value


def validate_metadata(camera_view, handedness, club, unmirrored, single_swing_confirmed, rights, timestamp_basis):
    if camera_view not in {'face-on', 'down-the-line'} or handedness not in {'left', 'right'}:
        raise ValueError('Explicit supported camera view and handedness required')
    if unmirrored is not True or single_swing_confirmed is not True:
        raise ValueError('Explicit unmirrored and reviewed single-swing declarations required')
    if not isinstance(club, str) or not club.strip() or len(club) > 80:
        raise ValueError('Nonempty club metadata of at most80 characters required')
    if not isinstance(rights, str) or not rights.strip() or len(rights) > 2000:
        raise ValueError('Explicit local media rights/provenance required')
    if timestamp_basis not in {'source', 'reencoded'}:
        raise ValueError('Timestamp basis must be source or reencoded')
    return {'camera_view': camera_view, 'handedness': handedness, 'club': club.strip(),
            'mirrored': False, 'single_swing_confirmed_by_operator': True,
            'rights_provenance': rights.strip(), 'timestamp_basis': timestamp_basis,
            'metadata_empirically_validated': False, 'input_distribution_validated': False,
            'physical_timing_validated': False}


def transform_geometry(width, height):
    """Describe pinned tensor resize232/center224 geometry, without applying it.

    This models integer output dimensions/center offsets, not an assertion that
    bilinear interpolation maps continuous coordinates without sampling effects.
    No bbox crop, letterbox removal, mirror or rotation is silently inserted.
    """
    if any(type(v) is not int or v < 1 for v in (width, height)):
        raise ValueError('Positive integer raster dimensions required')
    if width <= height:
        resized_width, resized_height = 232, int(232 * height / width)
    else:
        resized_width, resized_height = int(232 * width / height), 232
    left = int(round((resized_width - 224) / 2.0))
    top = int(round((resized_height - 224) / 2.0))
    return {'source_width': width, 'source_height': height,
            'resized_width': resized_width, 'resized_height': resized_height,
            'center_crop_left': left, 'center_crop_top': top,
            'crop_width': 224, 'crop_height': 224,
            'resize_short_edge': 232, 'interpolation': 'bilinear', 'antialias': True,
            'input': 'RGB-uint8-CHW', 'transform': 'IMAGENET1K_V2-tensor-resize232-center224',
            'pre_backbone_roi_crop': None, 'pre_backbone_padding': None,
            'crop_subject_coverage_validated': False}


def _check_geometry(stream, frame=None):
    try:
        rotation = float(stream.metadata.get('rotate', '0'))
    except (TypeError, ValueError):
        raise ValueError('Unknown display rotation is not admitted') from None
    if not math.isfinite(rotation) or rotation != 0:
        raise ValueError('Nonzero/unknown display rotation is not admitted')
    aspect = stream.sample_aspect_ratio
    if aspect is None or Fraction(aspect) != 1:
        raise ValueError('Explicit square pixel aspect is required')
    for side in getattr(stream, 'side_data', ()):
        if 'DISPLAYMATRIX' in str(side.type).upper().replace('_', ''):
            raise ValueError('Display matrix requires a separately verified transform')
    coded = stream.codec_context
    if (type(coded.width) is not int or type(coded.height) is not int
            or not 1 <= coded.width <= MAX_WIDTH or not 1 <= coded.height <= MAX_HEIGHT):
        raise ValueError('Coded dimensions exceed supported raster budget')
    if frame is not None:
        if getattr(frame, 'rotation', 0) != 0:
            raise ValueError('Nonzero decoded display rotation is not admitted')
        if any('DISPLAYMATRIX' in str(side.type).upper().replace('_', '') for side in frame.side_data):
            raise ValueError('Display matrix requires a separately verified transform')
        if not 1 <= frame.width <= MAX_WIDTH or not 1 <= frame.height <= MAX_HEIGHT:
            raise ValueError('Decoded raster exceeds supported dimensions')


def inspect_input(video):
    """Inspect every bounded native frame; caller must isolate native decoding.

    No model, annotation, sampling/FPS reconstruction, resampling or event label is
    loaded. The runner wraps this trusted-local helper in a45s child process.
    Duration uses observed PTS span plus a known final frame duration, never FPS.
    If no final duration is supplied, duration_seconds is an observed-span bound.
    """
    import av
    path = Path(video)
    if path.suffix.lower() not in FORMATS:
        raise ValueError('Unsupported local video extension')
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError('Video file is empty or exceeds150MiB')
    frames, first, previous, previous_float, dimensions, final_duration = [], None, None, None, None, None
    with av.open(str(path), format=FORMATS[path.suffix.lower()], options={'protocol_whitelist': 'file'}) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Exactly one video stream required')
        stream = container.streams.video[0]
        _check_geometry(stream)
        if stream.duration is not None and stream.time_base is not None:
            declared_duration = Fraction(stream.duration) * Fraction(stream.time_base)
            if declared_duration < 0 or declared_duration > MAX_SECONDS:
                raise ValueError('Declared recording duration exceeds30seconds or is invalid')
        for index, frame in enumerate(container.decode(video=0)):
            if index >= MAX_FRAMES:
                raise ValueError('Decoded frame count exceeds3600')
            _check_geometry(stream, frame)
            shape = (frame.width, frame.height)
            if dimensions is not None and shape != dimensions:
                raise ValueError('Changing decoded raster dimensions are not admitted')
            dimensions = shape
            if frame.pts is None or frame.time_base is None or type(frame.pts) is not int:
                raise ValueError('Actual integer decoded PTS and rational time base required')
            numerator, denominator = frame.time_base.numerator, frame.time_base.denominator
            if (not -(2**63) <= frame.pts < 2**63 or not 0 < numerator < 2**63
                    or not 0 < denominator < 2**63):
                raise ValueError('Native PTS/time base outside signed-int64 contract')
            exact = frame.pts * Fraction(numerator, denominator)
            seconds = float(exact)
            if (not math.isfinite(seconds) or (previous is not None and exact <= previous)
                    or (previous_float is not None and seconds <= previous_float)):
                raise ValueError('Actual PTS must increase and remain distinct as float64')
            if first is None:
                first = exact
            if exact - first > MAX_SECONDS:
                raise ValueError('Observed native PTS span exceeds30seconds')
            native_duration = getattr(frame, 'duration', None)
            if native_duration is not None and (type(native_duration) is not int or native_duration < 0):
                raise ValueError('Invalid decoded frame duration')
            final_duration = native_duration * Fraction(numerator, denominator) if native_duration else None
            frames.append({'frame_index': index, 'pts': frame.pts, 'time_base_numerator': numerator,
                           'time_base_denominator': denominator, 'pts_seconds': seconds})
            previous, previous_float = exact, seconds
    if len(frames) < 2:
        raise ValueError('At least two decoded frames required')
    duration = previous - first + (final_duration or 0)
    if duration > MAX_SECONDS:
        raise ValueError('Observed duration including last frame exceeds30seconds')
    return {'frames': frames, 'width': dimensions[0], 'height': dimensions[1], 'frame_count': len(frames),
            'duration_seconds': float(duration),
            'duration_method': 'native-PTS-span-plus-known-final-duration' if final_duration else 'native-PTS-span-only',
            'last_frame_duration_seconds': float(final_duration) if final_duration else None,
            'pixel_aspect': '1/1', 'display_transform': 'coded-upright-unrotated',
            'crop': None, 'resize': None, 'timestamp_method': 'decoded-presentation-timestamps',
            'transform_geometry': transform_geometry(*dimensions)}
