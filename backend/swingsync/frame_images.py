"""Bounded canonical decoded PNG evidence; no pose/model inference or browser capture.

Frames use coded orientation. Display rotation/mirroring is deliberately not
applied. Video playback may use another presentation convention; this explicit
frame-index/PTS image is the evidence for a manual annotation.
"""
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import av

MAX_PIXELS = 3840 * 2160
MAX_OUTPUT_BYTES = 15 * 1024 * 1024
FORMATS = {'.mp4': 'mov', '.mov': 'mov', '.webm': 'matroska', '.avi': 'avi'}


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _chunk(name, data):
    return struct.pack('>I', len(data)) + name + data + struct.pack('>I', zlib.crc32(name + data) & 0xffffffff)


def _png(frame):
    scale = min(1., 640 / max(frame.width, frame.height))
    width, height = max(1, round(frame.width * scale)), max(1, round(frame.height * scale))
    rgb = frame.reformat(width=width, height=height, format='rgb24').to_ndarray()
    raw = b''.join(b'\x00' + row.tobytes() for row in rgb)
    return (b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + _chunk(b'IDAT', zlib.compress(raw)) + _chunk(b'IEND', b''))


def extract_frames(path, info, indices):
    """Fresh full sequential decode verifies the complete saved PTS and byte hash."""
    path = Path(path)
    count, native = info['frame_count'], info['source_timestamps_seconds']
    if (type(count) is not int or not 2 <= count <= 3600 or not isinstance(native, list)
            or len(native) != count or any(type(t) not in (int, float) or not math.isfinite(t) for t in native)
            or any(b <= a for a, b in zip(native, native[1:]))):
        raise ValueError('Invalid inspected timestamp evidence.')
    if (not isinstance(indices, list) or not 1 <= len(indices) <= 8 or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < count for i in indices)):
        raise ValueError('Choose between one and eight valid distinct frame indices.')
    if path.suffix.lower() not in FORMATS:
        raise ValueError('Unsupported recording format.')
    if file_sha256(path) != info.get('sha256'):
        raise ValueError('Recording bytes changed after inspection. Upload the recording again.')
    wanted, images = set(indices), {}
    total = 0
    with av.open(str(path), format=FORMATS[path.suffix.lower()], options={'protocol_whitelist': 'file'}) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Recording must contain exactly one video track.')
        stream = container.streams.video[0]
        stream.thread_count = 1
        width, height = stream.codec_context.width, stream.codec_context.height
        if not width or not height or width * height > MAX_PIXELS:
            raise ValueError('Recording resolution exceeds the canonical frame limit.')
        for index, frame in enumerate(container.decode(stream)):
            if index >= count or frame.width * frame.height > MAX_PIXELS:
                raise ValueError('Recording frame count or resolution changed after inspection.')
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Decoded frame has no actual presentation timestamp.')
            timestamp = float(frame.pts * frame.time_base)
            if not math.isfinite(timestamp) or abs(timestamp - native[index]) > 1e-9:
                raise ValueError('Decoded frame timestamps differ from inspection.')
            if index in wanted:
                images[str(index)] = base64.b64encode(_png(frame)).decode('ascii')
            total += 1
    if total != count or len(images) != len(wanted) or file_sha256(path) != info['sha256']:
        raise ValueError('Recording changed or requested frames could not be verified.')
    return images


def extract_frames_bounded(path, info, indices):
    """Native decoding is isolated and bounded; at most eight 640px PNGs return."""
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(path)],
                                input=json.dumps({'info': info, 'indices': indices}),
                                capture_output=True, text=True, timeout=45, check=True)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('Frame extraction exceeded 45 seconds. Try a shorter recording.') from exc
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ValueError('The frame decoder stopped unexpectedly. Try a new export.') from exc
    if len(result.stdout) > MAX_OUTPUT_BYTES:
        raise ValueError('Frame evidence exceeded the output limit.')
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError('The frame decoder returned invalid evidence.') from exc
    if 'error' in payload:
        raise ValueError(payload['error'])
    if set(payload) != {str(i) for i in indices}:
        raise ValueError('The frame decoder returned incomplete evidence.')
    return payload


if __name__ == '__main__':
    try:
        request = json.loads(sys.stdin.read(256 * 1024))
        print(json.dumps(extract_frames(Path(sys.argv[1]), request['info'], request['indices'])))
    except (av.error.FFmpegError, OSError):
        print(json.dumps({'error': 'Canonical frames could not be decoded. Upload a new export.'}))
    except (ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'error': str(exc)}))
