"""Bounded video inspection using presentation timestamps, never guessed FPS."""
from pathlib import Path
import json
import subprocess
import sys

import av

FORMATS = {".mp4": "mov", ".mov": "mov", ".webm": "matroska", ".avi": "avi"}
MAX_SECONDS = 30.0
MAX_FRAMES = 3600
MAX_PIXELS = 3840 * 2160


def inspect_video_bounded(path: Path) -> dict:
    """Isolate native decoding so malformed media cannot stall the worker forever."""
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(path)],
                                capture_output=True, text=True, timeout=45, check=True)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Video inspection exceeded 45 seconds. Try a shorter or lower-resolution export.") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("The video decoder stopped unexpectedly. Try a new export.") from exc
    payload = json.loads(result.stdout)
    if "error" in payload:
        raise ValueError(payload["error"])
    return payload


def inspect_video(path: Path) -> dict:
    timestamps: list[float] = []
    with av.open(str(path), format=FORMATS[path.suffix], options={"protocol_whitelist": "file"}) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Upload a video containing exactly one video track.")
        stream = container.streams.video[0]
        width, height = stream.codec_context.width, stream.codec_context.height
        if not width or not height or width * height > MAX_PIXELS:
            raise ValueError("Video resolution must be at most 3840 × 2160 pixels.")
        if stream.duration is not None and stream.time_base is not None:
            if float(stream.duration * stream.time_base) > MAX_SECONDS + 0.001:
                raise ValueError("Trim the video to at most 30 seconds.")
        last_duration = 0.0
        for frame in container.decode(stream):
            if frame.width * frame.height > MAX_PIXELS:
                raise ValueError("Video frames exceed the resolution limit.")
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Video has missing frame timestamps. Re-export the recording.")
            timestamp = float(frame.pts * frame.time_base)
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError("Video has non-increasing frame timestamps. Re-export the recording.")
            timestamps.append(timestamp)
            last_duration = float((getattr(frame, "duration", 0) or 0) * frame.time_base)
            if len(timestamps) > MAX_FRAMES or timestamp - timestamps[0] > MAX_SECONDS:
                raise ValueError("Video exceeds 30 seconds or 3600 frames. Trim or reduce frame rate.")
        if len(timestamps) < 2:
            raise ValueError("Video must contain at least two decodable frames.")
        duration = timestamps[-1] - timestamps[0] + (last_duration or timestamps[-1] - timestamps[-2])
        if duration > MAX_SECONDS + 0.001:
            raise ValueError("Trim the video to at most 30 seconds.")
        return {"duration_seconds": duration, "frame_count": len(timestamps), "width": width,
                "height": height, "source_timestamps_seconds": timestamps,
                "timestamps_seconds": [t - timestamps[0] for t in timestamps],
                "source_start_seconds": timestamps[0], "timestamp_method": "decoded presentation timestamps"}


if __name__ == "__main__":
    try:
        print(json.dumps(inspect_video(Path(sys.argv[1]))))
    except (av.error.FFmpegError, OSError):
        print(json.dumps({"error": "The video could not be decoded. Upload an MP4, MOV, WebM or AVI recording."}))
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
