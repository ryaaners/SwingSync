"""Single-process local service. Bind to loopback; do not expose publicly."""
import base64
import fcntl
import html
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import av
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.exceptions import RequestValidationError

from .annotations import AnnotationConflict, AnnotationUpdate, LIMITATION, validated_video_info
from .frame_images import extract_frames_bounded, file_sha256
from .store import Store
from .video import FORMATS, inspect_video_bounded

MODEL = {"status": "unavailable", "reason": "Automatic event detection, pose measurements and coaching are unavailable. You can mark decoded frames manually and review descriptive recording-time measurements."}


class UploadLimit:
    """Cap the entire multipart request, including chunked requests before parsing."""
    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        hostname = urlsplit("//" + headers.get(b"host", b"").decode("latin-1")).hostname
        origin = headers.get(b"origin")
        allowed = {"localhost", "127.0.0.1", "::1", "testserver"}
        if hostname not in allowed or (origin and urlsplit(origin.decode("latin-1")).hostname not in allowed):
            return await JSONResponse({"detail": "SwingSync accepts requests from this computer only."}, 403)(scope, receive, send)
        try:
            too_large = int(headers.get(b"content-length", b"0")) > self.limit
        except ValueError:
            too_large = True
        if too_large:
            return await JSONResponse({"detail": "Upload exceeds the file size limit."}, 413)(scope, receive, send)
        total, exceeded, replaced = 0, False, False

        async def bounded_receive():
            nonlocal total, exceeded
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.limit:
                exceeded = True
                raise ValueError("Upload exceeds the file size limit.")
            return message

        async def bounded_send(message):
            nonlocal replaced
            if exceeded:
                if not replaced:
                    replaced = True
                    await JSONResponse({"detail": "Upload exceeds the file size limit."}, 413)(scope, receive, send)
                return
            await send(message)

        await self.app(scope, bounded_receive, bounded_send)


def create_app(data_dir: Path | str | None = None, max_upload_bytes: int | None = None):
    root = Path(data_dir or os.environ.get("SWINGSYNC_DATA_DIR", "data/local")).resolve()
    limit = max_upload_bytes or int(os.environ.get("SWINGSYNC_MAX_UPLOAD_BYTES", 150 * 1024 * 1024))
    stop = threading.Event()
    wake = threading.Event()
    store = None
    frame_slots = threading.BoundedSemaphore(2)

    def worker():
        while not stop.is_set():
            pending = [a for a in store.list() if a["status"] == "queued"]
            if not pending:
                wake.wait(0.25)
                wake.clear()
                continue
            item = pending[-1]
            item.update(status="processing", progress=20)
            store.save(item)
            try:
                path = root / item["file_name"]
                before = file_sha256(path)
                item["video"] = inspect_video_bounded(path)
                if file_sha256(path) != before:
                    raise ValueError("Recording bytes changed during inspection. Upload the recording again.")
                item["video"]["sha256"] = before
                item.update(status="completed", progress=100)
            except ValueError as exc:
                item.update(status="failed", error=str(exc), progress=100)
            except (av.error.FFmpegError, OSError):
                item.update(status="failed", error="The video could not be decoded. Upload an MP4, MOV, WebM or AVI recording.", progress=100)
            except Exception:
                item.update(status="failed", error="Video inspection failed unexpectedly. Delete this upload and try a new export.", progress=100)
            store.save(item)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal store
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / "worker.lock").open("a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError("SwingSync already has a worker for this storage directory. Run one server process.")
        store = Store(root / "analyses.sqlite3")
        for temporary in root.glob("*.part"):
            temporary.unlink(missing_ok=True)
        records = store.list()
        referenced = {item.get("file_name") for item in records}
        # A crash after atomic rename but before DB save must not retain invisible uploads.
        # Only reconcile generated UUID filenames inside this app-owned directory.
        for candidate in root.iterdir():
            if candidate.suffix in FORMATS and candidate.name not in referenced:
                try:
                    generated = uuid.UUID(hex=candidate.stem).hex == candidate.stem
                except ValueError:
                    generated = False
                if generated and candidate.is_file():
                    candidate.unlink(missing_ok=True)
        for item in records:
            if item["status"] == "processing":
                item.update(status="interrupted", error="Processing stopped during an earlier session. Upload again to explicitly retry.")
                store.save(item)
        stop.clear()
        thread = threading.Thread(target=worker, name="swingsync-local-worker", daemon=True)
        thread.start()
        app.state.store = store
        try:
            yield
        finally:
            stop.set()
            wake.set()
            thread.join()
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    app = FastAPI(title="SwingSync", lifespan=lifespan)
    app.add_middleware(UploadLimit, limit=limit + 64 * 1024)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Default FastAPI errors echo input values; raw JSON NaN/Infinity cannot
        # be serialized into a valid JSON response. Return safe diagnostics only.
        if request.url.path.endswith("/events"):
            message = "Invalid event marks. Use canonical event names, whole-frame indices and a nonnegative revision."
        else:
            message = "Check the required request fields and recording options."
        return JSONResponse({"detail": message, "errors": [
            {"location": error["loc"], "type": error["type"], "message": error["msg"]}
            for error in exc.errors()]}, status_code=422)

    def find(identifier):
        item = store.get(identifier)
        if item is None:
            raise HTTPException(404, "Analysis not found.")
        return item

    def public(item):
        result = {k: v for k, v in item.items() if k != "file_name"}
        result.setdefault("annotation_revision", 0)
        result.setdefault("annotation_history", [])
        return result

    def canonical_frames(item, indices):
        try:
            validated_video_info(item.get("video"))
        except AnnotationConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        if not frame_slots.acquire(blocking=False):
            raise HTTPException(429, "Frame evidence is busy. Wait a moment and try again.", headers={"Retry-After": "1"})
        try:
            return extract_frames_bounded(root / item["file_name"], item["video"], indices)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            frame_slots.release()

    @app.get("/api/health")
    def health():
        return {"status": "ok", "mode": "video-inspection", "model": MODEL,
                "limits": {"max_upload_bytes": limit, "max_duration_seconds": 30, "max_frames": 3600}}

    @app.get("/api/analyses")
    def list_analyses():
        return [public(a) for a in store.list()]

    @app.post("/api/analyses", status_code=202)
    async def upload(file: UploadFile = File(...),
                     camera_view: Literal["face-on", "down-the-line"] = Form(...),
                     handedness: Literal["right", "left"] = Form(...),
                     club: str = Form(..., min_length=1, max_length=60), mirrored: bool = Form(False)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in FORMATS:
            await file.close()
            raise HTTPException(415, "Choose an MP4, MOV, WebM or AVI video.")
        identifier = uuid.uuid4().hex
        temporary, target = root / f"{identifier}.part", root / f"{identifier}{suffix}"
        total = 0
        try:
            with temporary.open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise HTTPException(413, "Upload exceeds the file size limit.")
                    output.write(chunk)
            if total == 0:
                raise HTTPException(400, "The uploaded video is empty.")
            temporary.rename(target)
            item = {"id": identifier, "status": "queued", "progress": 0, "error": None,
                    "created_at": datetime.now(timezone.utc).isoformat(), "file_name": target.name,
                    "metadata": {"camera_view": camera_view, "handedness": handedness, "club": club,
                                 "mirrored": mirrored, "original_filename": Path(file.filename or "video").name},
                    "video": None, "events": [], "measurements": [], "observations": [], "model": MODEL,
                    "annotation_revision": 0, "annotation_history": [], "advice": []}
            store.save(item)
            wake.set()
            return public(item)
        except Exception:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get("/api/analyses/{identifier}")
    def get_analysis(identifier: str):
        return public(find(identifier))

    @app.get("/api/analyses/{identifier}/video")
    def video(identifier: str):
        item = find(identifier)
        if item["status"] != "completed":
            raise HTTPException(409, "Video is not ready for playback.")
        path = root / item["file_name"]
        media_type = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".avi": "video/x-msvideo"}[path.suffix]
        return FileResponse(path, media_type=media_type)

    @app.put("/api/analyses/{identifier}/events")
    def save_events(identifier: str, update: AnnotationUpdate):
        current = find(identifier)
        if current.get("status") != "completed":
            raise HTTPException(409, "Complete video inspection before marking events.")
        try:
            validated_video_info(current.get("video"))
            if file_sha256(root / current["file_name"]) != current["video"]["sha256"]:
                raise AnnotationConflict("Recording bytes changed after inspection. Upload it again.")
            return public(store.update_annotations(identifier, update))
        except KeyError as exc:
            raise HTTPException(404, "Analysis not found.") from exc
        except (AnnotationConflict, OSError) as exc:
            raise HTTPException(409, str(exc) if isinstance(exc, AnnotationConflict)
                                else "Recording is unavailable. Upload it again.") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/analyses/{identifier}/frames/{frame_index}")
    def frame_image(identifier: str, frame_index: int):
        item = find(identifier)
        if item.get("status") != "completed":
            raise HTTPException(409, "Complete inspection before reviewing decoded frames.")
        try:
            info = validated_video_info(item.get("video"))
        except AnnotationConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        if not 0 <= frame_index < info["frame_count"]:
            raise HTTPException(422, "Frame index lies outside this recording.")
        images = canonical_frames(item, [frame_index])
        return Response(base64.b64decode(images[str(frame_index)], validate=True), media_type="image/png",
                        headers={"Cache-Control": "no-store", "X-SwingSync-Frame-Index": str(frame_index),
                                 "X-SwingSync-Timestamp-Seconds": str(info["timestamps_seconds"][frame_index]),
                                 "X-SwingSync-Source-Timestamp-Seconds": str(info["source_timestamps_seconds"][frame_index]),
                                 "X-SwingSync-Orientation": "coded-frame; display rotation and mirroring not applied"})

    @app.get("/api/analyses/{identifier}/report", response_class=HTMLResponse)
    def report(identifier: str):
        item = find(identifier)
        if item["status"] != "completed":
            raise HTTPException(409, "Complete video inspection before exporting a report.")
        metadata, info = item["metadata"], item["video"]
        events = item.get("events", [])
        manual = ""
        if events or item.get("annotation_history"):
            try:
                validated_video_info(info)
                if file_sha256(root / item["file_name"]) != info["sha256"]:
                    raise AnnotationConflict("Recording bytes changed after inspection. Upload it again.")
            except (AnnotationConflict, OSError) as exc:
                raise HTTPException(409, "Recording evidence is unavailable or changed. Upload it again.") from exc
            images = canonical_frames(item, [event["frame_index"] for event in events]) if events else {}
            figures = "".join(
                f'<figure><img width="320" alt="{html.escape(event["name"], quote=True)} manual decoded frame" '
                f'src="data:image/png;base64,{images[str(event["frame_index"])]}">'
                f'<figcaption>{html.escape(event["name"])} · manual Frame {event["frame_index"] + 1} '
                f'(decoded index {event["frame_index"]}) · '
                f'{event["timestamp_seconds"]:.6f}s from first frame · native PTS {event["source_timestamp_seconds"]:.6f}s</figcaption></figure>'
                for event in events)
            measurement_rows = "".join(
                f'<li>{html.escape(value["name"].replace("_", " "))}: {value["value"]:.6f} '
                f'{html.escape(value["unit"])}. {html.escape(value["reliability"])} '
                f'{html.escape(value["method"])} Reference: {html.escape(value["reference"])} '
                f'Evidence frames: {html.escape(str(value["frame_indices"]))}; interval: '
                f'{html.escape(str(value["interval_seconds"]))}s.</li>'
                for value in item.get("measurements", []))
            history_rows = "".join(
                f'<li>Manual revision {entry["revision"]} '
                f'({"current" if entry["revision"] == item.get("annotation_revision") else "superseded"}) · '
                f'{html.escape(entry["updated_at"])}: '
                + html.escape(", ".join(f'{event["name"]} at Frame {event["frame_index"] + 1} (decoded index {event["frame_index"]})' for event in entry["events"])
                              or "All event marks cleared")
                + '</li>' for entry in item.get("annotation_history", []))
            manual = (f'<h2>Manual event annotations — revision {item.get("annotation_revision", 0)}</h2>'
                      '<p>All markers were selected by the user; they are not model estimates. '
                      'Images show coded decoded orientation; display rotation and mirroring are not applied. '
                      'Browser playback synchronization is not validated. Use the decoded frames as evidence.</p>'
                      + figures + '<h2>Descriptive manual timing</h2><p>' + html.escape(LIMITATION) + '</p><ul>'
                      + measurement_rows + '</ul>'
                      + ('' if measurement_rows else '<p>Mark Address, Top and Impact to compute descriptive timing.</p>')
                      + '<h2>Manual corrections</h2><ol>' + history_rows + '</ol>')
        empty_note = ("No event estimates, movement measurements, outcome predictions or practice suggestions were generated."
                      if not events else "No automatic event estimates, pose measurements, outcome predictions or practice suggestions were generated.")
        content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>SwingSync inspection report</title>
        <style>body{{font:16px system-ui;max-width:800px;margin:50px auto;padding:20px;color:#14382c}}dt{{font-weight:bold}}dd{{margin-bottom:15px}}.notice{{padding:20px;background:#f4f1e8}}</style>
        <h1>SwingSync</h1><h2>Video inspection report</h2><p>{html.escape(item['created_at'])}</p>
        <p class="notice">{html.escape(MODEL['reason'])}</p><dl>
        <dt>Recording</dt><dd>{html.escape(metadata['original_filename'])}</dd>
        <dt>Analysis identifier</dt><dd>{html.escape(item['id'])}</dd>
        <dt>Recording SHA-256</dt><dd style="overflow-wrap:anywhere">{html.escape(info.get('sha256', 'Unavailable for this legacy inspection'))}</dd>
        <dt>Setup</dt><dd>{html.escape(metadata['camera_view'])} · {html.escape(metadata['handedness'])}-handed · {html.escape(metadata['club'])} · mirrored: {metadata['mirrored']}</dd>
        <dt>Duration</dt><dd>{info['duration_seconds']:.3f} seconds</dd>
        <dt>Frames</dt><dd>{info['frame_count']} · {info['width']} × {info['height']}</dd>
        <dt>Timestamp mapping</dt><dd>Actual decoded presentation timestamps; source start {info['source_start_seconds']:.6f} seconds.</dd>
        </dl>{manual}<p>{empty_note}</p>
        <p>Files remain on this computer until deleted from SwingSync. This report is a software inspection artifact, not a trained-model demonstration.</p></html>'''
        return HTMLResponse(content, headers={"Content-Disposition": f'attachment; filename="swingsync-{identifier}.html"'})

    @app.delete("/api/analyses/{identifier}", status_code=204)
    def delete(identifier: str):
        item = find(identifier)
        if item["status"] in {"queued", "processing"}:
            raise HTTPException(409, "Wait for inspection to finish before deleting this upload.")
        (root / item["file_name"]).unlink(missing_ok=True)
        store.delete(identifier)

    return app


app = create_app()
