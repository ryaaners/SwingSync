"""Single-process local service. Bind to loopback; do not expose publicly."""
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .store import Store
from .video import FORMATS, inspect_video_bounded

MODEL = {"status": "unavailable", "reason": "No validated event checkpoint is installed. This release inspects video only; event detection, pose measurements and coaching are unavailable."}


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
                item["video"] = inspect_video_bounded(root / item["file_name"])
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

    def find(identifier):
        item = store.get(identifier)
        if item is None:
            raise HTTPException(404, "Analysis not found.")
        return item

    def public(item):
        return {k: v for k, v in item.items() if k != "file_name"}

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
                    "video": None, "events": [], "measurements": [], "observations": [], "model": MODEL}
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

    @app.get("/api/analyses/{identifier}/report", response_class=HTMLResponse)
    def report(identifier: str):
        item = find(identifier)
        if item["status"] != "completed":
            raise HTTPException(409, "Complete video inspection before exporting a report.")
        metadata, info = item["metadata"], item["video"]
        content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>SwingSync inspection report</title>
        <style>body{{font:16px system-ui;max-width:800px;margin:50px auto;padding:20px;color:#14382c}}dt{{font-weight:bold}}dd{{margin-bottom:15px}}.notice{{padding:20px;background:#f4f1e8}}</style>
        <h1>SwingSync</h1><h2>Video inspection report</h2><p>{html.escape(item['created_at'])}</p>
        <p class="notice">{html.escape(MODEL['reason'])}</p><dl>
        <dt>Recording</dt><dd>{html.escape(metadata['original_filename'])}</dd>
        <dt>Setup</dt><dd>{html.escape(metadata['camera_view'])} · {html.escape(metadata['handedness'])}-handed · {html.escape(metadata['club'])} · mirrored: {metadata['mirrored']}</dd>
        <dt>Duration</dt><dd>{info['duration_seconds']:.3f} seconds</dd>
        <dt>Frames</dt><dd>{info['frame_count']} · {info['width']} × {info['height']}</dd>
        <dt>Timestamp mapping</dt><dd>Actual decoded presentation timestamps; source start {info['source_start_seconds']:.6f} seconds.</dd>
        </dl><p>No event estimates, movement measurements, outcome predictions or practice suggestions were generated.</p>
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
