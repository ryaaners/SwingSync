# Architecture — increment 1

React/TypeScript client -> same-origin `/api` (Vite development proxy) -> FastAPI -> SQLite and local video files. One lifespan-owned worker processes persisted queued jobs. Interrupted processing is marked interrupted at startup, never automatically repeated. Start exactly one Uvicorn worker. No remote inference, accounts, queues or external service.

This increment inspects video, preserves presentation timestamps, and provides playback/history/deletion/HTML report. `completed` means inspection finished, not trained swing analysis. Models, event timelines, pose measurements and coaching are unavailable. No fabricated results are emitted.

## Shared API

GET/POST `/api/analyses`; multipart `file`, `camera_view` (face-on/down-the-line), `handedness` (right/left), `club`, `mirrored`. GET `/api/analyses/{id}`, `/video`, `/report`; DELETE `/api/analyses/{id}`. GET `/api/health`.

Records contain id, status, progress, error, metadata, video, empty events/measurements/observations, and explicit model availability. Source PTS and playback-relative times must remain distinct. Browser video codec support is independent from decoder support; rejected playback must have a useful message.

## Planned research boundaries

Audited GolfDB loader -> frozen shared MobileNetV2 features -> independent frame head and bidirectional LSTM head -> standard PCE evaluation. Optional ordered decoding reported separately. Pose feature extractor and templates must suppress unreliable results. Tabular outcome research stays separate until paired feature equivalence and golfer-held-out performance pass.

## Local trust boundary

Bind to loopback only. This local single-user application has no authentication and must not be exposed to a network. Upload limits and decode limits bound resource use; use no shell-based video commands. Local files persist until deleted through history; no automatic upload elsewhere. Do not run multiple backend instances against one storage directory. More robust worker isolation/timeouts are required before broader use.
