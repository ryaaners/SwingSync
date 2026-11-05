# Independent review — increment 001

Review date: 2026-11-04. Fresh reviewer context; implementation and Git operations were not modified by this reviewer. Scope: inspection-only application, timestamp and lifecycle contracts, frontend integration, dependency declarations, and recorded data/model boundaries. This is not model validation or a security certification.

## Decision

No unresolved inspection-release blocker identified after re-review. The inspected application truthfully exposes inspection completion while leaving events, measurements, observations and model inference unavailable. The P2 orphan-upload persistence finding below is resolved. Future synchronized event markers remain gated on playback-origin validation.

## Findings

### Resolved P2 — a crash after upload rename could strand a recording outside history

`backend/swingsync/app.py` renames a completed `.part` upload to its UUID video name before committing its record (`upload`, currently lines 166–172). Startup only removes `.part` files (currently lines 105–106). Abrupt process termination between rename and `store.save` therefore leaves a full source recording with no history record and no API deletion path. SQLite and a filesystem rename do not form one transaction.

Reproduced independently with a temporary storage directory: created `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.mp4` without a database record, started `TestClient(create_app(root))`, then observed `GET /api/analyses == []` and `orphan.exists() == True`. This constructs the exact post-crash durable state rather than terminating the user's running service.

Recommended fix: reconcile unreferenced files at startup, restricted to the application's generated UUID filename shape and accepted extensions, while preserving unrelated files; alternatively durably record an uploading state and reconcile that state on restart. Add a focused crash-state test. Also consider the inverse deletion crash window (video removed but DB row remains), so missing-file records get an explicit recoverable state rather than an apparently completed playable result.

Resolution independently verified: startup now computes referenced filenames and removes only unreferenced files with an exact lowercase 32-hex UUID stem and an accepted generated extension. The worker/storage lock is acquired before reconciliation. `test_startup_removes_only_generated_unreferenced_media` reconstructs the crash state and verifies orphan deletion while preserving both a tracked UUID recording and an unrelated recording. The inverse deletion window remains a future robustness improvement; it leaves a visible record that can still be deleted, rather than invisible retained media.

### P2 future gate — first-video-relative time is not yet a verified browser playback origin

`backend/swingsync/video.py` retains each decoded source PTS and subtracts the first video frame PTS to produce `timestamps_seconds` (currently lines 60–63). This correctly preserves variable inter-frame spacing. It does not inspect container start or earlier audio, and the synthetic fixture starts at zero with no audio. A recording whose audio/container timeline precedes video can require an additional mapping to the browser's `currentTime` timeline.

Current inspection playback has no event markers, so this does not block that scope. Before enabling synchronized markers or frame jumps, add a fixture with nonzero video PTS and earlier audio/container start, verify actual browser seeking against visible frame identity, and store the explicit origin/offset. Do not silently treat the first-video-relative field as verified HTML media time. The coordinator has already recorded this gate in current status.

## Independently verified

- `.venv/bin/python -m pytest`: initial **6 passed**, 0.80 seconds; after the persistence fix, independently rerun: **7 passed**, 0.68 seconds. One Starlette/AnyIO deprecation warning. Tests cover synthetic VFR PTS, upload/inspection/report/playback/deletion/restart, invalid upload handling, interrupted processing, second-worker refusal, foreign host/origin rejection, and restricted orphan reconciliation.
- `.venv/bin/python -m ruff check backend tests`: **passed**.
- `frontend/npm run build`: **passed**; TypeScript check and Vite 6.4.5 production build completed with 29 transformed modules.
- Additional streamed multipart request without `Content-Length`, exceeding the middleware limit: **413**, followed by health **200**. No unbounded-body bypass observed in that check.
- Reviewed public response filtering, generated filenames rather than user-controlled storage paths, parameterized SQLite access, report HTML escaping, loopback launch configuration, host/origin checks, one-storage-worker locking, and subprocess decoder isolation with a 45-second timeout.
- UI uses the implemented routes/form fields/statuses; handles unavailable service, upload/request errors and browser codec playback errors; disables deletion while work is active. Frontend source/build review does not establish visual or accessibility correctness.
- Data contracts explicitly preserve GolfDB official outer splits and source grouping, exclude all CaddieSet outcomes/derivatives and golfer IDs as predictors, require golfer grouping and feature-equivalence gates, and state that no project model result exists. No training or performance claim was inferred from those documents.

## Limits and remaining acceptance evidence

The reviewer did not independently reacquire upstream datasets or rerun the data auditor's remote-source inspection. Audit claims were reviewed as recorded evidence with fingerprints and revisions. No real golf recording, trained checkpoint, benchmark, advice validation, or browser event synchronization was validated. Native decoder memory is not OS-limited; its time limit and pixel/frame/file bounds reduce risk but do not constitute a general hostile-media sandbox. Loopback-only, single-user deployment remains required.

`autonomy/CURRENT_STATUS.md` and `autonomy/backlog.json` were absent on the first review read, then appeared and were read before this report was written. Running-process inspection using `ps` was denied by the environment; coordinator-owned runtime verification is required. This reviewer did not run `uv sync --locked`, `npm ci`, the live-browser smoke test, or any Git operation; those remain coordinator integration evidence.
