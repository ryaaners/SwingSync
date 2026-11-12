# Reproduce the local application

Use Python3.11, Node24 and uv0.9.8 on Apple Silicon macOS or Linux x86-64. The project uses POSIX file locks and is not supported on Windows. Dependency versions remain in the committed root and frontend locks. This is a local single-user application; automatic event detection, pose measurements, shot prediction and coaching remain unavailable in its ordinary flow.

## Build and run one server

From the repository root:

```sh
uv sync --locked
cd frontend
npm ci
npm run build
cd ..
uv run python scripts/serve_local.py
```

Open `http://127.0.0.1:8000`. The built interface and `/api` now share one local address; a separate Vite development server is unnecessary for this mode. The launcher fixes loopback binding and one worker. An absent frontend build produces a clear startup error. `--port`, `--frontend-dir` and `--data-dir` select a local port, built interface directory and local storage. Stop with Ctrl-C. Do not run two processes for the same storage directory. Keep the computer awake while processing.

Uploads remain under `data/local` by default and can be deleted through the app. Reports saved outside that directory remain wherever you save them. PyAV bundles decoder libraries; the FFmpeg command is unnecessary. Browser playback still depends on codec/container support; canonical decoded frame evidence remains authoritative and browser/audio synchronization is unvalidated.

The implementation follows the documented [StaticFiles](https://www.starlette.io/staticfiles/) mount and [Uvicorn](https://uvicorn.dev/settings/) programmatic configuration, verified against the installed pinned packages. [uv's locked synchronization](https://docs.astral.sh/uv/concepts/projects/sync/) refuses lock changes and installs the selected dependency subset. No project dependency or historical ML runtime is upgraded by this increment.

## Run the synthetic software demonstration

After building the interface:

```sh
uv run python scripts/demo_local.py \
  --output artifacts/local-demo-new \
  --frontend-dir frontend/dist
```

Use a new output directory. The tool creates an original, visibly labeled synthetic video and its own local data folder, starts an owned loopback server, exercises upload and decoded-frame inspection, manually marks/corrects frames, checks stale revisions, exports a report, restarts to verify persistence and checks deletion on a second copy. It stops its own server and retains the first inspection/report as evidence. It performs no model inference, downloads, training or coaching. The retained timestamps are decoded native presentation times, not values reconstructed from FPS.

The synthetic picture does not depict a golf swing. Its manual event names and timing values are scripted software fixtures, not golf labels, human reference annotations, model predictions or performance evidence. Do not describe this demonstration as the real permitted golf video/pose/coaching demonstration required by backlog005. Do not feed fabricated research provenance into the product. The research-draft handoff is reproduced separately by its clearly synthetic contract tests, or with the original real artifact and independently trusted checksum as described in [research-draft-handoff.md](research-draft-handoff.md).

Logs, generated videos, local databases and reports remain ignored under the chosen artifact directory. Interrupted or failed runs remain explicit and are not silently resumed. Inspect their recorded process/state before choosing a new attempt. A completed software demo does not unblock empirical event/crop reliability or independent pose references.

## Validation scope

The observed workstation is an Apple M1 with8GiB RAM on macOS26.3; this is observed hardware, not an established minimum. Fresh-environment/source-archive evidence and remote Linux CI status are recorded in the execution report after they actually complete. Docker is not installed on this workstation, so no container build or execution is claimed. Native CPU packaging is the verified path; container packaging remains optional work requiring an available runtime and its own validation.

## Completed execution014 evidence

Source checkpoint `4ff32c77a76083ac2628d090f8ffd19d4d32d488` was archived without datasets, models, environments, node_modules or built assets. A fresh Python base environment and frontend install were populated from prefetched package caches. All592base tests (13optional skips),9frontend tests, build and the real HTTP synthetic demo passed; Torch, MediaPipe and scikit-learn were absent. This verifies a fresh environment, not a fresh network download. The frontend build matched the working-tree bytes. Root fullML checks additionally passed690tests (1optional tabular skip).

The source-only phase used only generated media. A separate phase then supplied an existing genuine research draft, its independently trusted hash and exact paired recording externally. The fresh base app verified preview/confirmation/origin/report without MAT annotations, caches, checkpoints or ML dependencies. That is handoff software evidence, not new model inference or independent human labeling.

The fresh built interface loaded in the browser from one origin; the synthetic H264 video reached readyState4 and its canonical Top marker showed decoded index5. A report click returned HTTP200, while browser automation did not capture completed download. Report content and exact canonical PNG pixels passed the HTTP/reviewer checks; browser synchronization and completed browser download are not claimed. All owned QA servers were stopped. See [machine evidence](../autonomy/reports/local-reproduction-014.json) and [independent review](../autonomy/reports/review-014.md). Remote CI completion is recorded with the final checkpoint after actual success.
