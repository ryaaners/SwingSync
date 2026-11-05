# SwingSync

Local golf video inspection, with a staged path to reproducible swing-event learning and defensible movement feedback. No paid API or cloud inference.

**Current release: inspection only.** Upload a trimmed video, record camera setup, inspect real frame timestamps, play it back, export an HTML inspection report, and delete local history. There is no trained event checkpoint, pose extraction, coaching or shot prediction yet. A completed job means video inspection completed. No ML accuracy or training result is claimed.

## Setup and run

Use Python 3.11, Node 24 and [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.9.8. PyAV 16.0.1 wheels bundle FFmpeg libraries; the FFmpeg command is not required for inspection. Python 3.14 is deliberately excluded for the planned ML stack.

```sh
uv sync --locked
cd frontend
npm ci
cd ..
./scripts/dev.sh
```

Open http://127.0.0.1:5173. Backend documentation: http://127.0.0.1:8000/docs. Run one backend process. Bind both servers to loopback; this is a local single-user application without authentication. Stop with Ctrl-C.

For this workstation, project-local uv/Python were bootstrapped without needing global installations:

```sh
export PATH="$PWD/.tools/uv-env/bin:$PATH"
export UV_CACHE_DIR="$PWD/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
./scripts/dev.sh
```

## Recording and privacy

Upload one trimmed swing up to 30 seconds and 150 MiB (MP4/MOV/WebM/AVI). Keep your full body and club in frame, use a stable camera, and select face-on/down-the-line, handedness, club and mirroring. Inspection currently cannot verify that the recording contains a swing. Browser playback depends on codec support; H.264 MP4 is recommended. No transcoding is performed.

Files and SQLite history stay under `data/local` (override `SWINGSYNC_DATA_DIR`). History provides deletion after processing; it deletes the stored video and record. Exports/downloaded copies remain wherever you save them. Temporary partial uploads are cleaned after errors and startup. Interrupted active jobs are marked interrupted rather than silently re-run; queued jobs resume. Keep the computer awake while processing. Do not start two servers for the same data directory.

## Checks

```sh
uv run pytest
uv run ruff check backend tests
cd frontend && npm run build
```

Tests generate synthetic clips solely for software validation, never as ML evidence. CI checks software without datasets or retraining. See [architecture](docs/architecture.md), [data contracts](docs/data-contracts.md), [experiment ledger](docs/experiments.md), [model card](docs/model-card.md) and [saved status](autonomy/CURRENT_STATUS.md).

## Research roadmap

Next: audited GolfDB loader, source-grouped internal validation, shared frozen MobileNetV2 features with frame and temporal heads, CPU smoke training, then real training and official PCE evaluation. Pose measurements follow with view/visibility gates. CaddieSet tabular experiments are independent until feature equivalence and held-out golfer performance pass. No résumé performance bullets are justified yet.

Dataset, upstream code and weight licenses are separate; see the audit. No dataset, checkpoint or upstream code is redistributed in this increment. No license from an upstream repository is applied to original SwingSync code. A project license choice remains with the owner.
