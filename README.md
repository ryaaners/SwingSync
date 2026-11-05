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
uv run ruff check backend tests scripts
cd frontend && npm run build
```

Tests generate synthetic clips solely for software validation, never as ML evidence. CI checks software without datasets or retraining. See [architecture](docs/architecture.md), [data contracts](docs/data-contracts.md), [experiment ledger](docs/experiments.md), [model card](docs/model-card.md) and [saved status](autonomy/CURRENT_STATUS.md).

## Research roadmap

Implemented: audited immutable GolfDB loader, source-grouped inner validation manifests, reference PCE evaluator, and provenance-aware decoded frame mapping. Shared frozen MobileNetV2 features, frame/BiLSTM heads and resumable CPU smoke are implemented. Two real training clips completed five optimizer steps per head, which is not a validated event model. Next: full feature preparation, training/inner validation and official held-out PCE evaluation. Pose measurements follow with view/visibility gates. CaddieSet tabular experiments are independent until feature equivalence and held-out golfer performance pass. No résumé performance bullets are justified yet.

Dataset, upstream code and weight licenses are separate; see the audit. No dataset, checkpoint or upstream code is redistributed in this increment. No license from an upstream repository is applied to original SwingSync code. A project license choice remains with the owner.

## Prepare GolfDB research data

Obtain the annotations and preprocessed clips from the [official GolfDB repository](https://github.com/wmcnally/golfdb), observe applicable terms and keep all data outside Git. SciPy is pinned in the project lock. From the repository root:

```sh
uv run python -m scripts.golfdb_prepare \
  --annotations data/golfdb/golfDB.mat \
  --output-dir artifacts/golfdb/my-preparation \
  --archive data/golfdb/videos_160.zip \
  --clips-dir data/golfdb/videos_160 --clip-id 0 --clip-id 677
```

Replace `--clip-id 0 --clip-id 677` with `--all-clips` to validate the full extracted inventory. The archive check validates names and fingerprints; it does not extract files.

The archive and decoded-clip checks are optional; omit their flags to prepare annotations alone. Each run requires a new output directory. Output includes four official outer-fold manifests with source-grouped inner validation, exact fingerprints and evidence boundaries. The default seed is20261104. No model is trained or evaluated by this command. See data-contracts for the preserved annotation677 anomaly and exact official evaluation semantics. Original-source timestamps cannot be inferred from re-encoded clips.

## Run the model pipeline smoke

Install the optional locked ML stack (CPU wheels on Linux; macOS supports detected MPS, although this smoke deliberately uses CPU):

```sh
uv sync --locked --extra ml
uv run --extra ml pytest
uv run --extra ml python -m scripts.run_ml_smoke \
  --annotations data/golfdb/golfDB.mat \
  --manifest artifacts/golfdb/preparation-002-all/fold-1.json \
  --clips-dir data/golfdb/videos_160 \
  --run-dir artifacts/training/my-new-smoke --clips 2
```

Use a fresh run directory; the CLI refuses duplicate state. It selects only training IDs from an exactly validated manifest, records its identity before work, downloads official ImageNet V2 initialization if missing, caches frozen features, and checks exact CPU checkpoint resume for both heads. This explicit setup download needs internet once; model computation is local. Set TORCH_HOME to choose weight storage (default data/torch). Keep generated caches/checkpoints outside Git.

The completed smoke ran on an Apple M1 with 8GiB RAM. That is observed hardware, not an established minimum. CUDA/MPS availability is detected, but this lock installs CPU packages on non-macOS; a CUDA build requires a separately tested environment. Full training, held-out accuracy, inference latency and coaching validity remain unmeasured. Ordinary `uv sync --locked` keeps ML dependencies out of the inspection app; optional tests then skip explicitly. CI has a separate CPU ML-contract job without datasets or pretrained downloads. See autonomy/reports/validation-003.md and docs/experiments.md for evidence.

Supported runtime validation targets are Apple Silicon macOS and Linux x86-64. The dependency resolver requires binary wheel coverage for both. Windows execution is not supported by the current POSIX worker locks.

## Resume dataset preparation and train the registered heads

The registered first comparison is in experiments/event-v1.json and the two hashed head configurations. It fixes fold1, seed 20261105, frozen ImageNet V2 features, no augmentation, full-clip context,20 epochs/head, batch 4, learning rate0.001, and earliest-epoch ties on inner-validation PCE. Full-clip context differs from reference64-frame recurrent evaluation. No outer-test tuning is allowed.

```sh
uv run --extra ml python -m scripts.prepare_feature_cache \
  --annotations data/golfdb/golfDB.mat \
  --manifest artifacts/golfdb/preparation-002-all/fold-1.json \
  --clips-dir data/golfdb/videos_160 \
  --job-dir artifacts/features/fold1-job-v1 --max-new-clips 2
```

Use --resume for every later invocation of the same job. Omit --max-new-clips to process all remaining selected clips. The default selects all 840 training and210 inner-validation clips; outer-test clips require separate explicit permission. A bounded job is partial until all 1,050 clips are validated. Resume checks media inventory, exact model/pipeline/config fingerprints, existing cache bytes and prior process identity. Never launch a second process for a running job; inspect job.json and actual PID/start identity first. All feature/head/smoke CLIs use the same absolute repository training lock even if output directories differ. Corrupt or changed artifacts fail closed and require investigation; they are not overwritten silently.

Only after the matching cache job is completed:

```sh
uv run --extra ml python -m scripts.train_event_head \
  --config experiments/frame-v1.json \
  --annotations data/golfdb/golfDB.mat \
  --manifest artifacts/golfdb/preparation-002-all/fold-1.json \
  --cache-job artifacts/features/fold1-job-v1/job.json \
  --run-dir artifacts/training/frame-v1
```

Repeat sequentially for experiments/temporal-v1.json in its own run directory. A started run needs --resume; --max-batches bounds additional work and --checkpoint-batches controls persistence intervals (default 25). Checkpoints store model/Adam/RNG, clip order/cursor and best inner-validation selection. A single checkpoint published just before a crash but absent from the ledger is provenance-validated and adopted on resume. The driver rejects partial jobs, outer-test caches, unregistered configs/pipelines, changed artifact bytes and runtime changes. It computes weights from all training labels only. Selected weights remain gated from serving and require a separate finalized outer evaluation. This CLI does not evaluate the outer test.

For long local jobs, use an isolated locked environment so ordinary app setup cannot remove its ML dependencies: set `UV_PROJECT_ENVIRONMENT=.tools/ml-runtime`, run `uv sync --locked --extra ml`, and keep that environment setting for its `uv run --extra ml ...` commands. The current background preparation uses this isolated runtime. Keep the computer awake; inspect its recorded process and job.json before resuming. The schedule itself is not a continuously running agent.
