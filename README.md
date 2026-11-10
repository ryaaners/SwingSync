# SwingSync

Local golf video inspection, with a staged path to reproducible swing-event learning and defensible movement feedback. No paid API or cloud inference.

**Current release: video inspection and manual event review.** Upload a trimmed video, select verified decoded frames for eight events, correct saved marks, review descriptive recording-time tempo, and export an HTML report with images and revision history. Files stay local and can be deleted. Automatic event inference, pose extraction, coaching and shot prediction remain unavailable. Real research event heads are trained and evaluated on one held-out GolfDB fold; app-inference validation remains unfinished. See [manual review and its limits](docs/manual-event-review.md).

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
cd frontend && npm test && npm run build
```

Tests generate synthetic clips solely for software validation, never as ML evidence. CI checks software without datasets or retraining. See [architecture](docs/architecture.md), [data contracts](docs/data-contracts.md), [experiment ledger](docs/experiments.md), [model card](docs/model-card.md) and [saved status](autonomy/CURRENT_STATUS.md).

## Research roadmap

Implemented: audited GolfDB annotations/source-grouped splits, frozen MobileNetV2 features, two genuinely trained event heads with reproducible inner selection, and a fixed prospective held-out evaluator. The fixed 350-clip outer evaluation completed with independent replay: reference 64-frame PCE 45.11% baseline / 58.79% temporal, and separately registered temporal full-clip 64.43%. Manual events/timing are usable independently. An isolated research pose extractor now runs locally; empirical landmark/measurement validation and app integration remain pending. The fixed CaddieSet comparison is complete and remains separate: neither learned model improved pooled MAE over the club baseline; feature equivalence is unestablished. Research-only portfolio statements must retain this fold/protocol scope; no user-video accuracy or coaching claim is justified.

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

The archive and decoded-clip checks are optional; omit their flags to prepare annotations alone. Each run requires a new output directory. Output includes four official outer-fold manifests with source-grouped inner validation, exact fingerprints and evidence boundaries. The default seed is20261104. No model is trained or evaluated by this command. See data-contracts for the preserved annotation 677 anomaly and exact official evaluation semantics. Original-source timestamps cannot be inferred from re-encoded clips.

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

The completed smoke ran on an Apple M1 with 8GiB RAM. That is observed hardware, not an established minimum. CUDA/MPS availability is detected, but this lock installs CPU packages on non-macOS; a CUDA build requires a separately tested environment. At that smoke milestone, full training and held-out performance were not yet measured. Current fold-1 training/evaluation and scoped CPU timing are recorded below; coaching validity remains unmeasured. Ordinary `uv sync --locked` keeps ML dependencies out of the inspection app; optional tests then skip explicitly. CI has a separate CPU ML-contract job without datasets or pretrained downloads. See autonomy/reports/validation-003.md and docs/experiments.md for evidence.

Supported runtime validation targets are Apple Silicon macOS and Linux x86-64. The dependency resolver requires binary wheel coverage for both. Windows execution is not supported by the current POSIX worker locks.

## Resume dataset preparation and train the registered heads

The registered first comparison is in experiments/event-v1.json and the two hashed head configurations. It fixes fold 1, seed 20261105, frozen ImageNet V2 features, no augmentation, full-clip context,20 epochs/head, batch 4, learning rate0.001, and earliest-epoch ties on inner-validation PCE. Full-clip context differs from reference 64-frame-frame recurrent evaluation. No outer-test tuning is allowed.

```sh
uv run --extra ml python -m scripts.prepare_feature_cache \
  --annotations data/golfdb/golfDB.mat \
  --manifest artifacts/golfdb/preparation-002-all/fold-1.json \
  --clips-dir data/golfdb/videos_160 \
  --job-dir artifacts/features/fold1-job-v1 --max-new-clips 2
```

Use --resume for every later invocation of the same job. Omit --max-new-clips to process all remaining selected clips. The default selects all 840 training and210 inner-validation clips; outer-test clips require separate explicit permission. A bounded job is partial until all 1,050 clips are validated. Resume checks media inventory, exact model/pipeline/config fingerprints, existing cache bytes and prior process identity. Never launch a second process for a running job; inspect job.json and actual PID/start identity first. All feature/head/smoke CLIs use the same absolute repository training lock even if output directories differ. Corrupt or changed artifacts fail closed and require investigation; they are not overwritten silently.

Only after the matching cache job is completed, preserve a byte-identical snapshot as `completed-catalog-v1.json` (copy once for a new experiment; never overwrite a catalog bound to an existing run). Use that immutable snapshot for both heads and every resume:

```sh
uv run --extra ml python -m scripts.train_event_head \
  --config experiments/frame-v1.json \
  --annotations data/golfdb/golfDB.mat \
  --manifest artifacts/golfdb/preparation-002-all/fold-1.json \
  --cache-job artifacts/features/fold1-job-v1/completed-catalog-v1.json \
  --run-dir artifacts/training/frame-v1
```

Repeat sequentially for experiments/temporal-v1.json in its own run directory. A started run needs --resume; --max-batches bounds additional work and --checkpoint-batches controls persistence intervals (default 25). Checkpoints store model/Adam/RNG, clip order/cursor and best inner-validation selection. A single checkpoint published just before a crash but absent from the ledger is provenance-validated and adopted on resume. The driver rejects partial jobs, outer-test caches, unregistered configs/pipelines, changed artifact bytes and runtime changes. It computes weights from all training labels only. Selected weights remain gated from serving and require a separate finalized outer evaluation. This CLI does not evaluate the outer test.

For long local jobs, use an isolated locked environment so ordinary app setup cannot remove its ML dependencies: set `UV_PROJECT_ENVIRONMENT=.tools/ml-runtime`, run `uv sync --locked --extra ml`, and keep that environment setting for its `uv run --extra ml ...` commands. The completed preparation and head runs used this isolated runtime. Keep the computer awake; inspect its recorded process and job.json before resuming. The schedule itself is not a continuously running agent.

## Movement calculation library

`backend/swingsync/movement` now provides explicit pose/event contracts and six descriptive calculations (tempo, normalized sampled head motion and projected elbow angles). It consumes supplied landmarks, preserves evidence and unvalidated reliability, and emits no advice. It is not yet a MediaPipe video-inference pipeline or a user-facing coaching feature. See [measurement definitions and remaining validation](docs/movement-measurements.md). No root dependency or active ML-runtime changes are required for this NumPy-only module.

## Current full-training evidence

The first source-grouped feature inventory is complete and independently audited: 1,050 training/inner-validation clips, 295,221 frames. The frame head completed the registered 20-epoch budget and selected epoch 19 at 41.96% **inner-validation PCE**. This is model-selection evidence, not a final held-out accuracy claim. The temporal head also completed 20 epochs, selecting epoch 20 at 64.58% inner-validation PCE. The complete fixed fold-1 outer result is now recorded below. The app provides manual review; automatic inference remains gated by input/measurement reliability. See docs/model-card.md, docs/experiments.md and current durable status for completed run identities and next validation gates.

## Fixed held-out evaluation and CPU timing

The reviewed prospective contract [heldout-v1.json](experiments/heldout-v1.json) binds the already selected checkpoints before test processing. See [the evaluation protocol](docs/heldout-evaluation.md) for the reference 64-frame reset and separately reported full-clip context. Both are fixed comparisons; no test-driven tuning is allowed.

After verifying the contract/checkpoint hashes and checking the shared job lock, prepare the **separate test-only** cache inventory:

```sh
uv run --extra ml python -m scripts.prepare_feature_cache \
  --annotations data/golfdb/golfDB.mat \
  --manifest artifacts/golfdb/preparation-002-all/fold-1.json \
  --clips-dir data/golfdb/videos_160 \
  --job-dir artifacts/features/fold1-test-job-v1 \
  --cache-dir artifacts/features/fold1-test-v1 \
  --partitions test --allow-test --device cpu --batch-size 16
```

Use `--resume` for an existing job after verifying its process identity; `--max-new-clips 2` bounds extraction without evaluating anything. Do not repeat the preparation or train another head while the shared lock is held. Use the isolated ML environment for long jobs as described above.

Only a completed exact test inventory is admitted:

```sh
uv run --extra ml python -m scripts.evaluate_event_heads \
  --contract experiments/heldout-v1.json \
  --annotations data/golfdb/golfDB.mat \
  --manifest artifacts/golfdb/preparation-002-all/fold-1.json \
  --cache-job artifacts/features/fold1-test-job-v1/job.json \
  --run-dir artifacts/evaluation/fold1-v1
```

Interrupted evaluation needs explicit `--resume` with identical inputs/code. Per-clip results persist without publishing aggregate scores until coverage is complete. A completed run is verified and returned on resume, without repeat inference. Partial feature preparation is not a held-out result.

The separately scoped latency command uses only fixed training clips 0/1 and requires local ImageNet weights; it never downloads them:

```sh
uv run --extra ml python -m scripts.benchmark_event_pipeline \
  --contract experiments/heldout-v1.json \
  --annotations data/golfdb/golfDB.mat \
  --manifest artifacts/golfdb/preparation-002-all/fold-1.json \
  --clips-dir data/golfdb/videos_160 \
  --output artifacts/latency/fold1-train-v1-final.json
```

Timing output is exclusive: an existing output/state is refused. Initialization, warmup, decode, preprocessing, backbone and each head/context have explicit scopes; composed variant time shares one feature pass. These research commands do not enable app inference or advice.

## Offline research pose extraction

The separate [pose environment](docs/pose-assets.md) has a complete lock and an explicitly acquired, hashed upstream model. [Extraction instructions](docs/pose-extraction.md) describe the macOS network-denied CPU launcher, operator metadata, native PTS versus integer API clock, missing/multiple-pose behavior and saved artifacts. Root ML dependencies remain unchanged. The effective Apple Silicon binary minimum is macOS 14; execution was verified on macOS 26.3. Linux CI checks imports only; verified Linux extraction isolation is still pending.

One real GolfDB training-clip smoke produced 138 rows: 134 single-pose returns and 4 missing-pose rows. This proves local execution and frame correspondence, not accuracy, identity continuity or coaching reliability. MediaPipe's wheel declares utilization telemetry, so the supported extraction launcher denies native network access and the extractor checks denial before import. Model-member redistribution attribution remains unresolved. Models, videos and raw observations stay ignored. App pose serving, advice and CaddieSet feature integration remain disabled. See [smoke evidence](autonomy/reports/pose-smoke-009.json) and the earlier [source audit](docs/pose-runtime-audit.md).

## Completed fold-1 research result

Both project-trained event heads completed the fixed held-out evaluation on350-clips from144 sources. **Reference64-reset PCE: 45.11% frame baseline / 58.79% temporal**; separately preregistered temporalfull-clip context: 64.43%. Timing error is 22.224 / 11.700 / 6.212 frames respectively. No test-driven tuning, reselection or ordered decoder was used. Full per-event/frame/actualPTS-ms metrics and provenance are in [experiments.md](docs/experiments.md), [model card](docs/model-card.md) and [verified machine report](autonomy/reports/heldout-results-008.json). This is one source-disjoint official fold, not four-fold or unseen-golfer evaluation. Weak Address/Finish and unvalidated uploaded-video/pose reliability keep automatic app inference and advice disabled.

A completed evaluation `--resume` verifies existing results without repeating durable inference. Do not restart training or tune against these test results. Pure-stdlib `scripts.report_event_evaluation` packages a known-SHA completed run without models/media; see the experiment guide for the command. Dataset, source video and weight redistribution remain separate rights gates. [Supported portfolio statements](docs/portfolio-evidence.md) distinguish completed research from pending end-to-end coaching.

## Pose review workbench and remaining work

The [offline review workbench](docs/pose-review.md) turns an authenticated completed pose artifact into paired source/overlay evidence and a separate clean-only annotation page. Its [reference schema and agreement CLI](docs/pose-validation.md) preserve anatomical IDs, exact timestamps, units, coverage and model abstentions. Empty drafts produce no score; no empirical landmark labels/accuracy or advice are supplied. The current five-frame TRAIN-clip pilot runs locally and passed browser/software checks, while normal app pose serving remains disabled.

[Remaining work](docs/remaining-work.md) tracks four remaining major workstreams: empirical pose validation, uploaded-video event admission, full product integration/demo and final reproducibility/release. The first grouped CaddieSet research comparison is complete. Initial event training/evaluation is complete. Independent reference labels and broader permitted media remain a real validation gate; uploaded-video event input compatibility can proceed independently.

## Completed CaddieSet comparison

The [fixed golfer-grouped experiment](docs/caddieset-experiments.md) completed16 outer folds across two views and eight golfers. Pooled MAE in degrees (club mean /Ridge /HGB): FACEON3.335 /4.614 /3.502; DTL3.226 /3.235 /3.259. Neither learned model improves pooled MAE; small secondary DTL changes and a severe Ridge failure are retained and disclosed. No test-driven retuning or video-outcome integration followed. The [data contract](docs/caddieset-contract.md), separate [runtime](docs/tabular-runtime.md) and [machine evidence](autonomy/reports/caddieset-results-011.json) support reproduction. Predictions remain outside normal coaching.

## Research video-input parity — execution012

The opt-in local research runner and authenticated [proposal report](docs/event-proposal-review.md) now work with the fixed selected heads and unchanged reference64-reset pipeline. After prospective registration b112612, two TRAIN clips completed:462frames from **one source/golfer**. Generic features, native timestamps and both heads' probabilities match the old frozen-cache path bitwise (maximum absolute difference0). This verifies same-pixel software consistency, not accuracy or crop/domain equivalence on ordinary uploads. No training or outer evaluation was repeated.

Both eight-card reports passed browser canonical-image loading/navigation checks. Clip1 is slow motion with nonchronological temporal peaks; these remain unrepaired and visible. The browser cannot play the supplied reference codec and shows an explicit canonical-image fallback; playback synchronization remains unverified. No tempo, pose, outcome or advice is inferred from proposals. The normal app's automatic model remains unavailable. Aggregate hashes, scope and limitations are in [event-input-parity-012.json](autonomy/reports/event-input-parity-012.json). Broader permitted media/independent labels and prospective empirical admission remain necessary. Next independent software work is an authenticated opt-in draft handoff requiring explicit manual review and correction.
