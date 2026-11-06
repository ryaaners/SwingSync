# Current status — 2026-11-06

Execution 006 completed the fixed held-out evaluator and measured local CPU timing (backlog 003e). Full outer-test preparation/evaluation (003f), parent 003 and pose 004 remain in progress. The local app remains inspection-only / model unavailable. No outer-test prediction, aggregate benchmark or coaching efficacy is claimed.

## ACTIVE JOB — inspect before expensive work

Job `833e9ffe5592457f9257a46474c72153`: `artifacts/features/fold1-test-job-v1/job.json`. Actual worker **PID 40807**, start identity `Fri Nov  6 12:38:35 2026` (Toronto), command `.tools/ml-runtime/bin/python -m scripts.prepare_feature_cache ... --partitions test --allow-test --device cpu --batch-size 16 --resume`. Observed **running 15/350** at `2026-11-06T17:41:24.308112+00:00`; this is a snapshot, read actual state. The shared absolute lock is `artifacts/training/active-training.lock`. Do not duplicate extraction, run training/evaluation/timing concurrently, or change its isolated runtime or frozen feature/cache implementation.

Launch record: `artifacts/features/fold1-test-job-v1/launch-full-006.json`; log: `full-preparation-006.log`. The bounded first attempt genuinely completed 2 clips (IDs 10/11, 476 frames), independently audited. Explicit resume continues the same immutable job without repeating those caches. Launch and partial preparation are not completed preparation or model evaluation. Verify actual command/start identity as PIDs can be reused. The computer must remain awake/available.

The prospective held-out contract was committed/pushed as **641fe51** before any test extraction. Contract SHA256 `4b4339a469dd5fe09f97b57ae4b650ce378ce1c3a3863f86b46249ad8d9d0008`; do not edit it. The separate test cache directory is `artifacts/features/fold1-test-v1`; do not mix it into the completed training catalog.

## Completed training — do not retrain or retune

Both registered heads completed 20 epochs / 4,200 updates in execution005. Frame selected epoch 19, final continuation SHA256 `8203c5f5d2d84d2fb8519dffa5d053932cfe353069f145d54b17ff28ce5a2431`; temporal selected epoch 20, final SHA256 `910f6d06a253e8117e825f5d625a1dcc11fe6db61344d169daac3fd973ceb5c3`. Ledgers: `artifacts/training/{frame,temporal}-v1/run.json`. Both old training PIDs ended. Use **best_model** in the final continuation checkpoint, especially frame whose final-epoch **model** differs.

Frozen training catalog: `artifacts/features/fold1-job-v1/completed-catalog-v1.json`, SHA256 `daf5f108512fc8468476722e7a751e1d95fe64e51e045a9bc95ab65df9918b9e`:840 train / 210 inner, 295,221 frames; independently audited, zero outer caches. Preserve immutable bytes. Selected INNER PCE 41.96% frame / 64.58% temporal is model-selection evidence, not held-out performance. Fixed seed 20261105, CPU 1 thread, same frozen ImageNetV2 features, no augmentation, fullclip context. No four-fold or unseen-golfer claim.

## New verified capability

`evaluate_event_heads` requires complete exact test-only caches and validates actual fresh PTS/source/annotation/hash/pipeline/checkpoint bindings. It reports fixed reference64-reset and separate fullclip context with official independent peaks/chronological targets/tolerance/ties, including the 677 anomaly. Atomic perclip records resume without repeating durable inference; partial runs publish no aggregates. Crash after lastrow is recoverable. Live source/runtime drift fails closed. The actual partial 2/350 job was refused before inference/run creation. See `docs/heldout-evaluation.md`, `review-006.md`, `heldout-preparation-006.json`.

Final actual TRAIN 0/1 CPU timing: 462 frames, Apple M1 / 8 GiB/one CPU thread, shared decode/transform/backbone wall 26.867317 s = **58.154366 ms/frame**. Initialization 0.329 s / warmup 1.124 s excluded and separately recorded; selected heads/contexts have separate times. This is a two-training-clip/one-pass research sample, not app/user-population latency or a hardware minimum. Raw final artifact `artifacts/latency/fold1-train-v1-final.json`, SHA256 `47cd4c4851f6cf6ddd8c464b37bd485102988bed656392a951d5a6c0cc24d72c`; summary/result `autonomy/reports/latency-006.json`. Historical preliminary timing overlapped a helper edit, was preserved and excluded; final measurement used reviewed source-integrity guards and frozen code.

Checks: locked base 107 pass / 9 explicit ML skips; locked full ML **201 pass**, Ruff backend/tests/scripts clean; frontend npm ci/typecheck/build clean and 0 known vulnerabilities. Fresh independent review resolved material findings and audited the final timing artifact. Original ML source/config/lock remained unchanged. `validation-006.md` records evidence.

## Next highest-priority increment

Inspect the active test-cache job and process/lock first. If active, do not duplicate it; independent pose-adapter work may proceed. If stopped incomplete, inspect the exact saved error/process, then explicitly resume only with identical immutable provenance. Once **all 350** test caches actually complete, independently audit/freeze its catalog and run the fixed `scripts.evaluate_event_heads` CLI exactly as README. Resume interrupted evaluation unchanged; never tune/retrain/context-select against test results. Verify complete350clip results for both heads/contexts and independently replay metrics before reporting overall/per-event PCE and frame/actualPTS-ms errors. Do not equate completed features with completed evaluation. Additional outer folds need separate preregistration.

## Preserved gates

Pure movement arithmetic is implemented with 51 synthetic tests, explicit evidence and no advice. Actual local MediaPipe extraction, empirical pose/event reliability, app event-correction UI, permitted real video-to-report coaching demo and feedback validation remain pending. Original-source PTS/image correspondence needs authorized originals; browser synchronization remains unverified. CaddieSet grouped tabular training and paired feature equivalence/generalization remain separate. Cross-platform/relocation experiment reproduction and minimum hardware are not established. No new ML résumé accuracy/coaching claim is supported. Data/rights blockers stay documented in data-contracts/backlog without repeated polling.

## Git and scheduling

Only coordinator performs Git. Dedicated `https://github.com/ryaaners/SwingSync`, `agent/swingsync`; no force push/automatic merge. Prospective contract commit641fe51 precedes test access; final software checkpoint/remote SHA/CI outcome reported in chat, inspect actual HEAD/remote/CI next execution. Datasets/media/caches/checkpoints/jobs remain ignored. Daily user-authorized 1:30 a.m. Toronto continuation stays active; the old one-time task was removed. A scheduled chat is not a persistent agent. The recorded feature job is an actual OS process and does not launch model evaluation automatically.
