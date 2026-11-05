# Current status — 2026-11-05

Increment 003a completed: shared frozen image features, independent frame/BiLSTM heads, provenance-aware caches and an actual resumable CPU pipeline smoke. Parent 003 remains in progress: full training, inner validation/model selection and outer-test metrics are not complete. App remains inspection-only/model unavailable.

## Verified capability and evidence

Optional torch2.9.1/torchvision0.24.1 CPU stack; frozen ImageNet V2 MobileNetV2, packed temporal head, valid-only weighted CE and device detection. Exact cache identity includes media/annotations/events, weights, deterministic transform, dependency versions, source/runtime implementation and device. Source PTS provenance remains explicit.

Run artifacts/training/smoke-003-20261105/run.json is COMPLETED, not merely launched. Seed20261105, CPU, one thread, fold 1 training clips 0/1,462 frames,5 logical optimizer steps per head plus checkpoint replay verification. Both model/Adam/RNG restores match CPU continuation exactly. Training losses finite; no held-out metric. Apple M1/8GiB observed; CUDA false,MPS true, acceleration unbenchmarked.37.516 seconds total includes initialization/extraction/checkpoint work, not inference latency. Do not enable serving from smoke artifacts.

Evidence: autonomy/reports/smoke-003.json, validation-003.md, review-003.md.82 tests pass with ML extra; ordinary locked environment56 pass / 4 explicit optional skips. Ruff and frontend clean install/typecheck/build pass. Reviewer cache-implementation identity issue resolved and independently verified persisted artifacts. Data,caches,weights,checkpoints remain ignored.

## Next highest-priority unblocked increment

Preregister full baseline/temporal configuration and budgets using grouped inner validation only. Implement resumable dataset feature preparation with per-clip progress/cache checks and safe job identity; add appropriate sequence sampling and temporally consistent augmentation only if enabled (current deterministic frozen pipeline has no augmentation). Then run baseline/temporal training one at a time, select on internal validation and evaluate finalized outer test. Record full-train class counts, exact feature/decoder/backbone/split fingerprints and all departures from reference64-frame context. Do not tune on outer test.

## Existing jobs and scheduling

At execution start actual server PIDs10684(backend8000)/11415(frontend5173) existed; no training job. Smoke PID16282 completed and released training-root lock. Always recheck PIDs/start identities before later work. User separately authorized daily continuation at1:30a.m. Toronto and this one-time00:07run; daily schedule remains active, one-time schedule was removed after completion. Schedules are not a continuously running agent; computer/app must remain available.

## Preserved gates

Original source equivalence remains blocked pending authorized originals and paired source correspondence. Re-encoded reference-clip research may proceed with disclosed provenance. Browser event synchronization with earlier audio/container start is unverified. No pose feature validation, coaching reliability, real video-to-coaching demonstration or held-out event performance yet. Outcome integration remains closed pending upstream extraction/paired feature equivalence and grouped generalization. No ML résumé performance claim is supported.

Media/weight access for local research is distinct from redistribution rights; do not publish raw clips/weights or apply upstream licenses to original code. See docs/data-contracts.md and model-card.md.

## Git

Dedicated remote https://github.com/ryaaners/SwingSync; branch agent/swingsync. Previous 6df193c remote CI passed(run41968842615). Initial d11fce7 ML CI found a Linux x86-64 torchvision wheel-coverage gap; corrected required-environments/lock and validated a sequential warm-cache smoke in artifacts/training/smoke-003-ci-lock with fresh provenance. See validation-003.md. This increment checkpoint SHA and remote verification reported in chat; inspect actual HEAD/remote/CI on resume.
