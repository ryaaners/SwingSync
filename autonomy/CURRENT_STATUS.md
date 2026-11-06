# Current status — 2026-11-06

Increment 005 complete: full frozen-feature inventory independently audited, both registered event heads genuinely trained/selected, and a pure descriptive movement-calculation library implemented. Parent 003 and 004 remain in progress. The local web app remains inspection-only / model unavailable; no held-out benchmark or validated coaching is claimed.

## Jobs — inspect actual state before new expensive work

No preparation/head-training/backend/frontend worker remained at the final process inspection. Feature job `184c02b4709746b79f15a8976399101f` completed 1,050 clips / 295,221 frames at 2026-11-06T03:57:10UTC. Its old PID 21311 is absent. Completed frozen admission catalog: `artifacts/features/fold1-job-v1/completed-catalog-v1.json`, SHA256 `daf5f108512fc8468476722e7a751e1d95fe64e51e045a9bc95ab65df9918b9e`. Preserve these immutable bytes; do not restart extraction or overwrite the catalog.

`artifacts/training/frame-v1/run.json` completed 20 epochs / 4,200 updates at 06:37:06UTC after explicit mid-epoch resume. Final checkpoint SHA256 `8203c5f5d2d84d2fb8519dffa5d053932cfe353069f145d54b17ff28ce5a2431`; selected epoch 19 is **best_model**, distinct from last-epoch **model**.

`artifacts/training/temporal-v1/run.json` completed 20 epochs / 4,200 updates at 06:50:54UTC, PID 36179 ended. Final SHA256 `910f6d06a253e8117e825f5d625a1dcc11fe6db61344d169daac3fd973ceb5c3`; selected epoch 20. No expensive training remains active. Verify actual process/start identity, ledger and shared `artifacts/training/active-training.lock` before future work; do not duplicate completed runs. Keep datasets/caches/checkpoints ignored. Isolated `.tools/ml-runtime` remains available; no scheduled chat is a persistent background worker.

## Verified evidence

Independent full audit checked every selected cache/video hash, MAT label/source mapping, exact fresh decoded PTS and registered feature identity: 840 train / 210 inner-validation, zero outer-test caches. Detailed 1,050-row witness is ignored, with fingerprinted aggregate report `cache-completion-005.json/.md` committed.

Real registered head training used the fixed same frozen ImageNetV2 MobileNetV2 features, fold 1 source-grouped manifest, seed 20261105, 20 epochs, batch 4, Adam 0.001, CPU 1 thread, no augmentation and full-clip context. Full-clip recurrence differs from reference 64-frame resets. Independent reviewer verified all 168 checkpoints/head and replayed all selected 210-clip INNER metrics exactly. Selected **INNER model-selection PCE: 41.96% frame / 64.58% temporal**; MAE 26.867 frames / 896.339 ms versus 9.938 frames / 331.552 ms. No outer-test predictions were made; no four-fold or unseen-golfer claim. Evidence: `frame-training-005.json`, `temporal-training-005.json`, `review-005.md`, `docs/experiments.md`.

Movement library computes six descriptive results from supplied landmarks/events only: two durations, ratio, normalized sampled head movement and projected lead/trail elbow angles. It preserves real PTS and manual/predicted evidence, handles view/handedness/mirror/aspect/visibility, returns unavailable reasons and always suppresses advice. 51 synthetic tests validate arithmetic/contracts, not empirical pose accuracy. No MediaPipe adapter, app correction UI or CaddieSet equivalence is established.

Locked base install passed: 107 tests / 7 explicit optional ML skips. Locked ML environment: 161 tests pass; Ruff backend/tests/scripts clean. Frontend clean install/typecheck/build pass; npm audit zero known vulnerabilities. Fresh independent review found no unresolved material issue. See `validation-005.md`.

## Next highest-priority increment

Finalize a held-out evaluation contract before extracting the 350 fold 1 test features or viewing predictions. Bind selected checkpoints, feature signature, source-grouped manifest, NumPy tie behavior, official chronological target/tolerance (including the 677 anomaly), PTS provenance and immutable evaluation configuration. Implement reference 64-frame-reset temporal evaluation and disclose the separately registered full-clip context; report each separately without test-driven tuning. Evaluate each fixed selected head once, record overall/per-event PCE and frame/PTS-ms errors, and measure actual CPU decode/backbone/head latency separately. Keep test outcomes out of training/selection. Additional folds need separate preregistration.

Independent useful work: audit/install a local MediaPipe runtime/asset in a separate compatible environment if needed, then implement timestamp-preserving pose extraction and validate on permitted media. Do not connect unvalidated geometry or advice to normal coaching. Optional CaddieSet grouped tabular research must stay separate until feature-equivalence and generalization gates pass.

## Preserved blockers and release gates

Original-source image/PTS equivalence needs authorized originals and paired correspondence; reference clips are re-encoded. Browser leading-audio/container synchronization remains unverified. Held-out event evaluation, inference latency/serving validation, empirical pose/event reliability, real permitted video-to-report coaching demonstration and CaddieSet equivalence/generalization remain incomplete. No ML résumé accuracy/coaching claim is supported. Existing precise data/rights blockers remain in data-contracts/backlog; research can proceed independently.

## Git and scheduling

Only coordinator performs Git. Dedicated `https://github.com/ryaaners/SwingSync`, `agent/swingsync`; no force push or automatic merge. Base training source is 45b96c0; lock/registered ML source stayed unchanged during both runs. Current increment commit and remote CI result are reported in chat; inspect actual HEAD/remote/CI on resume. Daily user-authorized 1:30 a.m.Toronto continuation remains active; the prior one-time task was removed. Computer must be awake/available for execution.
