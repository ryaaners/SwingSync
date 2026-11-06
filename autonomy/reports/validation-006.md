# Validation 006 — fixed evaluation software and measured local CPU timing

Date: 2026-11-06. Backlog003e is complete;003f remains in progress. No outer-test prediction or aggregate benchmark is claimed.

## Checks

- Locked offline base `uv sync --locked`:passed. Base `uv run pytest`:107passed,9explicit optional-ML skips.
- Restored locked ML extra; `uv run --extra ml pytest`:201passed in19.36s (one existing Starlette/AnyIO deprecation warning).
- Ruff `backend tests scripts`:all checks passed.
- Frontend `npm ci && npm run build`:clean install/typecheck/bundle passed;0known vulnerabilities.
-40new synthetic tests cover reference64 tails/reset equivalence, selected-versus-final weights, source/data/PTS/tamper refusal, resume/lastrow crash recovery, timing units/stages, exclusive output and source/runtime drift. They establish software behavior, not model performance.

## Real evidence and independent review

Prospective contract independently reviewed and committed/pushed641fe51 before outer preparation; exactSHA4b4339a469dd5fe09f97b57ae4b650ce378ce1c3a3863f86b46249ad8d9d0008. Prior training sources/configs/lock/checkpoints/catalog remained immutable. No training or tuning occurred.

Test-cache job833e9ffe5592457f9257a46474c72153 produced a bounded2clip/476frame proof. Independent audit matched both NPZ/video hashes, actual fresh PTS, annotations, historical indices, labels, sources and pipeline. Actual incomplete2/350 inventory was refused by evaluation CLI before run creation/inference. Full extraction then explicitly resumed the same job in isolated runtime; a live process is not completed preparation. See heldout-preparation-006.json and CURRENT_STATUS.

Actual final TRAIN-only latency run completed462frames after one 16-frame warmup on AppleM1 / 8 GiB/CPU1thread. Shared feature wall 26.867317 s =58.154366ms/frame; head/context/startup/warmup scope separate. Independent reviewer rechecked full output/state hash, all current source/runtime/contract bindings, fresh actual mediaPTS/frame counts, timing denominators and composition arithmetic. No accuracy metric was computed. Two clips/one pass do not establish representative app latency or minimum hardware. Detailed result:latency-006.json.

Material findings fixed and independently retested: operational latency contract incorrectly compared descriptive scope; evaluator finalrow-before-summary crash recovery; live source/runtime drift could publish stale-provenance results. A preliminary real timing artifact exposed source drift; it was preserved and excluded, then final measurement used frozen corrected software. No unresolved material review finding. Review-006.md includes256independent synthetic parity cases against the official PCE algorithm and actual artifact audit.

## Boundaries

No real held-out score, four-fold result, uploaded-video inference, pose accuracy, coaching efficacy or shot-outcome integration is established. Source-original/browser synchronization and cross-platform artifact reproduction remain unverified. Datasets/media/caches/checkpoints/full operational job state remain ignored. Only coordinator commits/pushes; exact final commit/remoteSHA/CI outcome reported in chat and inspected next execution.
