# Increment 003b validation — 2026-11-05

## Completed software

Resumable frozen-feature dataset job with shared absolute expensive-work lock, immutable media/config/pipeline fingerprints and atomic per-clip ledgers. Full CPU head driver with all-training-label imbalance weights, deterministic whole-clip ordering, packed padding, complete-epoch inner-validation PCE selection and exact model/Adam/RNG/cursor/selection restore. Operational CLI admits only completed jobs matching the preregistered ImageNet feature pipeline and exact registered config hashes; outer-test input is forbidden. Single checkpoint/ledger crash-window recovery is covered by synthetic tests.

Registered fold1 protocol is in experiments/event-v1.json before full real head training. No augmentation is enabled;20 epochs, batch 4, lr0.001, hidden 64, seed 20261105, CPU one thread, full-clip context. This context differs from reference64-frame recurrent evaluation. Selection sees inner validation only; exact PCE ties keep earliest epoch. No outer evaluation has occurred.

## Actual bounded preparation proof

Job 184c02b4709746b79f15a8976399101f targeted1,050 train+inner-validation clips, zero outer-test clips. Attempt 1 reused 2 existing caches and extracted 2 new; ended partial 4/1050 in 35.733 seconds. Attempt 2 explicitly resumed, revalidated prior entries and extracted 1 additional; ended partial 5/1050 in 5.583 seconds. These times are data-preparation attempts, not model inference latency. See cache-resume-004.json and ignored immutable proof-004-partial.json snapshot for hashes, frames and exact identities. Reviewer independently validated all 5 cached rows, media/annotation/source/PTS semantics and the source-disjoint inventory.

The real head CLI rejected this partial job before creating artifacts/training/frame-v1, confirming that feature launch/progress does not become training. No new real head model or validation score was produced in003b. Synthetic training/validation fixtures remain software evidence only.

## Full preparation continuation — running, not completed

After tests/review, coordinator resumed the same job without a new-clip bound. Its initial full worker reached11completed rows, then coordinator verified and stopped only that owned worker to move it into isolated .tools/ml-runtime (same locked versions/feature identity). The ended attempt was marked interrupted under the shared lock; no completed row was reset. Replacement PID 21311/start01:53:08Toronto was observed running, revalidating 11 existing rows and holding the same global lock. Launch records/log are under the ignored job directory. Full 1,050 completion is not claimed. Read actual job.json/process before later work; no second expensive job may overlap. An OS process was actually launched; no persistent background agent is implied.

## Checks and review

- Base locked sync + pytest:56 passed,7 explicit optional-ML module skips; full locked ML sync + pytest:110 passed, one pre-existing Starlette/AnyIO warning.
- Ruff backend/tests/scripts:passed. Frontend clean npm ci --ignore-scripts, TypeScript/production build:passed; npm audit zero known vulnerabilities.
- Independent fresh review:79 relevant ML/CLI tests passed; full persisted bounded proof inspected. Three P2 findings resolved with regressions: shared-lock bypass, queued-media changes after inventory, and generic 1280-feature admission into registered training. See review-004.md.
- CI now runs the complete optional-ML test suite, with synthetic media/tensors only; no datasets or reference weight downloads required. Prior 6aa4321CI success observed. Current checkpoint CI result is reported after push.

## Evidence limits

Full cached dataset,20-epoch experiments, selected real models, held-out PCE/error metrics and actual serving latency remain incomplete. No pose/coaching/outcome inference enabled. Source/browser/feature-equivalence gates remain unchanged. Cached training progress is not a generalization or causal-coaching claim.
