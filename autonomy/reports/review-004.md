# Independent review 004 — increment 003b

Review date: 2026-11-05. Scope: resumable feature preparation, frozen-cache head training and inner-validation selection, experiment preregistration, and shared expensive-work locking. Implementation review is independent; reviewer changed only this report and performed no Git operations. All material findings below were resolved and independently re-reviewed. No unresolved material finding remains in the reviewed scope.

## CLI admission finding — resolved

**P2 — preregistered training CLI accepted an unregistered feature pipeline.** `scripts/train_event_head.py` initially checked only completed cache-job membership/file hashes; `HeadExperiment` appropriately permits generic shared 1280-dimensional pipelines for isolated software fixtures. Reproduction used `tests/test_experiment.py::inputs` to create synthetic 1280-dimensional caches in a temporary directory, an isolated dataset-loader fixture, config/manifest/registration hashes matching those fixture files, and a completed job ledger containing no `model_identity` or `weights_request`. `train_event_head.run(..., max_batches=1, checkpoint_batches=1)` accepted it, wrote `evidence_type=preregistered-inner-validation-head-training`, and executed one synthetic optimizer batch. This proves admission behavior only. The corrected CLI rejects missing/mismatched registered MobileNetV2 ImageNet V2/default-transform/CPU provenance and compares actual cache signatures with the registered job pipeline before training. Independent replay of the same missing-pipeline reproduction now raises `Cache job differs from registered frozen ImageNet feature pipeline`, with no run directory created. Dedicated CLI regressions cover partial jobs, absent pipeline, injected weights and changed transforms.

## Final integration check

`ruff check backend tests scripts` initially found F811 at the imported-fixture injection in `tests/test_head_cli.py`. The focused fixture annotation was corrected. Independent rerun: **all checks passed**.

## Findings identified and resolved during review

1. **P2 — smoke could bypass the shared expensive-work lock.** Initially `scripts/run_ml_smoke.py` locked `run_dir.parent/active-training.lock`, while feature preparation locked `artifacts/training/active-training.lock`. Reproduction: hold the latter lock, then invoke smoke with a run directory under a different parent; the former lock would still be acquired. All operational defaults now use the same absolute repository lock path. Fixture-specific API injection remains available for isolated software tests; it is not exposed as a CLI option. The smoke regression holds the shared lock with a different run-directory parent and verifies rejection before work.

2. **P2 — queued media could change after the initial inventory was recorded.** Initial feature preparation rebuilt per-clip identities without comparing video hashes against its durable inventory. A queued clip replaced during an earlier extraction could therefore produce a cache inconsistent with the job's inventory. Reproduction: on the first synthetic extraction, append bytes to the next queued MP4; the initial implementation would accept the new identity. `cache_job.py` now compares every queued identity against the initial inventory before cache reuse or extraction. `test_queued_media_change_after_initial_inventory_rejected` verifies failure after only the first completed clip, preserving that progress.

The head driver additionally now enforces and records one Torch CPU thread and deterministic algorithms, validates cache event/source-index/PTS semantics against audited records, and rejects runtime changes before continuation. An independent attempt to replace a cache's event indices and labels with reversed class positions, while retaining its original events and updating internal checksums, was rejected by the audited-record check.

## Evidence collected

- Read project instructions, reviewer role, current status/backlog, data and experiment contracts, preregistered JSON configurations, feature-cache implementation, head driver, and applicable tests.
- `.venv/bin/python -m pytest -q tests/test_training.py tests/test_features.py tests/test_models.py tests/test_evaluation.py`: **49 passed**.
- `.venv/bin/python -m pytest -q tests/test_cache_job.py tests/test_experiment.py tests/test_ml_smoke.py`: **18 passed** on the reviewed revision at that point.
- Final focused suite, adding the new CLI and experiment hardening tests: `.venv/bin/python -m pytest -q tests/test_cache_job.py tests/test_experiment.py tests/test_head_cli.py tests/test_ml_smoke.py tests/test_training.py tests/test_features.py tests/test_models.py tests/test_evaluation.py`: **79 passed in 3.42 seconds**.
- Independently checked both registered configuration byte hashes, registered feature-pipeline local source hashes, and exact registered pipeline identity against an existing completed smoke run's training-only cache. All agreed; no fresh extraction was performed.
- CLI run records now include their Git base identifier, runtime settings and per-attempt elapsed time; checkpoint publication/crash recovery has a synthetic regression.
- Cache tests use generated synthetic media/tiny frozen test modules. Head tests use synthetic 1280-dimensional cached tensors. These verify software behavior only.
- Attempted process inventory with `ps -axo pid,etime,command`; sandbox rejected it with `operation not permitted`. Reviewer did not launch extraction, real training, outer-test evaluation, or a long-running job.

## Independent inspection of bounded real cache preparation

Inspected ignored `artifacts/features/fold1-job-v1/job.json` after attempt 2 ended at **2026-11-05T06:43:36.175772+00:00**. Same durable job ID `184c02b4709746b79f15a8976399101f`, config SHA256 `23c9becd76a48727fd19bf61ce71b1f71d5599b5cc4d470eed7b0e05df0f4260`, and job fingerprint `c670683a50a968fdf06bdf16d22aef92846f5f9286262e48ea9a7ee87c69886b`. State is explicitly **partial: 5/1050 clips**. Attempt 1 reused two existing training caches and extracted two additional clips; attempt 2 explicitly resumed and extracted only one additional clip. Completed IDs 0–3 retain timestamps within attempt 1; ID 4 has its completion timestamp within attempt 2. Both attempts are recorded as partial, with distinct process/start identities.

Independent artifact assertions passed for:

- All five NPZ file byte hashes, pickle-free semantic/array integrity loads, cache keys, frame counts and float32 `[T,1280]` dimensions.
- All five media hashes against the initial durable inventory; actual MAT annotation hash, original events, event indices, historical source indices and source identities.
- Regenerated source-grouped fold-1 manifest; inventory is exactly its 1,050 training plus inner-validation clips, with **zero outer-test clips**. Every completed entry has correct training membership.
- Full cache model identity equals the registered feature pipeline; source/extractor implementation hashes, dependency lock, config hash and derived job fingerprint match current frozen inputs.

The reviewer only read these real artifacts; the coordinator performed the bounded extraction. A later continuation of this same job must be reported from its observed current state. This reviewed snapshot proves bounded preparation/resume, not completion of all dataset features.

No full dataset feature-preparation completion, full head training, model-selection result, held-out performance, inference latency, or coaching reliability is established by this review. The application remains model unavailable.
