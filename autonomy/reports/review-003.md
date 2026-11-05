# Independent review — increment 003a

Date: 2026-11-05. Scope: frozen MobileNetV2, shared frame/BiLSTM feature input, masked loss/padding, feature-cache identity/integrity, CPU checkpoint resume, and the real-training-partition smoke runner. Reviewer owns this report only; no implementation or Git changes and no duplicate real-data extraction/training launched.

## Review disposition

No unresolved material findings in the reviewed implementation. Approval is for this CPU software/pipeline subincrement, subject to coordinator validation and accurate final evidence scope.

## Resolved findings

[P2, resolved] Original cache identity did not fingerprint all feature-producing implementation code. `backend/swingsync/research/features.py:74–80` records the backbone class and state and hashes `features.py`, but an edit to `FrozenMobileNetV2.forward` in `models.py` (for example changing pooling) leaves these values unchanged. The old cache key is reused even though fresh feature extraction would differ. `timeline.py` can similarly change timestamp/label interpretation without invalidating the cache. The fix fingerprints `features.py`, `models.py`, `timeline.py`, `golfdb.py`, the actual backbone module, runtime forward code and Python version. New regression changes forward computation while keeping backbone state identical and confirms a different key. Independently re-read the fix and ran all feature/model/training tests: 24 passed in 2.53 seconds. Exact weights alone do not identify computation.

## Verified contracts

- Strict manifest equality against regenerated audited source-grouped partitions precedes selection; only `partitions.train.clip_ids` feed smoke extraction and class counts. No validation or outer-test inference path is called.
- Frozen backbone explicitly requests ImageNet V2 weights for the real smoke; `weights=None` is clearly labeled random initialization for software tests. Both heads consume the same feature clips.
- Valid-prefix masks exclude padded features/labels/loss; the BiLSTM uses packed sequences so backward recurrence excludes padding. Tests include NaN padding and differing batch partners.
- CPU checkpoint saves model, Adam, Python/NumPy/torch CPU RNG and provenance; strict provenance check precedes restore. Software tests independently verify equal losses, state and RNG continuation for both heads.
- Cache validation checks exact label semantics, actual strictly increasing PTS, historical indices, array hashes, no pickle, bounded archive size, and atomic no-overwrite publication.
- Independent `.venv/bin/pytest tests/test_training.py tests/test_models.py -q`: 17 passed in 2.65 seconds.
- Independent final complete `.venv/bin/pytest -q`: 82 passed in 2.57 seconds; one existing Starlette/anyio deprecation warning. Optional ML dependencies were present, so the new ML suites executed rather than skipped. New smoke preflight tests reject tampered train/test membership before creating a run, preserve existing run state, and reject an occupied artifact-root training lock.

## Actual smoke evidence at review

The coordinator alone launched `artifacts/training/smoke-003-20261105`. The reviewer observed durable launch identity before completion: PID 16282, process start recorded, seed 20261105, CPU/one thread, five steps per head, train clips 0/1, and dataset/manifest/code fingerprints.

After completion, the reviewer independently read `run.json` and verified:

- State `completed`, elapsed 37.5158 seconds, official `IMAGENET1K_V2` provenance, `outer_test_evaluated=false`, and `validated_for_serving=false`.
- All recorded implementation/dependency file hashes matched at original-run review; the later lock correction is separately verified below. Actual MAT SHA256 matches; regenerated fold-1 manifest exactly matches the saved manifest. Clips 0/1 and their source IDs are in train only; train/validation/test source sets are disjoint.
- Both cache file hashes match. Pickle-free semantic/cache validation passes; current actual clip byte hashes match cache video provenance. Feature shapes are `(138,1280)` and `(324,1280)`, totaling 462 frames. Persisted PTS remain strictly increasing; labels reproduce train-only counts `[2,2,2,2,2,2,2,2,446]`.
- All four step-2/step-5 checkpoint hashes match. Weights-only loads preserve expected steps, model identity, config, caches, dataset/manifest/code provenance, finite model tensors, nonempty optimizer state and Python/NumPy/torch CPU RNG. Five training losses are finite for both heads; the runner records exact CPU continuation/resume equality for each.

Frame loss changed from 2.21899 to 1.94978; temporal loss from 2.21221 to 1.77424. These are tiny training-smoke losses only and provide no prediction accuracy, generalization, latency benchmark or coaching claim. Reviewer performed artifact inspection only and did not retrain or repeat real feature extraction.

## CI portability correction and sequential warm-cache rerun

The coordinator reported that actual Linux x86_64 CI for initial commit `d11fce7` failed ML installation: the original lock selected the CPU-index torchvision 0.24.1 distribution with only an aarch64 wheel. This was an actual portability defect missed by local Mac tests. `pyproject.toml` now requires resolution coverage for Darwin/arm64 and Linux/x86_64. The corrected lock separates Linux x86_64 torchvision `0.24.1+cpu` from Linux aarch64 `0.24.1`, preserving Mac torch `2.9.1` and torchvision `0.24.1` wheel hashes.

Reviewer independently parsed the lock and evaluated its markers for both required targets: exactly one torch and torchvision package is selected for each, each has a SHA256-identified CPython 3.11 wheel for its required architecture. Linux x86_64 selects torch `2.9.1+cpu` and torchvision `0.24.1+cpu` with manylinux_2_28_x86_64 wheels; Mac arm64 selects its original wheels. No CUDA/NVIDIA/triton package is present. Independent offline locked Linux dry run passed (40 packages; would install the CPU-suffixed torch/torchvision pair); no environment mutation or package download occurred. Initial restricted invocation encountered uv's macOS system-configuration panic; the same read-only dry run succeeded with expanded process permissions. Actual corrected remote CI remains the coordinator's separate completion check.

Reviewer also verified `artifacts/training/smoke-003-ci-lock/run.json` is completed with PID 16915 and starts after the first run's completion. This separate sequential run records the corrected current `pyproject.toml` and `uv.lock` hashes (lock `8ce1d6674a07bf122541f555b90ad81532029b7ce343af54a78cd057fc2b26fc`), and all recorded code hashes match current files. Its two feature-cache hashes exactly match the first run. All four new checkpoints have verified hashes, steps, config/cache/code provenance, finite model state, nonempty Adam state and Python/NumPy/torch CPU RNG. The five losses per head exactly reproduce the first run and exact CPU resume is recorded. The 0.8141-second elapsed time is a warm-cache smoke duration, not fresh extraction or inference latency. No outer test or serving validation occurred. The reviewer did not launch a duplicate trainer.

## Evidence boundaries

Synthetic software tests establish software behavior only. The CPU smoke is not a benchmark or validation of event predictions. Original-source alignment, browser synchronization, outcome equivalence/generalization and coaching reliability gates remain unchanged. Full expensive training, temporal augmentation and finalized outer-test metrics are outside this subincrement and must not be marked completed by this review.
