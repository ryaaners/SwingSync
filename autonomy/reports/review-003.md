# Independent review — increment 003a

Date: 2026-11-05. Scope: frozen MobileNetV2, shared frame/BiLSTM feature input, masked loss/padding, feature-cache identity/integrity, CPU checkpoint resume, and the real-training-partition smoke runner. Reviewer owns this report only; no implementation or Git changes and no duplicate real-data extraction/training launched.

## Review disposition

No unresolved material findings in the reviewed implementation. Approval is for this CPU software/pipeline subincrement, subject to coordinator validation and accurate final evidence scope.

## Resolved finding

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
- All recorded current implementation/dependency file hashes match. Actual MAT SHA256 matches; regenerated fold-1 manifest exactly matches the saved manifest. Clips 0/1 and their source IDs are in train only; train/validation/test source sets are disjoint.
- Both cache file hashes match. Pickle-free semantic/cache validation passes; current actual clip byte hashes match cache video provenance. Feature shapes are `(138,1280)` and `(324,1280)`, totaling 462 frames. Persisted PTS remain strictly increasing; labels reproduce train-only counts `[2,2,2,2,2,2,2,2,446]`.
- All four step-2/step-5 checkpoint hashes match. Weights-only loads preserve expected steps, model identity, config, caches, dataset/manifest/code provenance, finite model tensors, nonempty optimizer state and Python/NumPy/torch CPU RNG. Five training losses are finite for both heads; the runner records exact CPU continuation/resume equality for each.

Frame loss changed from 2.21899 to 1.94978; temporal loss from 2.21221 to 1.77424. These are tiny training-smoke losses only and provide no prediction accuracy, generalization, latency benchmark or coaching claim. Reviewer performed artifact inspection only and did not retrain or repeat real feature extraction.

## Evidence boundaries

Synthetic software tests establish software behavior only. The CPU smoke is not a benchmark or validation of event predictions. Original-source alignment, browser synchronization, outcome equivalence/generalization and coaching reliability gates remain unchanged. Full expensive training, temporal augmentation and finalized outer-test metrics are outside this subincrement and must not be marked completed by this review.
