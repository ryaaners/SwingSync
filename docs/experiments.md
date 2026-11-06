# Experiment registry

No training or evaluation has completed in this project as of the first audit (2026-11-04). Schema inspection, synthetic software tests, reference-repository claims, and model training are separate evidence categories.

## Planned event comparison

Status: FULL BENCHMARK NOT STARTED; CPU pipeline smoke completed in 003a. Reference preprocessed clips and frozen ImageNet initialization are now local; initial audit and subsequent acquisition are recorded separately. Next gate is full training/selection preparation, not missing public clip inventory. Original-source image/PTS equivalence remains unverified.

1. Fingerprint annotations and clip inventory; decode one real clip and verify ten-entry event interpretation, all eight target frames, source index offset and timestamps visually.
2. Save official fold assignments and source-grouped inner validation manifest with seed 20261104. Assert no source recording overlaps.
3. Run one small frozen-MobileNetV2 training smoke test to verify finite weighted loss, valid masks and resumable checkpoint state. A smoke test is not a benchmark result.
4. Train the frame-independent baseline and BiLSTM head with the same frozen features, splits, train-only class weighting, optimizer budget and transform provenance. Run at most one expensive job at once.
5. Select hyperparameters using internal validation only. Evaluate official outer test once per finalized experiment; preserve per-clip predictions for audit.
6. Report standard independent-argmax PCE overall and per event, absolute timing error in frames and PTS milliseconds, and latency including decode separately from network time. Report any ordered decoder in a separate column/run.

Each run must write: run ID, process identity/start time, state, config, seed, Git commit, dependency lock hash, dataset hashes, split-manifest hash, backbone provenance, device/hardware, checkpoint path/hash, optimizer and RNG state, wall time, metrics and artifact paths. Resume the same run only after checking the recorded process and checkpoint; never infer completion from a launched process.

| Model | Evidence type | Outer fold | PCE | Timing error | Latency |
| --- | --- | --- | --- | --- | --- |
| SwingSync frame baseline | Trained 20 epochs; inner selection only | Fold 1 test not evaluated | Outer unavailable | Outer unavailable | Two-train-clip CPU timing; see 006 |
| SwingSync temporal | Trained 20 epochs; inner selection only | Fold 1 test not evaluated | Outer unavailable | Outer unavailable | Two-train-clip CPU timing; see 006 |

No upstream performance numbers are used as project results. Upstream source/code details and exact tolerance are recorded in [data-contracts.md](data-contracts.md).

## Planned CaddieSet DirectionAngle research

Status: AUDIT COMPLETE; TRAINING NOT STARTED. This work is independent of uploaded-video inference. Dataset fingerprint and defects are in [data-contracts.md](data-contracts.md).

Use continuous signed DirectionAngle degrees. Pre-register one FACEON experiment first to match the paper's original view scope and avoid pooling incompatible structural missingness. Outer evaluation: leave one entire golfer out (eight folds). Inner tuning: grouped folds on remaining golfers only. Keep exact duplicate/alternate-view records inside golfer groups, document duplicates and avoid treating 1,757 rows as 1,757 independent golfers. A separate DTL analysis requires its own feature contract.

Compare a train-only club-conditioned mean with global-train-mean fallback for unseen clubs, Ridge regression, and a modest gradient boosting regressor. Pipeline numeric cleaning must use an explicit allowlist of event feature columns; exclude every ball outcome and GolferId from input. Fit imputers, scalers, categorical encoders and any column removal on inner training only; refit using the chosen inner configuration on the outer training partition. Use a small fixed grid, not repeated held-out-golfer tuning. Empty feature columns and unseen metadata require tested deterministic behavior.

Report each fold's MAE, RMSE, sample count, club support and baseline delta. Report macro average across golfers as well as pooled metrics, spread, and R² only when target variance supports it. State that golfer 7 has just 13 records overall (seven FACEON), making that fold noisy. Explain associations without presenting an intervention or correction as established cause. Failure to beat the mean is a valid result and blocks normal coaching integration.

### Integration decision

DISABLED: no paired extraction equivalence evidence; no trained outcome model; no generalization evidence. Even a successful tabular benchmark cannot by itself open this gate.

## Increment 002 — data-preparation evidence, not a model experiment

Annotations loaded from audited MAT SHA36523fce…; source-grouped manifests created for all four folds with seed20261104 and validation fraction0.2 of source groups. Deterministic SHA256 seed/source ranking uses ceil group count, not row or frame sampling.

| Outer fold | Train clips | Inner validation clips | Test clips | Train / validation / test sources |
| --- | ---: | ---: | ---: | --- |
| 1 | 840 | 210 | 350 | 348 / 88 / 144 |
| 2 | 836 | 214 | 350 | 350 / 88 / 142 |
| 3 | 839 | 211 | 350 | 340 / 86 / 154 |
| 4 | 853 | 197 | 350 | 352 / 88 / 140 |

No source overlaps partitions and all1,400 clip IDs appear once per fold manifest. Local artifacts: artifacts/golfdb/preparation-002/{fold-1.json,…,fold-4.json,summary.json}; exact manifest hashes recorded in summary. Local reference media acquired and decoded integrity inspection recorded in validation-002. Source YouTube timestamp equivalence and browser alignment remain unverified. Annotation677 anomaly is preserved; see data-contracts. Synthetic predictions in unit tests establish evaluator mechanics only and are not included in model result tables. No training/checkpoint/inference latency has completed.

## Increment 003a — completed CPU pipeline smoke (2026-11-05)

Run `smoke-003-20261105` completed locally. Evidence: autonomy/reports/smoke-003.json; raw run identity/config and checkpoints remain under ignored artifacts/training/smoke-003-20261105/. The run recorded PID/start identity, seed 20261105, base Git revision plus source/lock hashes, dataset and split fingerprints before extraction/training. A lock prevents concurrent runs sharing the training artifact root; existing run directories are refused. Checkpoints can be restored through the tested training utilities; the smoke CLI intentionally refuses to restart an existing run automatically.

Only fold 1 **training** clips 0 and 1 (462 frames) were used. The manifest is regenerated and compared exactly before selection; neither inner validation nor outer test is used. Both event heads consume the same cached1280-dimensional frozen torchvision MobileNetV2 ImageNet V2 features. Reference ImageNet initialization is not project swing-event training. Transform is deterministic resize short edge232 then center crop224, normalization per weight metadata; this differs from historical160×160 preprocessing and must be reported in any eventual benchmark. No stochastic augmentation is implemented/enabled yet.

CPU on Apple M1/8GiB, one Torch thread; CUDA unavailable, MPS detected available but not benchmarked. Each head completed five Adam steps, learning rate0.001, weighted cross entropy. Inverse frequency weights were computed solely from these selected training clips (2 examples/event, 446 background); a full training run must recompute weights from its full relevant training partition. Temporal head uses one bidirectional LSTM layer with64 hidden units/direction and packed lengths. Both checkpoints saved after step 2 and resumed for3 more steps; losses and final parameters exactly matched uninterrupted CPU continuation. Optimizer, CPU Torch/Python/NumPy RNG state and provenance are restored; GPU RNG/exact cross-device resume is not claimed.

| Head | First training loss | Fifth training loss | Exact CPU resume |
| --- | ---: | ---: | --- |
| Independent frame | 2.218995 | 1.949782 | Passed |
| BiLSTM temporal | 2.212210 | 1.774242 | Passed |

These are **training-smoke losses only**, not PCE, held-out generalization, baseline superiority, coaching validity or benchmark evidence. Total37.516 seconds including initialization/weight acquisition and resume verification; extraction/cache preparation36.014 seconds. This is not inference latency. No full real training, hyperparameter selection or outer-test evaluation has completed. The app remains model unavailable.

Next: preregister full baseline/temporal training budgets and source-grouped inner validation; sequence sampling/masks and temporally consistent augmentation if enabled; resumable full feature-cache job; only then train/select and evaluate finalized outer test. Record any change in recurrence context relative to reference64-frame evaluation separately.

The CPU smoke was repeated sequentially with warm feature caches after adding required Mac-arm64/Linux-x86-64 wheel coverage to the lock. See smoke-003-ci-lock.json for its separate identity and checkpoint hashes; the original run record is preserved. Logical losses and exact CPU continuation matched. This follow-up validates the corrected environment and cache reuse, not held-out performance.

## Increment 003b — registered full comparison and resumable preparation

Registered experiments/event-v1.json before any full head training. Hash-bound frame-v1/temporal-v1 configs fix fold1 manifest, seed 20261105,20 epochs, batch 4, Adam0.001, hidden 64/direction, CPU one thread, no augmentation and identical frozen full-clip features. Complete-epoch inner-validation PCE selects the best checkpoint, retaining the earliest epoch on ties. Full-training class counts determine imbalance weights; validation labels never set them. No learning-rate sweep or repeated outer-test tuning is registered. Further outer folds require their own preregistration; do not claim four-fold results from fold1.

The feature job selects exactly840training+210 inner-validation clips and excludes all350 outer-test clips. Durable configuration inventories source/video hashes before extraction, records model and implementation fingerprints, and uses a shared absolute expensive-work lock. It writes each completed cache row atomically and revalidates every prior row on explicit resume. Bounded/interrupted runs remain partial or interrupted. Media changes during the same attempt and corrupt/stale cache state fail closed.

Real bounded resume proof: job 184c02b4709746b79f15a8976399101f reused the two earlier caches and extracted two new clips in attempt1; attempt2 revalidated those four and extracted one additional clip. Final proof snapshot is PARTIAL5/1050, not completed preparation or training. Exact measured times, hashes, clip/source membership and frames are in autonomy/reports/cache-resume-004.json. The real training CLI was exercised against this partial job and refused before creating a head run. Full preparation launch/progress, if any, must be read from the actual job.json and process; never inferred from a command being issued.

The head driver is implemented and tested with explicitly synthetic tensors for deterministic shuffle, packed padding, complete train/validation coverage, train-only class counts, mid-epoch/epoch-boundary resume and validation-only selection. These unit-test validation scores are not real GolfDB results and do not populate the model-performance table. No full real head training or held-out metric is claimed by this increment.

## Completed preparation and real head comparison — increment 005

An independent full audit passed for all 1,050 registered train/inner-validation caches: 295,221 frames, exact decoded re-encoded PTS, correct labels/source mapping and zero outer-test inputs. Frozen catalog SHA256: `daf5f108512fc8468476722e7a751e1d95fe64e51e045a9bc95ab65df9918b9e`. Aggregate audit: `autonomy/reports/cache-completion-005`; full per-clip witness remains ignored under `artifacts/features/fold1-job-v1`.

Both heads completed the fixed 20 epochs and 4,200 Adam updates on the same 840 training clips. The frame run explicitly resumed its initial 25-batch checkpoint. Selection used only 210 inner-validation clips, retaining the earliest maximum PCE. These are **inner-validation/model-selection results**, not outer-test performance, unseen-golfer generalization or coaching evidence. No outer-test predictions were made.

| Head | Selected epoch | Inner PCE (%) | MAE frames | MAE ms | Summed attempt seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| frame | 19 | 41.96 | 26.867 | 896.339 | 144.765 |
| temporal | 20 | 64.58 | 9.938 | 331.552 | 688.022 |

Elapsed times include CPU head training, validation, persistence and preflight work; they exclude the prior feature extraction and are not video inference latency. Hardware: Apple M1 / 8 GiB, one CPU Torch thread, deterministic algorithms; MPS was available but unused.

| Event | Frame inner PCE (%) | Temporal inner PCE (%) | Frame MAE frames / ms | Temporal MAE frames / ms |
| --- | ---: | ---: | ---: | ---: |
| Address | 8.10 | 29.52 | 61.514 / 2052.292 | 19.814 / 661.070 |
| Toe-up | 47.14 | 70.95 | 26.057 / 869.293 | 5.505 / 183.657 |
| Mid-backswing | 47.14 | 66.67 | 25.043 / 835.509 | 9.857 / 328.883 |
| Top | 32.38 | 62.86 | 15.390 / 513.441 | 7.833 / 261.342 |
| Mid-downswing | 57.14 | 92.86 | 16.962 / 565.895 | 6.019 / 200.798 |
| Impact | 68.10 | 94.76 | 32.157 / 1072.890 | 6.552 / 218.592 |
| Mid-follow-through | 66.67 | 86.19 | 16.310 / 544.133 | 7.119 / 237.499 |
| Finish | 9.05 | 12.86 | 21.500 / 717.263 | 16.805 / 560.573 |

PTS errors refer to re-encoded recording time, not original-source or physical swing time. Full-clip context differs from reference 64-frame recurrent resets. The higher temporal inner PCE does not justify selecting another configuration using outer data. Address and Finish remain weak even on the selection partition. Standard independent decoding was used; no order constraints.

Final checkpoint hashes and all epoch aggregates are in `frame-training-005.json` and `temporal-training-005.json`. Use **best_model** inside the final continuation checkpoint, not last-epoch **model**, for selected-head evaluation. Independent review is in `review-005.md`. Next: finalize a separate held-out evaluation contract before test feature extraction/predictions; report reference 64-frame and full-clip protocols separately, without tuning against either test result. The app remains model unavailable.

## Execution 006 — fixed held-out software and measured CPU timing

The prospective `experiments/heldout-v1.json` was independently reviewed and committed/pushed as 641fe51 **before outer-test extraction**. SHA256 `4b4339a469dd5fe09f97b57ae4b650ce378ce1c3a3863f86b46249ad8d9d0008` binds the original registration, manifest, runtime and final selected checkpoints. It is unchanged. Reference 64-frame resets and registered full-clip context are fixed, separately reported comparisons; no context or model is selected from test scores. Protocol: [heldout-evaluation.md](heldout-evaluation.md).

The evaluator requires complete exact test-only caches, fresh decoded PTS and matching source/annotation/feature/checkpoint provenance. Actual partial 2/350 admission was rejected before inference or run creation. Bounded real preparation produced clips 10/11, 476 frames; an independent audit verified every byte/PTS/label/source/pipeline field. This is software/preparation evidence, **not a held-out score**. Full preparation is an actual resumed job; read CURRENT_STATUS and its live ledger. No outer-test prediction or aggregate result has completed in this execution.

### Actual CPU timing on fixed training clips

Apple M1 / 8 GiB, CPU one Torch thread; CUDA unavailable, MPS available but unused. Final one-pass benchmark used fixed TRAIN clips 0/1, after one 16-frame warmup. There was no simultaneous training/cache worker during measurement. No event accuracy was computed. The two-clip sample is not a population latency guarantee, minimum hardware specification or measured user-app latency.

Initialization/preflight **0.329s**; warmup **1.124s**, both excluded from measured feature throughput. Shared feature extraction took **26.867317s / 462 frames = 58.154366 ms/frame**. It includes actual sequential video decode, RGB transform and backbone, with bounded 16-frame RGB batches. Head timing is separate.

| Training clip | Frames | Feature wall seconds | Frame reference64 head ms | Temporal reference64 head ms | Temporal full-clip head ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 138 | 7.776176 | 1.221 | 4.973 | 3.129 |
| 1 | 324 | 19.091141 | 2.369 | 8.787 | 7.026 |

Per-frame decode/preprocessing and per-forward-batch quantiles, head/context times, actual PTS hashes, hardware/runtime and immutable source/input hashes are in `autonomy/reports/latency-006.json`. A composed single-variant duration sums shared feature wall plus that head; it is not a separately replayed video pipeline. Observed clip wall includes all four head/context calls. No browser/render/pose/report latency is included. One repetition and two clips cannot establish confidence bounds.

A preliminary timing invocation overlapped a helper-source edit and is excluded from final metrics. Its artifact remains ignored and hash-bound in the report. Source-integrity guards now fail closed before/after work/publication; the final run was taken after fixes and freeze, not chosen for speed. Independent review verified final output/state/source/media/PTS/timing arithmetic. 201 full ML tests pass; synthetic fixtures validate mechanics only.
