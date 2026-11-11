# Experiment registry

Historical first audit (2026-11-04): no training or evaluation had completed then. Execution 008 now records the actual completed fixed fold-1 benchmark below. Schema inspection, synthetic software tests, reference-repository claims, and model training are separate evidence categories.

## Registered event comparison

Status: registered fold-1 training, inner selection and complete held-out evaluation finished; independent replay passed. This is one official outer fold, not four-fold or unseen-golfer evaluation. Current metrics are below; historical milestone evidence remains labeled by execution. Original-source image/PTS equivalence and app-inference/coaching validation remain open.

1. Fingerprint annotations and clip inventory; decode one real clip and verify ten-entry event interpretation, all eight target frames, source index offset and timestamps visually.
2. Save official fold assignments and source-grouped inner validation manifest with seed 20261104. Assert no source recording overlaps.
3. Run one small frozen-MobileNetV2 training smoke test to verify finite weighted loss, valid masks and resumable checkpoint state. A smoke test is not a benchmark result.
4. Train the frame-independent baseline and BiLSTM head with the same frozen features, splits, train-only class weighting, optimizer budget and transform provenance. Run at most one expensive job at once.
5. Select hyperparameters using internal validation only. Evaluate official outer test once per finalized experiment; preserve per-clip predictions for audit.
6. Report standard independent-argmax PCE overall and per event, absolute timing error in frames and PTS milliseconds, and latency including decode separately from network time. Report any ordered decoder in a separate column/run.

Each run must write: run ID, process identity/start time, state, config, seed, Git commit, dependency lock hash, dataset hashes, split-manifest hash, backbone provenance, device/hardware, checkpoint path/hash, optimizer and RNG state, wall time, metrics and artifact paths. Resume the same run only after checking the recorded process and checkpoint; never infer completion from a launched process.

| Head | Evaluation context | PCE (%) | MAE frames | MAE recording ms |
| --- | --- | ---: | ---: | ---: |
| Frame baseline | reference-64-reset | 45.11 | 22.224 | 741.489 |
| Temporal | reference-64-reset | 58.79 | 11.700 | 390.368 |
| Frame baseline | full-clip | 45.11 | 22.224 | 741.489 |
| Temporal | full-clip | 64.43 | 6.212 | 207.253 |

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

No source overlaps partitions and all 1,400 clip IDs appear once per fold manifest. Local artifacts: artifacts/golfdb/preparation-002/{fold-1.json,…,fold-4.json,summary.json}; exact manifest hashes recorded in summary. Local reference media acquired and decoded integrity inspection recorded in validation-002. Source YouTube timestamp equivalence and browser alignment remain unverified. Annotation677 anomaly is preserved; see data-contracts. Synthetic predictions in unit tests establish evaluator mechanics only and are not included in model result tables. At increment 002 no training/checkpoint/inference latency had completed.

## Increment 003a — completed CPU pipeline smoke (2026-11-05)

Run `smoke-003-20261105` completed locally. Evidence: autonomy/reports/smoke-003.json; raw run identity/config and checkpoints remain under ignored artifacts/training/smoke-003-20261105/. The run recorded PID/start identity, seed 20261105, base Git revision plus source/lock hashes, dataset and split fingerprints before extraction/training. A lock prevents concurrent runs sharing the training artifact root; existing run directories are refused. Checkpoints can be restored through the tested training utilities; the smoke CLI intentionally refuses to restart an existing run automatically.

Only fold 1 **training** clips 0 and 1 (462 frames) were used. The manifest is regenerated and compared exactly before selection; neither inner validation nor outer test is used. Both event heads consume the same cached1280-dimensional frozen torchvision MobileNetV2 ImageNet V2 features. Reference ImageNet initialization is not project swing-event training. Transform is deterministic resize short edge232 then center crop224, normalization per weight metadata; this differs from historical160×160 preprocessing and must be reported in any eventual benchmark. No stochastic augmentation is implemented/enabled yet.

CPU on Apple M1/8GiB, one Torch thread; CUDA unavailable, MPS detected available but not benchmarked. Each head completed five Adam steps, learning rate0.001, weighted cross entropy. Inverse frequency weights were computed solely from these selected training clips (2 examples/event, 446 background); a full training run must recompute weights from its full relevant training partition. Temporal head uses one bidirectional LSTM layer with64 hidden units/direction and packed lengths. Both checkpoints saved after step 2 and resumed for3 more steps; losses and final parameters exactly matched uninterrupted CPU continuation. Optimizer, CPU Torch/Python/NumPy RNG state and provenance are restored; GPU RNG/exact cross-device resume is not claimed.

| Head | First training loss | Fifth training loss | Exact CPU resume |
| --- | ---: | ---: | --- |
| Independent frame | 2.218995 | 1.949782 | Passed |
| BiLSTM temporal | 2.212210 | 1.774242 | Passed |

These are **training-smoke losses only**, not PCE, held-out generalization, baseline superiority, coaching validity or benchmark evidence. Total37.516 seconds including initialization/weight acquisition and resume verification; extraction/cache preparation36.014 seconds. This is not inference latency. At this smoke milestone, full training/selection/outer evaluation had not yet completed; the later registered result is recorded in execution 008 below. The app remains unavailable for automatic inference.

Next: preregister full baseline/temporal training budgets and source-grouped inner validation; sequence sampling/masks and temporally consistent augmentation if enabled; resumable full feature-cache job; only then train/select and evaluate finalized outer test. Record any change in recurrence context relative to reference 64-frame-frame evaluation separately.

The CPU smoke was repeated sequentially with warm feature caches after adding required Mac-arm64/Linux-x86-64 wheel coverage to the lock. See smoke-003-ci-lock.json for its separate identity and checkpoint hashes; the original run record is preserved. Logical losses and exact CPU continuation matched. This follow-up validates the corrected environment and cache reuse, not held-out performance.

## Increment 003b — registered full comparison and resumable preparation

Registered experiments/event-v1.json before any full head training. Hash-bound frame-v1/temporal-v1 configs fix fold 1 manifest, seed 20261105,20 epochs, batch 4, Adam 0.001, hidden 64/direction, CPU one thread, no augmentation and identical frozen full-clip features. Complete-epoch inner-validation PCE selects the best checkpoint, retaining the earliest epoch on ties. Full-training class counts determine imbalance weights; validation labels never set them. No learning-rate sweep or repeated outer-test tuning is registered. Further outer folds require their own preregistration; do not claim four-fold results from fold 1.

The feature job selects exactly840training+210 inner-validation clips and excludes all 350 outer-test clips. Durable configuration inventories source/video hashes before extraction, records model and implementation fingerprints, and uses a shared absolute expensive-work lock. It writes each completed cache row atomically and revalidates every prior row on explicit resume. Bounded/interrupted runs remain partial or interrupted. Media changes during the same attempt and corrupt/stale cache state fail closed.

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

Initialization/preflight **0.329s**; warmup **1.124s**, both excluded from measured feature throughput. Shared feature extraction took **26.867317 s / 462 frames = 58.154366 ms/frame**. It includes actual sequential video decode, RGB transform and backbone, with bounded 16-frame RGB batches. Head timing is separate.

| Training clip | Frames | Feature wall seconds | Frame reference 64-frame head ms | Temporal reference 64-frame head ms | Temporal full-clip head ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 138 | 7.776176 | 1.221 | 4.973 | 3.129 |
| 1 | 324 | 19.091141 | 2.369 | 8.787 | 7.026 |

Per-frame decode/preprocessing and per-forward-batch quantiles, head/context times, actual PTS hashes, hardware/runtime and immutable source/input hashes are in `autonomy/reports/latency-006.json`. A composed single-variant duration sums shared feature wall plus that head; it is not a separately replayed video pipeline. Observed clip wall includes all four head/context calls. No browser/render/pose/report latency is included. One repetition and two clips cannot establish confidence bounds.

A preliminary timing invocation overlapped a helper-source edit and is excluded from final metrics. Its artifact remains ignored and hash-bound in the report. Source-integrity guards now fail closed before/after work/publication; the final run was taken after fixes and freeze, not chosen for speed. Independent review verified final output/state/source/media/PTS/timing arithmetic. 201 full ML tests pass; synthetic fixtures validate mechanics only.

## Execution 008 — completed preregistered fold-1 held-out evaluation

The first complete outer run used exactly the already selected heads and both declared contexts. No head, epoch, feature, context, threshold or configuration was selected/tuned using these test scores. Prospective contract641fe51 preceded all test feature extraction; its SHA256 remains `4b4339a469dd5fe09f97b57ae4b650ce378ce1c3a3863f86b46249ad8d9d0008`. Registration [heldout-v1.json](../experiments/heldout-v1.json) fixes CPU one thread, NumPy 2.2.5 default-argsort ties, independent class peaks, chronological occupied-frame targets and official tolerance. Reference64 identifies recurrence resets, not a reproduction of the original SwingNet architecture/preprocessing.

Training 840 clips / 348 sources, inner validation 210 / 88 and test 350 / 144 are pairwise source-disjoint. The complete test inventory has98,956 frames; all 350cache/video/annotation/source/historical-index/actualPTS signatures passed a fresh independent audit before inference. Frozen test catalog SHA256 `159ec421d82ca82d66cfe070d4272e013c65604365cf2c31fc11a57af0d66cf4`; original training catalog remained `daf5f108512fc8468476722e7a751e1d95fe64e51e045a9bc95ab65df9918b9e`. See test-cache-audit-008.json/.md. No training/validation/test frame mixing or test-derived class weights occurred.

| Head | Evaluation context | PCE (%) | MAE frames | MAE recording ms |
| --- | --- | ---: | ---: | ---: |
| Frame baseline | reference-64-reset | 45.11 | 22.224 | 741.489 |
| Temporal | reference-64-reset | 58.79 | 11.700 | 390.368 |
| Frame baseline | full-clip | 45.11 | 22.224 | 741.489 |
| Temporal | full-clip | 64.43 | 6.212 | 207.253 |

The primary reference 64-frame comparison is **45.11% → 58.79% PCE**, an observed **13.68 percentage-point** temporal-minus-frame difference on this fixed test fold. Full-clip temporal **64.43%** is a separate predeclared context matching training/inner selection; it must not replace the reference 64-frame result. Both frame contexts were independently computed; probabilities differ at some floating-point bits despite identical event peaks and metrics. No order-constrained decoder was used.

| Event | Frame, reference 64-frame PCE (%) | Temporal, reference 64-frame PCE (%) | Temporal, full-clip PCE (%) |
| --- | ---: | ---: | ---: |
| Address | 5.71 | 13.43 | 20.86 |
| Toe-up | 50.00 | 62.00 | 67.43 |
| Mid-backswing | 50.86 | 62.57 | 66.86 |
| Top | 34.29 | 50.86 | 61.71 |
| Mid-downswing | 66.29 | 87.71 | 91.14 |
| Impact | 68.00 | 90.86 | 94.29 |
| Mid-follow-through | 73.43 | 85.71 | 90.86 |
| Finish | 12.29 | 17.14 | 22.29 |

| Event | Frame, reference 64-frame MAE frames / ms | Temporal, reference 64-frame MAE frames / ms | Temporal, full-clip MAE frames / ms |
| --- | ---: | ---: | ---: |
| Address | 51.897 / 1731.505 | 34.500 / 1151.086 | 14.803 / 493.891 |
| Toe-up | 25.180 / 840.112 | 8.409 / 280.564 | 4.674 / 155.963 |
| Mid-backswing | 15.723 / 524.580 | 11.674 / 389.518 | 5.469 / 182.461 |
| Top | 13.417 / 447.661 | 9.877 / 329.544 | 4.309 / 143.754 |
| Mid-downswing | 9.583 / 319.731 | 4.171 / 139.176 | 2.020 / 67.396 |
| Impact | 33.254 / 1109.529 | 3.743 / 124.880 | 2.254 / 75.213 |
| Mid-follow-through | 10.697 / 356.896 | 4.351 / 145.185 | 2.751 / 91.798 |
| Finish | 18.040 / 601.895 | 16.874 / 562.989 | 13.414 / 447.550 |

Frame full-clip per-event metrics equal the displayed frame reference 64-frame metrics; all four full per-event tables and exact floats remain in [heldout-results-008.json](../autonomy/reports/heldout-results-008.json). Milliseconds use actual re-encoded recording PTS; they are not original YouTube time, calibrated exposure time or necessarily physical swing time. No test annotation-order anomaly exists. Clip 677's original Impact/follow-through inversion remains in TRAIN/officialfold3; it is not an outer-test row. Official chronological benchmark mapping was separately verified without repairing semantic labels.

The first evaluation attempt actually completed350-clips; no resumed or launched process was labeled completed training. Raw run `artifacts/evaluation/fold1-v1/run.json` SHA256 `93d9aba2332fd90fdb7cf77891a580550e42a38c68c2b10c0e774fa754c6bdb0`. Its19.520488 seconds measures the recorded evaluation loop after admission/preflight, including cache IO/result persistence; it is **not** decode-to-inference latency. Real CPU timing remains the separately scoped two-TRAIN-clip measurement in execution006. Explicit completed-run resume was checked without changing the run bytes or adding attempts. Do not rerun training or select a different context after these results.

Fresh independent reviewer loaded the selected best_model tensors into new model objects, used a separate batching loop and transcription of pinned official util.py, and matched **all 1,400per-clip/head/context result dictionaries** plus every aggregate/per-event field exactly. It also rechecked all row/cache/run/catalog/contract/checkpoint hashes and source semantics. Replay script/witness are ignored under `artifacts/evaluation/fold1-v1/review-independent-replay-008.{py,json}` and hash-bound in review-008.md. This replay verifies the same frozen experiment; it is not a second selection/tuning experiment.

### Post-hoc descriptive reliability diagnostics — no changed decoding

These counts were computed from already saved independent peaks after the primary evaluation. They were not preregistered performance endpoints, reliability calibration, filtering or ordered-decoding improvements, and do not change any PCE/result.

| Head/context | Strictly chronological predicted timelines | Clips with duplicate event peaks |
| --- | ---: | ---: |
| Frame, either context | 111/350 | 8/350 |
| Temporal, reference 64-frame | 279/350 | 3/350 |
| Temporal, full clip | 321/350 | 1/350 |

Weak Address/Finish PCE (temporal reference 64-frame **13.43% / 17.14%**) and nonchronological timelines prevent treating the overall score as dependable eight-event coaching input. Source groups are separated; golfers may recur across sources, so unseen-golfer generalization is not established. No statistical-significance or population-performance guarantee is claimed. Uploaded-video crop/view/time-origin compatibility, event-error calibration, pose reliability and coaching efficacy remain separate gates. No normal-app inference was enabled.

### Reproduce result packaging without model inference

The new stdlib-only packager requires an independently recorded expected raw-run SHA, complete exact test coverage, bound contract/manifest and consistent row checksums/metric arithmetic. It recomputes all aggregates, refuses partial/changed inputs and publishes exclusively. It does not load checkpoints, caches or videos. Fresh output paths are required.

```sh
uv run python -m scripts.report_event_evaluation \
  --run artifacts/evaluation/fold1-v1/run.json \
  --run-sha256 93d9aba2332fd90fdb7cf77891a580550e42a38c68c2b10c0e774fa754c6bdb0 \
  --contract experiments/heldout-v1.json \
  --manifest artifacts/golfdb/preparation-002-all/fold-1.json \
  --output artifacts/evaluation/fold1-report-copy.json --include-order-diagnostics
```

Independent cache/first-run/replay/report evidence is in execution 008 reports. Further folds need separate prospective registration before their training/testing; avoid exploratory held-out tuning against this fold.

## 009 — isolated upstream pose extraction smoke

No event training, evaluation or held-out tuning was repeated. Root ML lock and historical source/config/artifact identities remain unchanged. A separate complete pose lock (`edc1d9c8c3e6e3a794bbb0e582e25f292ce5bf3b640f11026c3cce6df9135919`) installed 20 packages on Python 3.11.14/macOS 26.3 ARM64; native import and OpenCV conversion passed. The upstream full/float16/v1 model bundle hashes to `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`, with both exact member identities and separate code/model/notices evidence. These are pretrained upstream pose weights, not project training.

One exclusively identified CPU run processed GolfDB clip 0, which belongs to TRAIN in registered fold 1 (official outer_split 3). It used the reencoded clip's actual native PTS, down-the-line view and explicitly declared operator hand/mirror/single-golfer assumptions. The macOS launcher denied all native network access; IPv4/IPv6 denial guards ran before import because the wheel declares performance telemetry. No portable Python opt-out or Linux extraction isolation is claimed.

All 138 decoded frames have one result row: 134 `single_pose` and 4 `no_pose`; no invented/interpolated points. An independent fresh decode verified every integer PTS/time base/float64 second, raster and exact half-up API clock. Completed ledger/result SHA binding, pinned assets, full lock and live adapter/launcher hashes passed. Decode plus synchronous detector calls took 3.589050 seconds for this one pass, excluding imports/preflight/native task initialization. Native thread count was not explicitly limited. This is not landmark accuracy, golfer identity validation, minimum hardware or representative app latency. Native feedback-tensor and NORM_RECT/IMAGE_DIMENSIONS warnings remain documented; empirical projection/measurement validation is pending.

The ignored run is `artifacts/pose/smoke-009-train-0/run.json`, SHA256 `772ab832647ccede39291b95b0179e54f5cf84e67c4af86980e1c1bf5691fb65`; raw result SHA256 `e972b761bc519b7544f85fc1a19224d6e9c8d39d3c2caa6d246de1ec9c432ba1`. Small aggregate evidence is in pose-smoke-009.json and pose-environment-009.json; independent review is review-009.md. Model-member redistribution attribution remains unresolved. App pose serving, automated events, advice and outcome integration remain disabled. See pose-extraction.md for setup, output and explicit recovery instructions.

## 010 — pose evidence/reference software pilot, no empirical accuracy result

Registered `experiments/pose-validation-v1.json` SHA256 `29e82013cc4de62b043976cb1098d50383f4aa6007919cb9cf96c7f7575965f5` fixes source/run/result identity, five diagnostic decoded indices 0, 37, 68, 87 and 137, anatomical IDs, pixel units and missingness/conditional agreement rules. The009 output was already observed; this is a software/annotation-method pilot, not a prospectively blinded model study. No human reference coordinates or landmark accuracy scores were generated.

The actual offline workbench authenticates the saved 138-frame pose sequence and fresh source PTS/raster, renders exact source/overlay evidence, and provides a separate clean reference page. All five canonical PNGs independently match decoded RGB. Browser QA verified scaled coordinate placement, navigation/persistence, clearing and JSON export. The test mark was cleared: the exported draft has 165 unreviewed/null slots and both reviewer attestations false. The operational scorer refuses it without publishing a result.

Three no-clobber report bundles preserve software-review corrections; no model inference, training, threshold tuning or held-out evaluation repeated. The final ignored bundle is `artifacts/pose/review-010-verified/manifest.json`, SHA256 `47d5141ec64c6ea29bebf64a2524422fcf243ac374abd54aef0c90069f9215f0`, with all nine derivation/input identities and generated file hashes. Its page/template bytes are identical to browser-tested content. Aggregate/witness evidence is pose-review-010.json; primary-source projection warning/geometry audit is pose-projection-audit-010.json/.md; fresh independent review is review-010.md.

The warning branch's public-source cause and square-ROI arithmetic are explained, with a deliberately nonsquare-ROI 40 px synthetic counterexample. This is not an observed native prediction error or proof of compiled-binary internal rectangles. Empirical coordinate/identity/measurement reliability awaits independent repeated reference annotations and broader permitted recordings under a separate prospective protocol. Advice, app pose/event serving, outcomes and source/model redistribution remain disabled. Both locks and historical extraction/training/evaluation artifact/source fingerprints stayed unchanged.

## 011 — completed grouped CaddieSet DirectionAngle comparison

Data contract/audit and fixed design committed `dcd1b0b` before actual fitting. New isolated tabular lock `ea54460e550b3528de4fceb93f03f3b7a910a75e734909bd282078123b6bf772`; root/pose locks unchanged. CSV1757×80/8golfers/69swing features, all outcomes/identity excluded from inputs. Newly audited11 literal`inf` features were explicitly mapped to missing before fitting, alongside58,917empty and11`#NAME?` cells. Six duplicates retained; all records of each golfer stayed together. Views analyzed separately, no row-random split.

Actual train-only64-row smoke completed first; then one fixed outer fold plus explicit15-fold resume completed all16. Baseline club means, Ridge alpha1/10/100 and two modest HGB configurations used grouped inner validation of the seven training golfers only. All preprocessing was fit inside the applicable training partition, HGB early stopping disabled, earliest candidate ties fixed. No test-driven changes, clipping or extra models were added. Completed resume verified unchanged bytes/twoattempts/no newfits.

Pooled MAE degrees: FACEONbaseline3.3352/Ridge4.6142/HGB3.5025; DTLbaseline3.2258/Ridge3.2355/HGB3.2587. Neither learned model improves pooled MAE. DTL macro MAE and Ridge RMSE have small gains, which must be reported alongside the worse pooled MAE and weak FACEON results; no broad improvement claim. Full per-fold MAE/RMSE/R² and population fold variation, selected configurations and severe FACEON Ridge extrapolation are in caddieset-experiments.md and caddieset-results-011.json. There are only eight independent golfers; no significance or unseen-population guarantee.

Comparison ledgerSHA `1b96b88964b9ef278e2bf1bc0482b83508d028d53a3a2808784c1006e183c2e2`; rawreportSHA `cddb4d1bd3fb54ec518bc0dc35f392223607b3d290f73c3e9b822fb6a4410632`; actualsmokeSHA `51c7de24a5304bcda8a77c091998c26eb25c521d0fab894a2fc611e959296083`. Independent raw-CSV/partition/preprocessing/prediction/aggregate review passed without refitting. This completes the scoped first tabular comparison, not serving admission. No models, CSV or row-level predictions committed. Outcomes stay separate because consistent generalization and feature equivalence are both unestablished. Read the fixed protocol; do not repeatedly tune against these held-out golfers.

## Research video-input parity — execution012

The opt-in local research runner and authenticated [proposal report](event-proposal-review.md) now work with the fixed selected heads and unchanged reference64-reset pipeline. After prospective registration b112612, two TRAIN clips completed:462frames from **one source/golfer**. Generic features, native timestamps and both heads' probabilities match the old frozen-cache path bitwise (maximum absolute difference0). This verifies same-pixel software consistency, not accuracy or crop/domain equivalence on ordinary uploads. No training or outer evaluation was repeated.

Both eight-card reports passed browser canonical-image loading/navigation checks. Clip1 is slow motion with nonchronological temporal peaks; these remain unrepaired and visible. The browser cannot play the supplied reference codec and shows an explicit canonical-image fallback; playback synchronization remains unverified. No tempo, pose, outcome or advice is inferred from proposals. The normal app's automatic model remains unavailable. Aggregate hashes, scope and limitations are in [event-input-parity-012.json](../autonomy/reports/event-input-parity-012.json). Broader permitted media/independent labels and prospective empirical admission remain necessary. Next independent software work is an authenticated opt-in draft handoff requiring explicit manual review and correction.

## Explicit research-draft handoff — execution013

The [optional handoff](research-draft-handoff.md) exports an authenticated completed research artifact to a checksum-bound local draft. The app validates the same recording bytes, full native timestamp sequence, metadata and proposal fields, previews without saving, and requires explicit replacement and reviewed/corrected manual confirmation. It revalidates on save and preserves original proposals and final frames across revision history, later edits/clears and reports. The expected hash must come independently from the trusted exporter; it is not a signature or evidence of accuracy. Reversed/duplicate phases are retained and prevent saving until manually corrected.

Existing TRAIN0/1 artifacts passed actual export/API checks with no new inference or training. Browser software QA verified preview, replacement, confirmation, reset after an edit and saved revision2; no independent human event labels were created. Tests:533base/631ML passed (13/1explicit optional skips),45new backend cases and9frontend state tests, lint and build. See [scoped evidence](../autonomy/reports/research-handoff-013.json). Manual recording-time arithmetic remains unvalidated and slow-motion dependent; automatic model, pose, outcome and advice admission remain closed. This completes the handoff software, not empirical uploaded-video reliability or the final coaching demo.
