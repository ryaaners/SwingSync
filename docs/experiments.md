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
| SwingSync frame baseline | Not trained | — | Unavailable | Unavailable | Unavailable |
| SwingSync temporal | Not trained | — | Unavailable | Unavailable | Unavailable |

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
