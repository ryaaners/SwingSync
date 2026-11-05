# Experiment registry

No training or evaluation has completed in this project as of the first audit (2026-11-04). Schema inspection, synthetic software tests, reference-repository claims, and model training are separate evidence categories.

## Planned event comparison

Status: NOT STARTED. Required next input: permitted GolfDB preprocessed clips plus verified decoder/annotation alignment. Public annotations were inspected; videos and pretrained weights were not downloaded by the data auditor.

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
