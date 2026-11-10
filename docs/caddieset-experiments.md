# CaddieSet grouped tabular comparison

This is a separate research module for the upstream CSV. It does not accept video-derived features, serve shot predictions, validate coaching, or infer that changing a position causes a better shot. DirectionAngle is initial ball direction in degrees, not slice/hook curvature. Missing feature-equivalence evidence keeps video integration closed even if tabular performance improves.

## Fixed design and data admission

The coordinator registered `experiments/caddieset-v1.json` and its protocol before actual fitting. The fixed configuration SHA256 is `8fc099bc65e4b77bbab5dd88f896d27231c0f5b524e712e932e2c9f7dadb832d`; the dataset byte fingerprint is `646459b081863e244d75efe6f09fad4d34750d2fed7fde44db654e7720a9a306`. Read [the actual data contract](caddieset-contract.md) first. All 1,757 rows are retained, including six exact duplicate rows. Known empty cells, spreadsheet errors and the audited literal `inf` feature tokens become missing values; no preprocessing statistic is learned from the whole table.

Exactly the 69 audited event-prefixed swing columns plus ClubType enter learned predictors. All eight ball outcomes, their derivatives, GolferId and View are excluded from X. GolferId supplies groups only; separate FACEON and DTL models prevent structural view differences from becoming implicit feature-selection rules. Entire golfer records stay together, so potential alternate views and duplicate shots cannot cross a golfer split. There is no authoritative shot identifier; do not claim one was recovered from ball outcomes.

Each view has eight outer folds, leaving one golfer out. Each outer training set has seven golfers. GroupKFold with three non-shuffled inner folds selects hyperparameters using the unweighted mean of all seven individual validation-golfer MAEs, each contributing once. It does not average three unequal fold means or select on held-out golfers. Candidate ties select the earliest registered candidate. Saved selection rows sort golfer IDs before accumulation for deterministic resume arithmetic.

Models compared:

- Club-conditioned training mean, with the overall training mean for an unseen club.
- Ridge regression, alpha 1, 10 or 100.
- Histogram gradient boosting: registered paired leaf/L2 candidates (7,1) and (15,10), learning rate0.05,100 iterations and minimum20 samples/leaf. Early stopping is disabled so no row-random internal validation is introduced.

Numeric median imputation, retention of all-empty training columns as zero and missingness indicators are fit within each relevant inner training partition, then refit only on the complete outer training partition for final prediction. Ridge adds train-only standardization. Club encoding learns only training categories and maps unseen clubs to all-zero encoded categories. No all-data missingness selector, scaler, imputer, feature ranking or repeated held-out tuning is used.

These APIs were checked against the [scikit-learn1.7 SimpleImputer](https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html), [GroupKFold](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GroupKFold.html), [OneHotEncoder](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.OneHotEncoder.html) and [HistGradientBoostingRegressor](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) documentation. The isolated runtime pins scikit-learn1.7.2; later versions are not silently interchangeable with this experiment.

## Isolated environment and train-only smoke

Keep root event/pose locks unchanged. From the repository root:

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.tools/tabular-runtime" \
  uv sync --project tabular_runtime --locked --python 3.11
```

Use that explicit interpreter. Before a full comparison, run a fresh train-only smoke:

```sh
.tools/tabular-runtime/bin/python scripts/run_caddieset.py \
  --csv data/caddieset/CaddieSet.csv \
  --config experiments/caddieset-v1.json \
  --expected-config-sha256 8fc099bc65e4b77bbab5dd88f896d27231c0f5b524e712e932e2c9f7dadb832d \
  --run-dir artifacts/caddieset/smoke-v1 --smoke
```

The smoke fits all three models on at most32 FACEON records from each of the first two outer-training golfers (2 and3; golfer1 excluded). It checks finite predictions on those same training rows and reports no accuracy score or held-out metric. A completed smoke is software execution evidence, not a completed comparison or generalization result. Use a new directory; interrupted smoke state requires inspection rather than automatic retraining.

## Bounded run and explicit resume

After registration, audit, smoke and independent review, the coordinator can process one actual outer fold first:

```sh
.tools/tabular-runtime/bin/python scripts/run_caddieset.py \
  --csv data/caddieset/CaddieSet.csv \
  --config experiments/caddieset-v1.json \
  --expected-config-sha256 8fc099bc65e4b77bbab5dd88f896d27231c0f5b524e712e932e2c9f7dadb832d \
  --run-dir artifacts/caddieset/comparison-v1 --max-folds 1
```

A one-fold run is partial and publishes no aggregate report. Continue the same immutable identity by adding `--resume` and removing `--max-folds` (or choose a further bounded count). Never start a duplicate run because the first appears slow. The shared repository expensive-work lock is nonblocking and held throughout execution.

Before fitting, run.json records job/attempt identity, actual PID/process-start identity, seed/configuration, CSV hash, installed versions/isolated lock/root lock, source/protocol/contract hashes and the pending fold. Each completed fold is atomically checkpointed inside the same ledger, alongside its checksum, outer and inner row/group memberships, selected parameters, training-only imputer/scaler/category summaries, predictions and metric arithmetic. No separate orphan fold file can be mistaken for an admitted checkpoint. An explicit resume can repeat an interrupted in-flight fold from its beginning; model-internal boosting iterations are not checkpointed. Completed folds are validated and reused without fitting again.

Source/config/data/runtime drift fails closed before publication. Resume validates grouped memberships, predictions/metric arithmetic, baseline means, candidate selection and train-only preprocessing summaries. Failed final ledger publication removes an unbound completed report. A completed `--resume` verifies report identity and aggregates and returns without changing the ledger or fitting again. Small tabular models are not exported as product checkpoints; raw predictions and operational ledgers stay ignored.

## Reporting and limits

Only all16 exact outer folds permit report.json. For each view/model, it includes every held-out-golfer MAE/RMSE/R², pooled row-weighted metrics and macro-golfer fold mean, population standard deviation, minimum and maximum. R² is null for a constant target or too few observations. Error units are degrees of initial direction. Pooled row metrics and equal-golfer summaries answer different questions; report both and preserve per-fold variation.

Eight golfers, tiny golfer7 groups, duplicate/alternate records and strong golfer/club imbalance limit population conclusions. Compare against the baseline for each view and per golfer. Failure to improve must be reported rather than hidden by choosing a favorable view or repeatedly tuning outer results. No automatic integration gate is granted by a lower mean error; pose/video feature equivalence remains unestablished. Associations and feature explanations are not causal fixes. Every output keeps advice empty and app integration false.

Synthetic tests verify software behavior only. Actual smoke/comparison completion, hashes and measured results must be recorded by the coordinator after execution; this document does not claim that launching a process completed training.

## Completed fixed comparison — execution 011

The prospective protocol/configuration was committed and pushed as `dcd1b0b2d3dd302ffe54a8f366fe15a1070b469d` before any actual CSV fitting. The train-only smoke genuinely completed64 rows from golfers2/3, zero held-out rows and no accuracy metrics. Comparison job `ab223a3043834df6a68ba29e2ad5050f` then completed one bounded fold, explicitly resumed the remaining15, and published only after all16 view/golfer folds completed. Completed `--resume` returned byte-identical ledger/report with the same two attempts and no additional fits. No test-driven tuning, clipping, feature/target changes or model reselection occurred.

| View | Model | Pooled MAE (degrees) | Pooled RMSE | Pooled R² | Macro golfer MAE ± population SD |
| --- | --- | ---: | ---: | ---: | ---: |
| FACEON | Club mean | 3.3352 | 4.6924 | −0.1380 | 3.2945 ±1.4686 |
| FACEON | Ridge | 4.6142 | 15.2434 | −11.0090 | 4.4999 ±2.1988 |
| FACEON | HGB | 3.5025 | 4.9893 | −0.2866 | 3.3602 ±1.5191 |
| DTL | Club mean | 3.2258 | 4.5507 | −0.1451 | 3.3465 ±1.3682 |
| DTL | Ridge | 3.2355 | 4.4637 | −0.1017 | 3.3407 ±1.1667 |
| DTL | HGB | 3.2587 | 4.5782 | −0.1590 | 3.2996 ±1.3346 |

Neither learned model improves **pooled MAE** over the club-conditioned mean in either view. DTL macro MAE improves slightly (0.0058° Ridge,0.0468° HGB), and Ridge's DTL RMSE/R² improve slightly; these mixed changes do not demonstrate consistent superiority across views or weighting. HGB lowers fold MAE in4/8 FACEON golfers and5/8 DTL golfers; Ridge1/8 and4/8. Eight independent golfers, small groups, retained duplicates and club/view confounding prevent broad population or significance claims. Do not retune against these held-out results.

Retained failure diagnostic: FACEON Ridge held-out golfer1 predicts −70.72° through358.42°, with fold RMSE41.0924° and R²−376.2715. This post-completion diagnostic describes saved extrapolation, not a verified cause or a repair. Predictions were not clipped or wrapped after seeing the test outcome. Every Ridge fold selected alpha100 using inner golfers only; other selected parameters and all per-fold metrics are in the aggregate evidence. No actual fold encountered an unseen club; the fallback was exercised by synthetic tests only.

The completed comparison ledger hashes to `1b96b88964b9ef278e2bf1bc0482b83508d028d53a3a2808784c1006e183c2e2`; raw report `cddb4d1bd3fb54ec518bc0dc35f392223607b3d290f73c3e9b822fb6a4410632`. The first completed fold remained unchanged across resume. Small reproducible aggregate/provenance evidence is [caddieset-results-011.json](../autonomy/reports/caddieset-results-011.json); all per-golfer tables and independent checks are in [review-011.md](../autonomy/reports/review-011.md). Raw per-row predictions and CSV remain ignored.

Independent review verified all16 folds/1,757 rows/48 prediction vectors, arithmetic and aggregation, all2,208 final training-feature imputer statistics, missing indicators, Ridge scalers, club encoders, baseline train means and inner grouping/selection. No actual-data reviewer refit was needed; synthetic software tests did fit fixture models. CPU provenance: Apple M1/8GiB, Python3.11.14, separately pinned stack; fit-loop elapsed1.013s for the first fold plus16.478s for the remaining15, excluding preflight/import/admission. This is a single small experiment's execution evidence, not inference latency or minimum hardware. The source/config/lock identities are fixed; cross-platform exact result reproduction has not been performed.

**Video outcome serving remains disabled.** No consistent baseline gain was established, and the upstream-video/extractor/formula/coordinate/event/handedness equivalence gate is independently closed. This completed negative/mixed research comparison fulfills the scoped first tabular experiment; it does not manufacture an end-to-end coaching pipeline. Feature associations are not causal swing fixes.
