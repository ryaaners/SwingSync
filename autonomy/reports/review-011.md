# Independent review — increment 011

**Complete: no unresolved material findings in the fixed CaddieSet tabular experiment.** This clearance covers research software, completed artifact integrity and arithmetic only. Uploaded-video outcome prediction, feature equivalence, coaching reliability and causal advice remain unvalidated and disabled.

The reviewer owns this report only: no implementation edits or Git operations. No reviewer fit used actual CaddieSet rows, and no real held-out evaluation was repeated. Synthetic spy/fixture checks were software evidence only.

## Reviewed boundaries and resolved issues

Read `AGENTS.md`, `autonomy/roles/reviewer.md`, `autonomy/CURRENT_STATUS.md`, `autonomy/backlog.json`, data contracts, the CaddieSet loader, experiment driver/CLI, synthetic tests, prospective configuration/protocol, isolated runtime definition/lock, and actual smoke/comparison ledgers/reports.

The independent initial CSV audit found11 exact literal `inf` feature cells omitted by the historical contract. Before actual-data fitting, the data worker registered an explicit counted cleaning policy: only feature empty strings, exact `#NAME?` and literal lowercase `inf` become NaN; all other malformed/nonfinite tokens fail. Outcomes remain finite and never become inputs or imputation indicators. Transient synthetic numeric-underscore/int64-conversion fixture issues were corrected before the independent data test execution. These are resolved, not remaining findings.

Prelaunch review requested stronger provenance: bind the prospective protocol/contract text, require process-start identity before fitting, and durably record each in-flight fold before fitting with explicit incomplete-fold replay evidence on resume. These were implemented before real outer evaluation. No further correctness/leakage blocker was found in the frozen driver.

The coordinator supplied verified prospective commit/remote identity `dcd1b0b2d3dd302ffe54a8f366fe15a1070b469d` and successful preregistration CI42781688208, before any actual-CSV fit. This is coordinator Git/CI evidence; the reviewer performed no Git operations. The preregistration fixes all16 view/golfer folds, whole-golfer inner partitions, candidate order, seven-golfer macro-MAE selection, train-only preprocessing, baseline fallback and reporting semantics before results.

## Actual data and leakage audit

Independent standard-library parsing confirmed CSV SHA256 `646459b081863e244d75efe6f09fad4d34750d2fed7fde44db654e7720a9a306`,1757 records/80 columns,924 FACEON/833 DTL,eight golfers,six exact duplicate rows,58,917 empty cells,11 `#NAME?` and11 literal `inf` feature cells. All eight outcomes are finite/nonmissing. Independent golfer/club/eight-outcome signatures confirm803 candidate cross-view groups; these are audit evidence, never authoritative shot IDs or split/predictor inputs.

A separate cell-by-cell oracle compared all1757×69 raw feature tokens, all1757 targets, golfer/view/club fields and original CSV row IDs to the loader: every value matched; exactly58,939 feature cells become NaN. Arrays are owned/read-only. No imputation or fitting was performed for that loader parity check.

Final artifact audit reconstructs every outer training/test set directly from raw CSV view and golfer fields. All16 folds have exact ordered row coverage; each view's held-out rows cover its records exactly once. Every inner split covers only the seven outer-training golfers, has disjoint train/validation groups, and covers every training row once across three validation sets. Grouping entire golfers keeps exact duplicates and potential alternate-view pairs together. `View` selects separate analyses; only the explicit69 body features plus `ClubType` enter a learned model. All eight outcomes and `GolferId` remain excluded.

Independent synthetic spy audit used unequal golfer sizes: one outer fold produced exactly17 candidate/refit calls, zero outer-golfer rows entered any fit, inner groups were disjoint, candidate scoring matched a seven-golfer unweighted MAE oracle, and earliest exact ties won. HGB explicitly disables random early-stopping validation; every imputer/indicator/scaler/encoder is constructed and fit inside its relevant training partition.

## Completion, provenance and independent arithmetic

Actual train-only smoke `artifacts/caddieset/smoke-v1/` is completed and hash-bound:64 FACEON rows from golfers2/3,zero held-out rows,zero outer fold records,zero accuracy metrics,advice empty,integration false. Smoke run SHA256 `51c7de24a5304bcda8a77c091998c26eb25c521d0fab894a2fc611e959296083`.

Actual comparison job `ab223a3043834df6a68ba29e2ad5050f` in `artifacts/caddieset/comparison-v1/` is completed16/16. There are exactly two recorded attempts: first bounded attempt completed1 fold and remained partial without an aggregate; explicit resume began at1 and completed the remaining15. Both attempts record PID/start identity; no incomplete fold was replayed, and final in-flight identity is null. First-fold canonical SHA256 remains `0777d22645b16448da58c673be9c4fec65875f3b0e3353addc40d404d02b162c`.

A separately written standard-library oracle imported no production evaluator or sklearn and performed no fits. It verified all16 fold hashes, all1757 held-out row IDs/labels, all48 model prediction vectors and their per-fold MAE/RMSE/R², all six view/model pooled metrics, every macro mean/population standard deviation/range and all per-fold report rows. R² uses `1−SSE/SST` and is null for singleton/constant targets; actual folds have nonzero variance. It independently recomputed all2208 outer-training feature medians (69×16×2 learned models), training missing-indicator sets, club categories and Ridge scaler means/scales from raw CSV; agreement within2e−10. Every baseline club/global mean and prediction matched training-only values. All16 actual folds have zero unseen clubs; fallback behavior is covered by synthetic tests, not exercised in this real comparison.

All560 saved candidate inner-golfer MAEs are finite/nonnegative with exact seven-golfer coverage; their macro arithmetic and earliest-minimum candidate selection were independently checked. Inner predictions are not separately persisted, so this audit verifies those stored-score arithmetic/partitions and the implementation, rather than claiming a fresh inner-prediction replay. No models were refit to reconstruct missing predictions.

Every recorded source/config/lock hash matches the current reviewed bytes. Independent inspection of the actual isolated interpreter confirmed Python3.11.14,macOS26.3 arm64,the executable SHA and all14 installed package versions matching the locked recorded environment. Both comparison attempts use identical identity and source hashes; root ML lock is unchanged. Thread limit is explicitly1 in the driver; these short execution records are not an app-latency or portability benchmark.

| Reviewed input/artifact | SHA256 |
| --- | --- |
| `backend/swingsync/research/caddie_experiment.py` | `1c4db0b28895874dafaa12579b1901f3ff03592042579dad1841fd1f58432954` |
| `scripts/run_caddieset.py` | `0694a5afbed1cfba647eb838a3bbf2d2cdcff9a395f16f55d2d32a219b0e200a` |
| `backend/swingsync/research/caddieset.py` | `d6fade7760bbd664c1defc2b5bdf06da8c1f07eb30bbb1a6a7ebb3696a4f0469` |
| `experiments/caddieset-v1.json` | `8fc099bc65e4b77bbab5dd88f896d27231c0f5b524e712e932e2c9f7dadb832d` |
| `experiments/caddieset-v1.md` | `519857b1043564aa9d6a9b447c747e77ccf6a95bed8b81b43b7f4de4fa05d5c9` |
| `docs/caddieset-contract.md` | `386062aedf4750fae180f6d103309385f7985eed4f9dcbeeff18556603509a3a` |
| `tabular_runtime/uv.lock` | `ea54460e550b3528de4fceb93f03f3b7a910a75e734909bd282078123b6bf772` |
| `uv.lock` | `8ce1d6674a07bf122541f555b90ad81532029b7ce343af54a78cd057fc2b26fc` |
| `artifacts/caddieset/comparison-v1/run.json` | `1b96b88964b9ef278e2bf1bc0482b83508d028d53a3a2808784c1006e183c2e2` |
| `artifacts/caddieset/comparison-v1/report.json` | `cddb4d1bd3fb54ec518bc0dc35f392223607b3d290f73c3e9b822fb6a4410632` |

## Completed results and limits

Pooled metrics independently recomputed from saved out-of-golfer predictions:

| View | Model | Records | MAE, degrees | RMSE, degrees | R² |
| --- | --- | ---: | ---: | ---: | ---: |
| FACEON | club_mean | 924 | 3.335242 | 4.692447 | -0.138005 |
| FACEON | ridge | 924 | 4.614194 | 15.243385 | -11.009032 |
| FACEON | hgb | 924 | 3.502482 | 4.989335 | -0.286562 |
| DTL | club_mean | 833 | 3.225756 | 4.550695 | -0.145069 |
| DTL | ridge | 833 | 3.235496 | 4.463704 | -0.101709 |
| DTL | hgb | 833 | 3.258692 | 4.578202 | -0.158954 |

Neither learned model improves pooled MAE over the club-mean baseline in either view. FACEON Ridge golfer1 has RMSE41.0924° and R²−376.2715; this authentic failure is retained, without clipping, exclusion or post-result tuning. DTL unweighted golfer-macro MAE is slightly lower for Ridge/HGB than baseline (3.340656/3.299639 versus3.346480°), while their pooled MAE is worse. This disagreement follows unequal golfer sizes and must not become an unsupported broad improvement claim. Every pooled R² is negative. Eight golfers, small golfer7 folds, correlated records, structural missingness and golfer/club/view confounding limit generalization. No population significance or16-independent-subject claim is justified.

Every held-out golfer is shown below. Metric triplets are **MAE / RMSE / R²**, rounded here only; full precision remains in the immutable artifacts. Selected HGB entries show `(max_leaf_nodes,l2_regularization)`; Ridge selects alpha100 in all16 folds.

### FACEON

| Golfer | n | Club mean | Ridge | HGB | Selected Ridge / HGB |
| --- | ---: | --- | --- | --- | --- |
| 1 | 112 | 1.6394 / 2.2542 / -0.1353 | 8.6728 / 41.0924 / -376.2715 | 1.9840 / 2.5859 / -0.4941 | 100 / (7,1) |
| 2 | 165 | 2.4827 / 3.2043 / -0.2158 | 2.6326 / 3.4640 / -0.4208 | 2.3633 / 2.9959 / -0.0628 | 100 / (15,10) |
| 3 | 230 | 2.0896 / 2.6336 / -0.0507 | 2.6120 / 3.5652 / -0.9255 | 2.3853 / 2.9745 / -0.3404 | 100 / (7,1) |
| 4 | 74 | 3.4193 / 4.5487 / -0.0141 | 4.5123 / 5.7869 / -0.6413 | 3.1819 / 4.3344 / 0.0792 | 100 / (15,10) |
| 5 | 182 | 5.7183 / 7.2777 / -0.7377 | 6.5435 / 8.2853 / -1.2522 | 6.3924 / 8.2882 / -1.2538 | 100 / (15,10) |
| 6 | 109 | 4.4135 / 5.4760 / -0.5610 | 4.5148 / 5.5431 / -0.5995 | 4.1337 / 5.1943 / -0.4045 | 100 / (7,1) |
| 7 | 7 | 1.6747 / 1.9043 / -0.0723 | 1.3930 / 1.7630 / 0.0810 | 1.6777 / 2.1107 / -0.3173 | 100 / (7,1) |
| 8 | 45 | 4.9186 / 6.7040 / -0.1006 | 5.1183 / 7.2018 / -0.2701 | 4.7631 / 6.4498 / -0.0187 | 100 / (15,10) |

### DTL

| Golfer | n | Club mean | Ridge | HGB | Selected Ridge / HGB |
| --- | ---: | --- | --- | --- | --- |
| 1 | 123 | 1.7938 / 2.3746 / -0.2275 | 2.2839 / 2.8512 / -0.7696 | 2.0917 / 2.5437 / -0.4085 | 100 / (7,1) |
| 2 | 165 | 2.6564 / 3.4169 / -0.3824 | 2.4509 / 3.0518 / -0.1028 | 2.2456 / 2.9259 / -0.0137 | 100 / (7,1) |
| 3 | 230 | 2.1039 / 2.6938 / -0.0993 | 2.2597 / 2.8603 / -0.2394 | 2.4049 / 3.1074 / -0.4628 | 100 / (15,10) |
| 4 | 61 | 3.3834 / 4.3183 / -0.0554 | 3.5855 / 4.7493 / -0.2766 | 3.2151 / 4.3567 / -0.0743 | 100 / (7,1) |
| 5 | 170 | 5.6701 / 7.2437 / -0.7907 | 5.4088 / 6.9099 / -0.6295 | 5.7608 / 7.4159 / -0.8768 | 100 / (15,10) |
| 6 | 33 | 4.0193 / 4.7523 / -0.3865 | 3.9721 / 4.6650 / -0.3360 | 3.5421 / 4.1365 / -0.0504 | 100 / (7,1) |
| 7 | 6 | 2.0576 / 2.2553 / -0.3280 | 2.1426 / 2.2674 / -0.3424 | 2.0502 / 2.1636 / -0.2222 | 100 / (7,1) |
| 8 | 45 | 5.0874 / 6.8768 / -0.1581 | 4.6217 / 6.5608 / -0.0541 | 5.0866 / 6.5499 / -0.0506 | 100 / (15,10) |

## Verification and reproduction

- Independent data test command: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_caddieset.py -p no:cacheprovider --basetemp=/private/tmp/swingsync-review-011-pytest` —43 passed.
- Independent selected driver tests: `PYTHONDONTWRITEBYTECODE=1 .tools/tabular-runtime/bin/python -m pytest tests/test_caddie_experiment.py -q -p no:cacheprovider --basetemp=/private/tmp/swingsync-review-011-safe-tests -k 'metric or club_baseline or outer_inner or predictor or early_stopping or selection_weights or registration_hash or atomic_publication or global_lock'` —15 passed,10 deselected. The `metric` selection also runs one synthetic fitted-fold tampering fixture; it uses zero real CaddieSet rows. Physical-core lookup was sandbox-restricted and emitted a logical-core fallback warning.
- Worker separately reported all25 driver tests /68 isolated tests passing, including full synthetic partial/resume/completed-resume and live-source-drift/final-ledger-failure checks. This is worker evidence, not an independent full-suite rerun.
- The session's independent oracle is `/private/tmp/swingsync_review_caddie011.py`; executed as `python3 /private/tmp/swingsync_review_caddie011.py artifacts/caddieset/comparison-v1/run.json`. It is a temporary review aid, not shipped implementation. To reproduce its arithmetic directly: parse the exact CSV/ledger JSON; derive train/test indices by view and held-out golfer; calculate club means on train rows; calculate prediction errors, `MAE=mean(abs(error))`, `RMSE=sqrt(mean(error²))`, `R²=1−sum(error²)/sum((target−mean(target))²)`; concatenate only the eight held-out folds within each view for pooled metrics; take an unweighted mean/population standard deviation across golfer metrics for macro variation. Reconstruct each feature median, missingness indicator and scaler statistic using outer-training rows only; verify all stored canonical fold SHA256 values and source/runtime/config hashes. This requires no model refit.

Reviewer process enumeration was sandbox-blocked (`ps: operation not permitted`); coordinator owns actual process/lock inspection and reports no active expensive worker after completion. The ledger's completed states alone are not a process-list proof. Full repository checks and final Git/remote checkpoint remain coordinator responsibilities. Outcome-serving, source/video/extractor equivalence, empirical reliability and advice gates remain closed regardless of these research results.

## Final coordinator verification and selected inner scores

Coordinator completed-resume evidence: operational CLI returned completed16/16 without a new attempt; ledger SHA256 remained `1b96b88964b9ef278e2bf1bc0482b83508d028d53a3a2808784c1006e183c2e2`, report SHA256 remained `cddb4d1bd3fb54ec518bc0dc35f392223607b3d290f73c3e9b822fb6a4410632`, and attempts remained2. Reviewer independently re-read the same unchanged ledger/report hashes. Coordinator also reports491 full-root-ML tests passing with1 optional-tabular skip,68 isolated tests passing,Ruff clean,and4 frontend tests/build passing. These full-suite/operational-resume results are coordinator evidence; the reviewer did not repeat actual fitting.

Selected inner scores below are the unweighted means of the seven saved validation-golfer MAEs, independently recomputed before accepting each selected candidate. All seven component scores and partition row IDs remain in each ledger selection record.

| View | Held-out golfer | Selected Ridge inner macro MAE | Selected HGB inner macro MAE |
| --- | ---: | ---: | ---: |
| FACEON | 1 | 4.363845 | 3.648567 |
| FACEON | 2 | 4.718961 | 3.575141 |
| FACEON | 3 | 6.301703 | 3.544750 |
| FACEON | 4 | 4.626610 | 3.463490 |
| FACEON | 5 | 3.140039 | 2.831765 |
| FACEON | 6 | 4.382290 | 3.183982 |
| FACEON | 7 | 5.043450 | 3.649829 |
| FACEON | 8 | 4.254221 | 3.234717 |
| DTL | 1 | 3.504273 | 3.390541 |
| DTL | 2 | 3.493528 | 3.455448 |
| DTL | 3 | 3.682736 | 3.266878 |
| DTL | 4 | 3.306034 | 3.314284 |
| DTL | 5 | 2.888673 | 2.875693 |
| DTL | 6 | 3.330831 | 3.314742 |
| DTL | 7 | 3.487594 | 3.494265 |
| DTL | 8 | 3.132297 | 3.244723 |

## Public aggregate claims review

The small public aggregate `autonomy/reports/caddieset-results-011.json`, SHA256 `d2b3f46396d6600cd1bbc16eef6467b00f917dccf2b13e8ba43556f29cc58cd7`, exactly matches the independently audited raw report results, runtime and fixed config. Independent checks also passed all32 selected parameter/row-count/inner-score records, all four paired baseline delta and golfer-win/loss summaries, the retained FACEON Ridge golfer1 prediction minimum/maximum/error metrics, packaging witness SHA and closed gates. The interpretation correctly distinguishes the small DTL macro-MAE/Ridge-RMSE improvements from worse pooled MAE in both views. `docs/caddieset-experiments.md` accurately describes explicit incomplete-fold replay, absence of model-internal iteration checkpoints, separate isolated runtime and research-only outputs. No material public-results claim issue was found in these reviewed versions.


## Final documentation/claims review — ready for checkpoint

Reviewed the final `autonomy/CURRENT_STATUS.md`, `autonomy/backlog.json`, `autonomy/reports/validation-011.md`, README, model card, portfolio evidence, remaining-work overview, tabular runtime documentation and completed-experiment documentation. They consistently mark006 as a completed fixed16-fold research comparison, preserve the negative/mixed results and severe Ridge failure, distinguish initial direction from curvature, retain eight-golfer/duplicate/club/view limitations, and leave uploaded-video outcomes, feature equivalence, empirical reliability and coaching disabled. Four major remaining workstreams and the separate next003g input-admission increment are accurately scoped. Stale unrun/remaining-tabular wording and the overbroad prohibition on claiming any supervised outcome research were corrected; the documentation now permits only the actually evidenced research claim. It explicitly distinguishes synthetic fixture fits from zero actual-data reviewer refits.

The public aggregate remains byte-identical at SHA256 `d2b3f46396d6600cd1bbc16eef6467b00f917dccf2b13e8ba43556f29cc58cd7`; raw completed run/report and frozen implementation hashes remain those audited above. Coordinator full-check evidence remains base394passed/12optional skips,fullML491passed/1optional-tabular skip,isolated68passed,Ruff clean,frontend4tests/typecheck/build passed. No new fitting or implementation change was required by the final claims check. **No unresolved material finding remains; ready for the coordinator's final checkpoint.** Final commit/push/remote CI verification remains coordinator-owned and is not preclaimed by this review.

Final claims documents reviewed at these byte identities:

| File | SHA256 |
| --- | --- |
| `autonomy/CURRENT_STATUS.md` | `07bb9280479d796833cd2e1c7e1ef9a83d9c95d6d2ab0eaf3bf904f4970cd7e4` |
| `autonomy/backlog.json` | `c19e3d7c21c3b17f4604807658ab7a5bd82f51fcdc1d5c34eac341a97a9b3bfc` |
| `autonomy/reports/validation-011.md` | `997f593a0ff83300f9a7570fd05c5ed2d47adae196fdfc1852817f9883d5bb15` |
| `README.md` | `d11184745b724804aed85843ee0896f2cba7a8dcd9d7b75e105c3ac2e705e6a3` |
| `docs/model-card.md` | `79e3986b5d1a88f7baf0d8d90c604bfcfd56d1280a09ce60f9eb97d3b3b4c09b` |
| `docs/portfolio-evidence.md` | `337b8a69bd482272abe9100c5e01e0211f62642c0bd5bac863b1adc286dd8f63` |
| `docs/remaining-work.md` | `9076c7355b5e8c901e9bfcbb41f006f34d69ce7e3a92e426b93cfe69ae7bbb09` |
| `docs/tabular-runtime.md` | `158f8f1e7d9effe15b295319dc7a06b8344996093aa996bc2992f8f5388dec2d` |
| `docs/caddieset-experiments.md` | `87b937db7b5f183d88b7f2b97a55159513d3b11405bc748aa44118300340852c` |
