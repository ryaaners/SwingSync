# Current status — 2026-11-10

Execution011 completed006: the fixed first CaddieSet grouped tabular comparison, with real smoke,16 outer folds and independent verification. Neither learned model improves pooled MAE over club means in either view; small secondary DTL changes are mixed and disclosed. This is a completed negative/mixed research result, not a serving pipeline. Four major workstreams remain (docs/remaining-work.md). The normal app remains manual-event review; automatic events, pose measurements, outcomes and advice stay disabled.

## Jobs — none active at completion

Coordinator inspected actual processes and saved identities: old event training, feature/evaluation and009 pose workers ended; shared `artifacts/training/active-training.lock` was free. One new CaddieSet train-only smoke and one comparison job genuinely completed sequentially. No duplicate expensive worker or background training remains. Inspect actual processes/lock/ledgers before any new fitting. Launched/partial is not completed; do not restart old jobs or silently duplicate an interrupted fold.

Actual smoke `artifacts/caddieset/smoke-v1/`:job615ea576ede341c2bf1c428660bb0e63,64 FACEON training rows/golfers2/3,zero held-out rows/accuracy metrics,completed. Ledger SHA256 `51c7de24a5304bcda8a77c091998c26eb25c521d0fab894a2fc611e959296083`.

Actual comparison `artifacts/caddieset/comparison-v1/`:jobab223a3043834df6a68ba29e2ad5050f,completed16/16 folds with two explicit attempts (one then15resume). LedgerSHA `1b96b88964b9ef278e2bf1bc0482b83508d028d53a3a2808784c1006e183c2e2`; reportSHA `cddb4d1bd3fb54ec518bc0dc35f392223607b3d290f73c3e9b822fb6a4410632`. Completed `--resume` returned byte-identical ledger/report with no new fits/attempts. First-foldSHA0777d22645b16448da58c673be9c4fec65875f3b0e3353addc40d404d02b162c remained unchanged. Do not refit/reselect/tune these held-out golfers.

## Completed CaddieSet experiment — preserve fixed evidence

Prospective config/protocol/data contract/runtime were committed+pushed `dcd1b0b2d3dd302ffe54a8f366fe15a1070b469d` BEFORE any actual CSV fitting; preregCI42781688208 passed. Config `experiments/caddieset-v1.json` SHA `8fc099bc65e4b77bbab5dd88f896d27231c0f5b524e712e932e2c9f7dadb832d`. Separate non-workspace tabular lockSHA `ea54460e550b3528de4fceb93f03f3b7a910a75e734909bd282078123b6bf772`; env `.tools/tabular-runtime`,Python3.11.14/NumPy2.2.5/SciPy1.15.3/pandas2.3.3/sklearn1.7.2. Native imports/locked offline sync passed. Apple M1/8GiB,CPU numerical thread limit requested1; supported OpenMP pool1 observed separately. Fit-loop elapsed1.013s+16.478s excludes admission/import; not app latency/minimum hardware. No root/pose lock changes.

Dataset pinned revision3c73d9d40580bb8a5a10711ad1fa10735a205ffe, CSV SHA646459b081863e244d75efe6f09fad4d34750d2fed7fde44db654e7720a9a306,1757×80/eight golfers. New audit corrected11 literal`inf` features, separately counted/mapped to NaN before fitting along with58,917empty and11`#NAME?` cells. All outcomes finite. Retain six duplicate rows. Only69 exact swing features+ClubType enter X; all8ball outcomes/derivatives and GolferId forbidden as predictors. View splits separate analyses. WEIGHT-SHIFT is a projected line angle, not pressure/force. Whole-golfer grouping prevents duplicate/potential alternate-view records crossing partitions; no authoritative shotID or MediaPipe equivalence exists.

Each view has8leave-one-golfer-out folds; inner3GroupKFold uses only seven training golfers with equal-golfer MAE selection, fixed first-candidate ties and no random rows. Numeric median imputation/indicators/retained empty columns, Ridge scaler and club encoder fit inside relevant training partitions. Baseline uses training club means with global fallback. Ridge alpha1/10/100; HGBpaired7/1or15/10leaf/l2,100iterations,early_stoppingFalse,seed20261110. No globally fitted selector or outer-driven changes.

| View/model | Pooled MAE degrees | RMSE | R² | Macro golfer MAE |
| --- | ---: | ---: | ---: | ---: |
| FACEON club mean | 3.3352 | 4.6924 | −0.1380 | 3.2945 |
| FACEON Ridge | 4.6142 | 15.2434 | −11.0090 | 4.4999 |
| FACEON HGB | 3.5025 | 4.9893 | −0.2866 | 3.3602 |
| DTL club mean | 3.2258 | 4.5507 | −0.1451 | 3.3465 |
| DTL Ridge | 3.2355 | 4.4637 | −0.1017 | 3.3407 |
| DTL HGB | 3.2587 | 4.5782 | −0.1590 | 3.2996 |

Neither learned model improves pooled MAE for either view. DTL macro MAE/Ridge RMSE improve slightly; do not hide that or claim broad gains. FACEON Ridge golfer1 predicts−70.72..358.42degrees (RMSE41.0924), retained without clipping/wrapping/test-driven repair. All Ridge folds chose alpha100 on inner data. Actual unseen-club count0; fallback exercised by synthetic tests only. Eight golfers, row correlation and club/view confounding prevent population/significance claims. DirectionAngle is initial direction, not slice/hook curvature. Model associations cannot prescribe causal fixes.

Small aggregate `autonomy/reports/caddieset-results-011.json` with raw-job/model selection/provenance/witness details; independent review011 verified all16 folds/1757rows/48 prediction vectors,2208 training-feature medians,indicators/scalers/encoders/baselines,inner grouping/selection and all metrics without actual CSV refitting. Raw CSV/predictions stay ignored. Follow caddieset-experiments.md; critical frozen files include caddie_experiment.py,caddieset.py,run_caddieset.py,experiments/caddieset-v1.md,docs/caddieset-contract.md and tabular lock. Editing them invalidates existing resume provenance; do not mutate to make results look better.

## Existing app and pose evidence

The manual app supports upload/progress/history/deletion, canonical SHA/PTS PNGs, exact decoded-index manual events, revision conflicts/history, recomputed descriptive tempo and self-contained reports. Synthetic007 browser QA verifies software, not accuracy/user testing. Browser/audio/display-rotation correspondence remains unverified; canonical decoded frame evidence is authoritative.

Pose extraction004d is genuinely working as separate research only: env `.tools/pose-runtime` with independent lockSHA `edc1d9c8c3e6e3a794bbb0e582e25f292ce5bf3b640f11026c3cce6df9135919`; upstream bundleSHA5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1. MacCPU native inference ran under OS network denial because the wheel declares utilization telemetry; no verified Python optout/Linux extraction isolation. Model/member/media redistribution stays disabled; exact-member attribution unresolved.

Pose smoke009 completed TRAINclip0,138 frames134single/4missing,actual nativePTS plus separate integer-msclock,cpu3.589sdecode+calls excludinginit. Not accuracy/identity/minhardware evidence. LedgerSHA772ab832647ccede39291b95b0179e54f5cf84e67c4af86980e1c1bf5691fb65; rawresultSHAe972b761bc519b7544f85fc1a19224d6e9c8d39d3c2caa6d246de1ec9c432ba1. Presence/visibility engineering filters are unvalidated, missing/malformed/multi explicit; anymulti suppresses whole spatial conversion, no guessed handedness/ID swaps or calibrated3D.

004e1 workbench completed010: all138 savedPTS verified,5sourcePNGs exactRGB,diagnostic overlays plus cleanmodel-data-free reference page,165unreviewed/null slots, false blinding/independence declarations. Browser click/navigation/persistence/clear/export tested, QA mark cleared. No empirical reference labels/accuracy scores created; draft scoring refused. Final ignored bundle `artifacts/pose/review-010-verified/manifest.json` SHA47d5141ec64c6ea29bebf64a2524422fcf243ac374abd54aef0c90069f9215f0; protocolSHA29e82013cc4de62b043976cb1098d50383f4aa6007919cb9cf96c7f7575965f5. Source warning branch/square-ROI arithmetic explained, not proof of native internal ROI/anatomical accuracy. See010 reports/pose-review.md/pose-validation.md.

## Preserve completed event benchmark

Registered fold1 research003 is complete. Train840/348sources,inner210/88,test350/144;seed20261105,splitseed20261104. Frozen upstream ImageNet V2 MobileNetV2 backbone, genuinely trained frame/BiLSTM heads20epochs/4200updates; selected inner epochs19/20. Use best_model. Frame checkpointSHA8203c5f5d2d84d2fb8519dffa5d053932cfe353069f145d54b17ff28ce5a2431; temporal910f6d06a253e8117e825f5d625a1dcc11fe6db61344d169daac3fd973ceb5c3.

Prospective heldoutcontractSHA4b4339a469dd5fe09f97b57ae4b650ce378ce1c3a3863f86b46249ad8d9d0008 committed641fe51 beforetestprocessing. Fixed reference64-reset PCE45.1071%frame/58.7857%temporal,MAE22.224frames/741.489ms versus11.700frames/390.368ms; separate preregistered fullcliptemporal64.4286%,6.212frames/207.253ms. Independent1400variant replay exact. RawrunSHA93d9aba2332fd90fdb7cf77891a580550e42a38c68c2b10c0e774fa754c6bdb0; testcatalogSHA159ec421d82ca82d66cfe070d4272e013c65604365cf2c31fc11a57af0d66cf4. Rootuvlock `8ce1d6674a07bf122541f555b90ad81532029b7ce343af54a78cd057fc2b26fc` and old model sources unchanged. Scoped TRAINfeatureCPU58.154366ms/frame,not app latency.

Weak temporal reference Address/Finish13.43%/17.14%,279/350 strictlyordered timelines,one source-disjoint fold and uncalibrateduploadedinputs do not justify dependable eight-event coaching. Do not tune against completedtest. Morefolds/newmodels need prospective registration; original-source PTS equivalence needs authorized originals. No four-fold/unseen-golfer event accuracy or automatic-serving claim.

## Next highest-priority unblocked increment and gates

Proceed003g uploaded-video event input compatibility/admission: define a prospective raw-upload crop/view/raster/time contract, implement research-only preprocessing and parity checks on permitted TRAIN media before ordinary serving. Separate input/software parity from empirical error/confidence/order reliability. Keep normal model unavailable and advice empty until appropriate validation; do not retune current held-out fold. This software work can progress independently of reference collection.

004e2 precise unblock remains independent truthfully model-blinded manual landmarks, repeat-review/disagreement and broader permitted recordings with known view/hand/mirror/identity under a separate prospective protocol. Templates/agentQA/arithmetic cannot substitute for labels. Original-source002 precision, browser synchronization, empiricalpose/event reliability,real permittedvideo-to-reportdemo,Linux pose isolation and source/model redistribution remain gates. Record blockers once and continue independent work, not repeated polling.

006 research is complete; optional video-outcome integration remains closed for BOTH absent feature equivalence (videos/extractor/definitions/eventtiming/pairedexamples) and lack of consistent baseline improvement. Do not add these predictions to normal coaching or repeatedly tune this dataset's held-out golfers. Professional/fault labels, accounts/payments/livecamera/cloud remain outsideMVP.

## Checks, Git and scheduling

Final011 checks: base394passed/12explicit optional skips,fullapp/ML491passed/1optionaltabularmodule skip;isolatedtabular68passed(43sharedloader+25driver). Ruffbackend/tests/scripts andfrontendcleaninstall4tests/typecheck/build pass;0known npm vulnerabilities;one existingAnyIOdeprecationwarning. Fresh independent review011 has no unresolved material findings. Lightweight CI adds isolatedtabularcontracts withoutdatasets/fulltraining, alongside existing4jobs. No usermedia/datasets/models/rawpredictions in Git.

Only coordinator uses dedicated `https://github.com/ryaaners/SwingSync`, branch`agent/swingsync`,noforce/automerge. Preregcheckpointdcd1b0b/CI42781688208passed; finalverifiedcommit/remoteSHA/CIreported in execution reply and ignoredcheckpointartifact. Inspect actualHEAD/remote next time. Daily user-authorized1:30a.m.Toronto automation stays active; no new schedule/persistentagent created or expensiveworker left running.
