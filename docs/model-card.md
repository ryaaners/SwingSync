# SwingSync event model card

Status: PROJECT-TRAINED RESEARCH HEADS EVALUATED ON OFFICIAL FOLD1; AUTOMATIC APP INFERENCE UNAVAILABLE. Updated 2026-11-10.

## Intended use and current app behavior

Research localization of eight golf swing events in a trimmed video. The current local React/FastAPI app supports video inspection and explicitly manual frame annotations, corrections, descriptive recording-time tempo and image reports. Automatic proposals, pose extraction, shot outcomes and coaching remain disabled. This is a student research/portfolio project, with no validated coaching or injury-prevention efficacy.

## Trained artifacts and selection

Both heads were genuinely trained in this project on the same **frozen pretrained ImageNet V2 MobileNetV2 features**. The image backbone was not trained here. No pretrained SwingNet event checkpoint/result is used. Features are deterministic RGB tensor resize short edge 232/center crop 224/ImageNet normalization, 1280 dimensions. This differs from reference 160-pixel preprocessing.

The baseline is a frame-independent linear head with nine classes. The temporal head has one bidirectional LSTM layer with 64 hidden units per direction and a nine-class head. Classes are eight events plus background. Training uses complete clips, packed valid lengths, padding-safe weighted cross entropy, inverse train-only class frequency, no augmentation, 20 epochs per head, batch size 4, Adam learning rate 0.001 and seed 20261105 on one CPU thread. Official fold 1 contains 840 training clips from 348 sources, 210 inner-validation clips from 88 sources, and 350 test clips from 144 sources, with no source overlap. The earliest maximum inner-validation PCE selected frame epoch 19 (41.96%) and temporal epoch 20 (64.58%). Those inner scores are not test scores.

Final continuation checkpoints retain both final-epoch model and selected **best_model**. Evaluate selected weights; frame final-epoch weights differ. Final SHA256:

- Frame: `8203c5f5d2d84d2fb8519dffa5d053932cfe353069f145d54b17ff28ce5a2431`.
- Temporal: `910f6d06a253e8117e825f5d625a1dcc11fe6db61344d169daac3fd973ceb5c3`.

## Fixed held-out result

The prospective contract was committed before test extraction, binding checkpoints/selected epochs/manifest/runtime/pipeline and both contexts. One actual first attempt evaluated all 350 test clips / 98,956 frames. No held-out tuning, reselection or order constraint was used. Independent replay reproduced every one of 1,400 stored per-clip variant results and all aggregate/per-event metrics exactly.

| Head | Evaluation context | PCE (%) | MAE frames | MAE recording ms |
| --- | --- | ---: | ---: | ---: |
| Frame baseline | reference-64-reset | 45.11 | 22.224 | 741.489 |
| Temporal | reference-64-reset | 58.79 | 11.700 | 390.368 |
| Frame baseline | full-clip | 45.11 | 22.224 | 741.489 |
| Temporal | full-clip | 64.43 | 6.212 | 207.253 |

The standard reference 64-frame temporal result is **58.79% PCE**, compared with **45.11%** baseline; full-clip **64.43%** is separately disclosed. The reference context uses independent 64-frame reset calls with unpadded tail, not complete upstream architecture/training replication. Training/inner selection use full-clip context; the differing test context is deliberately disclosed. Official PCE uses chronological occupied annotation frames, variable tolerance `max(round((slot5-slot0)/30),1)`, inclusive error and pinned NumPy default argsort. Timing errors use actual re-encoded PTS. Clip 677's upstream ordering anomaly is retained in training / official fold 3 and does not occur in this test fold.

Full per-event errors, fixed config, provenance and limitations: [experiments.md](experiments.md), [held-out contract](../experiments/heldout-v1.json), [machine results](../autonomy/reports/heldout-results-008.json), [independent review](../autonomy/reports/review-008.md). Raw run SHA256 `93d9aba2332fd90fdb7cf77891a580550e42a38c68c2b10c0e774fa754c6bdb0`.

## Reliability and integration limits

- This is one official outer fold. Other folds have not been trained/evaluated; do not claim four-fold performance. Source-recording separation does not prove unseen-golfer performance because golfers can recur across sources.
- Address/Finish remain weak: temporal reference 64-frame PCE 13.43% / 17.14%. Post-hoc saved-peak diagnostics show 279/350 strictly chronological reference 64-frame temporal timelines (321/350 full-clip). These diagnostics did not alter decoding, filter results or choose a model/context.
- No automatic event-error/probability calibration or uploaded-video input-pipeline generalization has been validated. A class probability is not validated reliability. Trained/evaluated research artifacts alone do not enable normal coaching.
- Original-source image/PTS and browser/audio/display-rotation synchronization remain unverified. Slow motion/time remapping can change physical timing. Canonical manual frames are the current app evidence.
- A real isolated local pose extractor completed a training-clip smoke, but empirical landmark/movement accuracy remains unestablished. Visibility/view/handedness/mirroring/event error affect projected measurements. Ordinary landmarks do not support pressure, forces, clubface angle or calibrated 3D rotation.
- Manual marks retain current/superseded revisions, recording hash, native/relative timestamps and frame images. Their descriptive timing reliability is unvalidated; no ideal threshold, quality score or advice is inferred. Synthetic software tests are not accuracy, user testing or a permitted-real-golf demonstration.
- CaddieSet grouped tabular training completed as a separate fixed research comparison; outcome serving remains disabled. HRNet/FasterR-CNN/SwingNet features are not established equivalent to MediaPipe; uploaded-video outcomes stay disabled. DirectionAngle describes initial direction, not curvature. Associations/feature importance cannot establish causal fixes.

## Data, code, weights and hardware

GolfDB annotations and reference implementation audited at `a63b4ce8c0900d09abbe92519223efe16810f5f4`: 1,400 clips / 580 sources / four official outer splits. Dataset MAT SHA256 `36523fcef016d93334691ea058d643bcf0d3a4a8c70b733f3b4cc0b8e5e0cf0a`; exact manifest and media/cache fingerprints are retained. Authors' reference clip archive is for local research; source-video redistribution rights are not assumed. Code/data/backbone/weights licenses are reviewed separately; no data or weights are committed or relabeled as original project code.

Actual measured hardware: Apple M1 / 8 GiB, Torch 2.9.1/torchvision 0.24.1/NumPy 2.2.5/PyAV 16.0.1/Python 3.11.14, CPU one thread. CUDA unavailable; MPS available but unused. Separate one-pass timing on two TRAIN clips / 462 frames measured shared decode/transform/backbone wall 26.867317 s (58.154366 ms/frame), with initialization 0.329 s / warmup 1.124 s excluded and per-head/context time separate. Two clips do not establish representative app latency or minimum hardware. Outer run 19.520488 s is evaluation-loop elapsed, not video inference latency. See latency-006.json.

Checkpoints/catalogs/raw media remain ignored local artifacts. Exact runtime/source fingerprints are intentionally bound; relocation/cross-platform artifact reproduction is unverified. New environments/folds require explicit prospective registration. Historical smoke losses and bounded cache proofs remain separate software evidence. Execution 009 installed a separately locked MediaPipe 1.2.2/NumPy 2.2.5/OpenCV 4.12/PyAV 16 CPU runtime and acquired exact upstream bundle/member hashes with separate license evidence. Binary Mac ARM floor is 14; actual execution used macOS 26.3. Root ML dependencies stayed unchanged.

The real pose smoke on one TRAIN clip retained all 138 native frame timestamps: 134 single-pose returns and 4 explicit missing frames. Decode plus detector calls took 3.589 seconds in one pass, excluding import/preflight/task initialization; native threads were not explicitly limited. This is execution/frame-correspondence evidence, not landmark accuracy or general latency. The OS launcher denied native network access because the wheel declares utilization telemetry; Linux extraction awaits verified isolation. The upstream NORM_RECT/IMAGE_DIMENSIONS projection warning remains an accuracy-validation concern. No pose model/member/media redistribution or ordinary app integration was enabled. See pose-assets.md, pose-extraction.md and pose-smoke-009.json.

## Supported portfolio scope

Research-only training/evaluation statements can cite this specific fixed fold, trained heads and frozen-backbone provenance; [portfolio-evidence.md](portfolio-evidence.md) gives bounded examples. No completed end-to-end automatic coaching, unseen-golfer event performance, validated outcome serving or accuracy-on-user-videos claim is supported. Remaining release gates are actual input/pose reliability, permitted real-video-to-report demonstration and reproducibility beyond this local environment.

## Separate CaddieSet outcome research — execution 011

The fixed first DirectionAngle study completed eight leave-one-golfer-out folds separately for FACEON and DTL, with grouped inner tuning and training-only transforms. All69 registered swing features plus club metadata were used; all ball outcomes/derivatives and golfer identity were excluded from predictors. The initial-direction target is not shot curvature. The eight golfers and correlated/duplicate view records are not a large independent population.

Pooled MAE (degrees) is FACEON3.335 club-mean /4.614 Ridge /3.502 HGB; DTL3.226 /3.235 /3.259. Neither learned model improves pooled MAE in either view; small DTL macro/RMSE changes are mixed and do not establish consistent generalization. FACEON Ridge extrapolates severely for one golfer; the raw failure is retained without clipping or test-driven repair. Full per-fold MAE/RMSE/R², variability, selected inner configurations, provenance and caveats are in caddieset-experiments.md and caddieset-results-011.json. No current held-out feedback may be used to retune this fixed comparison.

Outcome models remain research-only, with no exported product estimator, uploaded-video feature compatibility, calibrated coaching claim or serving endpoint. Lack of consistent baseline improvement and absent feature equivalence both preserve the closed integration gate. The event-model comparison and its original locks/checkpoints are unchanged.
