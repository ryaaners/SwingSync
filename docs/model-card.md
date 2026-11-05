# SwingSync model card

Status: VALIDATED EVENT MODEL UNAVAILABLE. Updated 2026-11-05.

## Intended use

Local analysis of one trimmed golf swing with explicit camera view, handedness and club. Future event models identify eight events, after which supported 2D measurements can reference specific frames. This is a student portfolio/research application, with no validated coaching or injury-prevention efficacy.

## Artifacts and evidence

No validated SwingSync event checkpoint, pretrained-reference swing inference result, outcome checkpoint, benchmark, measured model latency or minimum hardware requirement is available yet. Local smoke artifacts now contain five optimizer updates for each head on two real training clips; these are not deployable/validated event models. Audit metadata and synthetic software fixtures are not model-performance evidence. The ordinary app must say model unavailable until a real artifact is validated.

Implemented architecture: shared frozen ImageNet V2 MobileNetV2 features with a frame classifier and packed bidirectional LSTM temporal head, nine classes including background. Full comparison is pending. Training/evaluation and exact GolfDB PCE are described in [experiments.md](experiments.md) and [data-contracts.md](data-contracts.md). Device selection detects CUDA/MPS/CPU; CPU smoke passed on Apple M1/8GiB with exact checkpoint resume. No accelerator timing or minimum specification is established.

## Data provenance

GolfDB annotations inspected at revision `a63b4ce8c0900d09abbe92519223efe16810f5f4`: 1,400 clips, 580 sources, four official folds. The initial audit acquired no media. Increment002 acquired the authors’ linked preprocessed clip archive for local research and verified decoded clip integrity; no event weights acquired. Original source PTS and exact source-image alignment remain unavailable. Code, media and weights require distinct licensing review. CaddieSet CSV inspected at `3c73d9d40580bb8a5a10711ad1fa10735a205ffe`: 1,757 view records, eight golfers, missing features and spreadsheet errors. No released extraction code or videos found in the inspected tree.

## Limitations and gates

- Standard GolfDB splits separate source recordings; do not claim unseen-golfer evaluation without an additional grouped study.
- No automatic event reliability calibration has been performed. Class probability alone must not be called validated reliability.
- Pose visibility, camera orientation, occlusion and event timing affect derived 2D measures. Monocular landmarks do not support pressure, forces, clubface angle or calibrated 3D rotation.
- Descriptive observations must include value, interval, evidence, method and limitations. Suppress unreliable observations; no universal ideal thresholds or professional-similarity quality score.
- Manual event edits must retain provenance and trigger recomputation. They cannot masquerade as automatic predictions.
- CaddieSet uses an HRNet/Faster R-CNN/SwingNet feature process, with no demonstrated equivalence to MediaPipe. Video-to-outcome inference remains disabled. DirectionAngle describes initial direction, not curvature.
- Neither training performance nor feature importance proves that changing a body position improves a shot.

## Required before release claims

Attach final config/seed/split and dataset fingerprints, trained checkpoint hash and provenance, baseline/temporal held-out PCE and timing errors, software and leakage checks, measured local latency, a permitted real-video demonstration and an independent review. Until then, no résumé bullet may claim measured ML performance or completed end-to-end coaching.

## Smoke provenance

See autonomy/reports/smoke-003.json. Reference ImageNet initialization was downloaded from torchvision, not trained here. Event heads received five optimizer steps on fold 1 training clips 0/1; no validation or test partition was evaluated. Training losses and exact CPU resume only demonstrate software operation. No model is enabled in the web app, and no measured shot outcome or coaching advice was added.

## Dataset preparation and full-training status (003b)

A fixed first comparison is now registered, and resumable full feature preparation is running for training plus inner validation only. Completion is not inferred from launch. The complete-cache training/selection driver is implemented and software-tested, but has not trained a full real model. The web app remains model unavailable. See validation-004 and current job state before making any completion or performance claim.
