# SwingSync model card

Status: MODEL UNAVAILABLE. Updated 2026-11-04.

## Intended use

Local analysis of one trimmed golf swing with explicit camera view, handedness and club. Future event models identify eight events, after which supported 2D measurements can reference specific frames. This is a student portfolio/research application, with no validated coaching or injury-prevention efficacy.

## Artifacts and evidence

No SwingSync-trained event checkpoint, pretrained-reference inference result, outcome checkpoint, benchmark, measured model latency or minimum hardware requirement is available yet. Audit metadata and synthetic software fixtures are not model-performance evidence. The ordinary app must say model unavailable until a real artifact is validated.

Planned comparison: shared frozen MobileNetV2 features with a frame classifier and a bidirectional LSTM temporal head, nine classes including background. Training/evaluation and exact GolfDB PCE are described in [experiments.md](experiments.md) and [data-contracts.md](data-contracts.md). Device selection must detect CUDA/MPS/CPU and support CPU rather than assuming CUDA as the historical reference does.

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
