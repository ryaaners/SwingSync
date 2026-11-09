# Pose reference review and agreement

The offline extractor now has a strict manual-reference format and a descriptive agreement evaluator. They support inspection and future validation work; they do not establish landmark accuracy or enable app coaching. The registered pilot contains **zero human reference annotations and no empirical accuracy result**. Do not fill coordinates from model outputs or describe generated templates as labeled data.

## Registered pilot and future accuracy study

[pose-validation-v1.json](../experiments/pose-validation-v1.json) binds the completed execution009 video/run/result hashes, 160×160 coded raster, explicit view/hand/mirror declarations and five fixed decoded indices: 0, 37, 68, 87 and 137. These are diagnostic sample indices, not detected golf events. The source is one GolfDB preprocessed training clip with reencoded timestamps. The existing model output was already observed, so this pilot is not a prospectively blinded evaluation of a new model.

The protocol fixes units, missingness, reference collection and conditional error accounting before new diagnostic reference collection. It never authorizes model/threshold selection, app integration, outcome inference or coaching. A future empirical study needs a separate prospective recording/view/sample plan, independent repeated annotations and disagreement reporting, credible identity/mirror conventions, and a broader set of independent sources. Frames from one recording must not be counted as independent golfers. Do not use the completed event-model held-out fold for tuning these policies.

## Clean annotation workflow

The evidence builder produces a clean reference page with source frames and empty annotation slots. It contains no model coordinates or overlays. Annotate from the clean page before seeing the diagnostic overlay page if you intend to attest blinding. Blinding and independence checkboxes must start unchecked; a reviewer must affirm their actual process and supply an identifier. A tool cannot prove those declarations.

The template starts with all 165 slots (five frames ×33 landmarks) marked `unreviewed`, both coordinates null, empty reviewer identifier and false attestations. A draft may be saved and structurally validated. Scoring requires explicit model-blinded and independent attestations, an identifier and at least one manually visible point. Partial coverage remains partial; unreviewed slots are counted rather than silently dropped. A reference from someone who already viewed the model overlay must not claim model blinding.

Build a fresh local pilot bundle using the already completed artifact; this does not run pose inference:

```sh
uv run --extra ml python scripts/build_pose_evidence.py \
  --run artifacts/pose/smoke-009-train-0/run.json \
  --run-sha256 772ab832647ccede39291b95b0179e54f5cf84e67c4af86980e1c1bf5691fb65 \
  --video data/golfdb/videos_160/0.mp4 \
  --output artifacts/pose/review-pilot-new \
  --frame-index 0 --frame-index 37 --frame-index 68 --frame-index 87 --frame-index 137
```

Use `reference.html` for clean annotation and export its JSON. `review.html` is a separate diagnostic page that reveals model output; showing it to the reference annotator defeats blinding. Neither source frame redistribution nor a qualified independent reviewer is supplied by these commands. Keep local outputs ignored.

The interface is anatomical: MediaPipe IDs 0..32 refer to the subject's anatomy. Do not swap left/right IDs to match screen position. View, handedness and mirror declarations must match the registered source. The schema requires all 33 IDs exactly once at every registered frame, in canonical order, and refuses missing/reordered frames, altered timestamps, geometry, source hashes or coordinate units.

Each landmark has one of four manual states:

| State | Coordinates | Interpretation |
| --- | --- | --- |
| `visible` | Finite image-edge pixel x/y | Reviewer can identify the joint in this frame. |
| `occluded` | Both null | Joint cannot be located visibly; do not guess its hidden position. |
| `out_of_frame` | Both null | Joint is beyond the raster. |
| `unreviewed` | Both null | No reference decision has been made. |

Coordinates are continuous image-edge pixels: x∈[0,W], y∈[0,H]. Pixel centers run from (0.5,0.5) through (W−0.5,H−0.5). Model normalized x/y map to x*W/y*H, using each axis separately; W−1/H−1 scaling is a different convention and is not accepted. Display resizing must map pointer positions back to these original raster units. Neither z nor estimated world depth enters agreement calculations.

## Error, coverage and abstention

The evaluator reports radial Euclidean pixel error for eligible visible manual points and aggregate/per-joint mean, median, RMSE and maximum. It reports all reference-state counts, total slots, eligible visible points, scored points and model abstentions. No-pose/malformed returns, missing or presence-filtered points, visibility below the registered engineering floor and out-of-image model points remain abstentions. Model coordinates are never clamped, interpolated or matched to a different frame. Any multiple-pose observation anywhere makes the whole sequence identity-ambiguous and abstains on all eligible points.

Optional torso-normalized error divides the same pixel error by shoulder-midpoint to hip-midpoint length, using **independent visible manual landmarks 11/12/23/24 in that same frame**. If those references are missing or their scale is degenerate, normalized error is unavailable; the model's own torso is not substituted. This makes the normalization independently defined, not calibrated anatomical measurement.

Error summaries are conditional on scored points. Coverage and model abstentions must accompany every comparison so a model cannot look better by returning fewer points. A single operator reference is not ground truth, and scores do not prove causal coaching fixes. No universal ideal/error threshold or automatic reliability promotion is implemented. Every report keeps advice empty and product/coaching/outcome integration false.

The pure APIs are in `backend/swingsync/research/pose_validation.py`: `make_reference_template`, `validate_reference` and `evaluate_reference`. They require separately authenticated completed pose provenance from the caller; passing self-consistent dictionaries alone does not authenticate artifacts. The operational evaluator uses the source/hash-checked artifact loader before calling these functions. Keep source media, rendered images and completed reference files in ignored local artifact storage.

Only after an actual independent reviewer exports eligible reference points and truthful attestations, run the separate evaluator with a fresh output file:

```sh
uv run python scripts/evaluate_pose_reference.py \
  --run artifacts/pose/smoke-009-train-0/run.json \
  --run-sha256 772ab832647ccede39291b95b0179e54f5cf84e67c4af86980e1c1bf5691fb65 \
  --video data/golfdb/videos_160/0.mp4 \
  --reference artifacts/pose/manual-reference.json \
  --output artifacts/pose/reference-agreement-new.json
```

`reference-template.json` intentionally fails scoring until manually reviewed. The evaluator binds exact reference-file/protocol/source identities and refuses overwrite or drift. Its output remains conditional operator-reference agreement; it cannot complete the empirical accuracy gate, enable app measurements or establish a coaching threshold.

## Projection warning scope

The observed NORM_RECT/IMAGE_DIMENSIONS message is a graph-input warning, not a direct test of input-raster aspect or measured accuracy. Pinned public source builds square physical ROIs; the legacy projection formula is consistent with that assumption even on rectangular source images. Independent square-ROI and deliberately nonsquare-ROI arithmetic checks explain the distinction without changing saved predictions. See [source/geometry audit](../autonomy/reports/pose-projection-audit-010.md) and its exact source hashes. Native internal-ROI correspondence and empirical agreement remain validation gates; the warning is not globally suppressed.
