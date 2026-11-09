# Local pose review and independent references

The offline workbench turns an authenticated, completed pose extraction into two separate local HTML pages. It does **not** run the pose model, change event predictions, validate accuracy or enable the product's pose measurements. It is research tooling, separate from the normal upload app.

- `review.html` shows canonical decoded source frames beside diagnostic projected pose overlays, raw landmark scores and explicit missing/malformed/multiple-pose statuses. Looking at this page reveals model output.
- `reference.html` contains only canonical clean source images, anatomical landmark names and the reviewer's own manual marks. It contains no model coordinates, scores or rendered overlays. The reviewer must not inspect model output before creating an independent, blinded reference.
- `reference-template.json` starts with all 33 landmarks unreviewed, no coordinates, empty reviewer identity and unchecked blinding/independence declarations. It is not ground truth or a completed accuracy evaluation.
- `manifest.json` binds both pages and the template by SHA256, records source/run/result identities, prospective protocol SHA256, exact renderer/CLI/loader/study/manual-measurement source hashes and root dependency lock hash and selected frame indices, and explicitly keeps coaching validation and accuracy evaluation false.

## Build the registered pilot

Install the root project's optional locked environment. Pillow belongs to this environment; do not use the separate MediaPipe interpreter to build the pages:

```sh
uv sync --locked --extra ml
uv run --extra ml python scripts/build_pose_evidence.py \
  --run artifacts/pose/smoke-009-train-0/run.json \
  --run-sha256 772ab832647ccede39291b95b0179e54f5cf84e67c4af86980e1c1bf5691fb65 \
  --video data/golfdb/videos_160/0.mp4 \
  --output artifacts/pose/review-010 \
  --frame-index 0 --frame-index 37 --frame-index 68 --frame-index 87 --frame-index 137
```

Use a new output directory. Existing output is never replaced. The current v1 pilot binds exactly this source, completed extraction and five prospectively selected indices; arbitrary replacement sources or frame selection do not become a registered validation study. The [protocol](../experiments/pose-validation-v1.json) and [reference schema](pose-validation.md) define the study and strict scoring requirements. This is one TRAIN clip's software/diagnostic pilot, not a generalization study or empirical accuracy result.

Before rendering, the artifact loader checks the independently supplied expected run SHA, completed ledger/result binding, pinned model/configuration, recorded source/runtime identity and complete native frame correspondence. Rendering performs another full sequential decode in a child with a 45-second limit, matching every actual native PTS/time base and coded raster. Source bytes are checked before and after decode. At most 12 selected frames and 12 million total selected source pixels are allowed; v1 uses five small frames. Full-resolution PNGs are encoded without changing raster dimensions, rotation, crop or aspect. Derivation source and protocol identities are snapshotted before rendering and rechecked immediately before publication. A source/protocol or artifact provenance change removes the incomplete output bundle. No inference or training is repeated.

The fixed frames are zero-based decoded indices. Human-facing frame numbers are index plus one. Recording-time native seconds and the model's rounded integer-ms API clock are displayed separately; neither is reconstructed from FPS or browser video playback.

## Review and annotate

Open `reference.html` in a local browser before reviewing any diagnostic model output. Give the clean page to a separate reviewer if you have already inspected the overlay.

1. Select an anatomical landmark by its MediaPipe index/name. Left and right refer to the golfer's anatomy, not the screen.
2. Click a visible location in the image to record continuous image-edge pixel coordinates. The mapping is `x_normalized × width`, `y_normalized × height`; pixel centers range from `(0.5, 0.5)` through `(width−0.5, height−0.5)`. The page maps pointer positions through the displayed image size to the unchanged raster. It does not use `width−1` normalization.
3. Mark an occluded landmark or a landmark outside the image with its status and **no guessed coordinate**. Leave uncertain or unfinished landmarks unreviewed. Every frame retains all 33 IDs, including unreviewed entries.
4. Add reviewer identity and notes. Only check blinded and independent declarations when they describe the actual annotation process. They begin unchecked and are not inferred from use of the page.
5. Export the reference JSON. Drafts can be exported, but unchecked declarations, missing reviewer identity or unreviewed marks do not silently become a validated study. The strict reference evaluator documents eligible coverage, abstentions and missingness rather than inventing scores.

Annotations remain in the tab's memory until exported. Reloading or closing the page discards unsaved work; export before closing. Export is a browser-local download. The pages contain no external resources, network requests or persistence services. Their Content Security Policy denies connections and external resources. Source frames embedded in exported pages remain subject to the original video's terms and privacy restrictions: keep them in ignored local storage and do not commit or redistribute them.

The image load handler rejects stale asynchronous loads, so quickly changing frames cannot attach an old image to a newer frame's identity. The annotation page itself uses only source pixels and manually placed marks; a completed annotation remains a human declaration that requires methodological review.

## Optional manual-event arithmetic

The workbench does not create swing event labels or import GolfDB benchmark events as manual reviews. To recompute descriptive measurements, explicitly supply `--manual-events path/to/events.json` with this exact shape:

```json
{
  "schema_version": 1,
  "video_sha256": "the exact recording SHA256",
  "events": [
    {"name": "Address", "frame_index": 0, "source": "manual"},
    {"name": "Top", "frame_index": 68, "source": "manual"},
    {"name": "Impact", "frame_index": 87, "source": "manual"}
  ]
}
```

The indices above illustrate the format; they are **not reviewed event labels** for the example recording. A human must review and provide the actual marks. Names must be canonical, unique and increasing, and source must be manual. Actual event timestamps are looked up from the bound decoded frame records. The existing measurement extractor retains aspect-correct formulas, unvalidated reliability, missing-data reasons and empty advice. It refuses spatial conversion after any multiple-pose observation. No thresholds for an ideal swing, causal fix or coaching reliability are inferred.

## Remaining gates

A diagnostic overlay establishes inspectable rendering and frame correspondence. It does not establish anatomical accuracy, identity continuity, camera/mirror correctness, calibrated landmark confidence, original-source timing, browser video synchronization, coaching efficacy or CaddieSet feature equivalence. The upstream projection warning and coordinate audit are discussed in the study report. Product pose serving and automatic event/outcome coaching remain disabled until their separate admission gates pass.
