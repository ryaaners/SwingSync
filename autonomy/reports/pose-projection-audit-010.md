# Projection warning audit — execution 010

This audit explains the observed warning using pinned official public source and independent synthetic geometry. It did not run inference, annotate human joints or measure empirical landmark accuracy. Full URLs, source byte hashes, line ranges and geometry fixtures are in [pose-projection-audit-010.json](pose-projection-audit-010.json). Downloaded source stays in ignored storage; no upstream source is vendored.

## What the warning establishes

At MediaPipe revision `2c2d4a7454d6358e0d6f0b54e254d231fe4cc78f`, the [projection calculator](https://github.com/google-ai-edge/mediapipe/blob/2c2d4a7454d6358e0d6f0b54e254d231fe4cc78f/mediapipe/calculators/util/landmark_projection_calculator.cc#L72) logs whenever a normalized rectangle is connected without image dimensions. The log precedes inspection of rectangle values. It therefore identifies a graph-input branch; it does not prove a nonsquare input image, a nonsquare ROI or an inaccurate output.

That branch rotates centered normalized ROI coordinates and scales x/y by normalized ROI width/height. The [pose landmark graph](https://github.com/google-ai-edge/mediapipe/blob/2c2d4a7454d6358e0d6f0b54e254d231fe4cc78f/mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarks_detector_graph.cc#L480) connects its letterbox-adjusted landmarks and pose rectangle to projection without the dimensions input. Both the [initial detection ROI](https://github.com/google-ai-edge/mediapipe/blob/2c2d4a7454d6358e0d6f0b54e254d231fe4cc78f/mediapipe/tasks/cc/vision/pose_detector/pose_detector_graph.cc#L143) and tracked ROI configure `square_long=true`, equal expansion factors and image size. This is consistent with the calculator's square-physical-ROI assumption, even when the full image is rectangular. The [upstream tests](https://github.com/google-ai-edge/mediapipe/blob/2c2d4a7454d6358e0d6f0b54e254d231fe4cc78f/mediapipe/calculators/util/landmark_projection_calculator_test.cc#L262) separately exercise a dimension-aware rotated crop on a nonsquare image.

## Independent arithmetic checks

Let image dimensions be W/H and normalized ROI extents rW/rH. When `rW*W == rH*H`, rotating centered ROI coordinates in the legacy formula is equivalent to rotating the square ROI in pixels and then dividing by each image axis. Nine synthetic cases cover square/landscape/portrait rasters and three rotations.

The assumption matters: with a 640×320 raster, normalized ROI extents (0.5,0.5), ROI point (0.75,0.5) and a 90° rotation, the no-dimensions formula gives (0.5,0.625), while independent pixel rotation gives (0.5,0.75): a 40-pixel y difference. This is a deliberately constructed formula counterexample, **not an observed native pose error**. It does not justify altering or reprojecting saved model coordinates.

The existing smoke raster itself is 160×160. Its native projection warning remains recorded. Public source does not authenticate every compiled-binary internal ROI, and internal per-frame rectangles were not exposed in that saved artifact. Therefore no universal coordinate-correctness or landmark-accuracy conclusion follows. The source-level cause is explained; empirical coordinate/identity/measurement validation remains open.

## Reference work completed and outstanding

The new protocol and evaluator define explicit pixel units, fixed source/frame identities, all 33 anatomical slots, manual missingness states, unchecked independence/blinding attestations, visible-point error and model abstention denominators. Fifty-one synthetic tests pass. The registered five-frame pilot is diagnostic software work: the existing pose output had already been observed, no manual reference coordinates have been created, and no accuracy result is reported. It cannot select thresholds, establish reliability, enable product integration or generate advice.
