# Descriptive movement calculations — software contract

Status: pure measurement arithmetic is implemented; no MediaPipe inference adapter, empirical pose validation, user-facing measurement integration or coaching efficacy is claimed. The app still reports model unavailable. Input is a supplied MediaPipe-anatomical 33-landmark sequence, not interchangeable HRNet/CaddieSet features.

## Inputs and provenance

PoseSequence owns read-only copies of actual frame PTS seconds and T×33×4 values (normalized image x/y, unused z, visibility), with image width/height, face-on/down-the-line/unknown view, handedness and a mirrored-coordinate flag. Pixel geometry multiplies x by width and y by height; computing Euclidean angles directly in unequal normalized axes would distort geometry. Anatomical left/right indices are preserved; mirroring reflects x only. Re-inference on mirrored imagery is not asserted to preserve anatomical identity.

PoseEvent records one canonical event name, real timestamp, manual/predicted origin and optional confidence. Confidence is not validated event reliability. Duplicate/nonchronological events, missing coverage or required events make affected measurements unavailable. Manual event corrections create a new event sequence and recompute values and evidence; this is a library capability, not a completed video-app correction interface.

## Six available descriptive calculations

| Output | Definition and evidence |
| --- | --- |
| Backswing duration | Top time minus Address time, using real supplied seconds |
| Downswing duration | Impact time minus Top time |
| Tempo ratio | (Top−Address)/(Impact−Top); no ideal ratio is asserted |
| Normalized maximum sampled head displacement | Over Address..Impact, maximum sampled 2D movement of the mean of nose and both ears, divided by Address shoulder-midpoint-to-hip-midpoint image-plane length |
| Lead elbow angle at Top | Projected shoulder–elbow–wrist angle; anatomical left for right-handed golfers, right for left-handed golfers |
| Trail elbow angle at Top | The other arm's projected image-plane angle |

Durations are recording-time quantities. Slow-motion or time-remapped video does not establish original physical swing duration; nonuniform speed ramps can alter the ratio too.

All spatial values are specific to the supplied view, camera, crop and perspective. They do not measure calibrated 3D rotation, anatomical joint flexion, pressure, force or clubface angle. Camera movement remains a confound. Head movement is a sampled image-plane quantity, not the continuous motion maximum or a universal technique score.

## Fail-closed quality policy

Selected spatial points must be finite, in image bounds and visible; missing head points do not trigger a switch to another centroid. Body-scale and arm degeneracy suppress values. Spatial event frames must coincide with actual observed PTS (1e-9-second numerical tolerance); there is no FPS guessing, interpolation or nearest-frame snapping. Between-frame manual event times can still support timestamp arithmetic, with bracketing frame evidence. Gaps over a configurable budget suppress the head trajectory. Unknown view suppresses spatial values.

Visibility0.5 and sample-gap0.25seconds are explicit configurable engineering filters. They are not empirically calibrated reliability or biomechanical standards. Every result includes its value/unit or unavailable reason, event/interval, frame indices and PTS, source/confidence provenance, method/reference and unvalidated reliability. Advice is always empty. A manually supplied event does not validate the pose or justify a practice suggestion.

## Validation still required

Install/audit a compatible local MediaPipe runtime and pose asset with separate code/data/weight provenance. Validate landmarks and these projected quantities against reviewed 2D references on permitted media across views, handedness, mirroring, visibility and body scales. Quantify sampling and event-timing error sensitivity, then define data-backed reliability/suppression criteria. Only after that should descriptive values enter the normal video flow; advice needs its own referenced evidence. Paired CaddieSet feature equivalence and grouped generalization remain separate closed gates.

Sources define landmark indices and coordinates, not golf ideals: [MediaPipe landmark enumeration](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmark), [pose coordinate documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker). Synthetic tests establish arithmetic/contract behavior only.
