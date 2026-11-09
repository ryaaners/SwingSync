# Local pose extraction

This research CLI produces actual local pose observations. It does not enable the
normal app's pose overlays, automated events, shot predictions or advice. Golf
landmark accuracy, identity continuity and measurement reliability have not been
validated. A successful extraction demonstrates execution and frame correspondence.
Synthetic tests demonstrate software behavior only.

## Separate environment and model

Use Python 3.11 and the standalone `pose_runtime/pyproject.toml` and `uv.lock`.
The root ML lock and historical experiments remain unchanged. The native stack
uses MediaPipe 1.2.2, NumPy 2.2.5, OpenCV contrib 4.12.0.88 and PyAV 16.0.1.
Install the complete lock into `.tools/pose-runtime`; see the environment audit
for commands and platform limits. Inference imports these packages only in that
explicit separate process. All installed distribution versions must occur in the
pose lock and the four direct versions must match the registered pins.

Acquire the explicitly pinned official bundle once, outside inference:

```sh
python3 scripts/acquire_pose_asset.py --directory artifacts/pose
```

The helper records an exact byte/member manifest and separate license evidence.
It does not copy an upstream license onto original SwingSync code. Model member
redistribution attribution remains unestablished; do not distribute the weights.
Missing, changed, re-labeled or unregistered assets are refused. Even a jointly
changed model and manifest cannot change the independently pinned registry.

MediaPipe's inspected NOTICE says performance/utilization metrics are transmitted.
No supported Python opt-out has been verified. Therefore native extraction must
run through the supported OS network-denial launcher, not the direct extractor:

```sh
python3 scripts/run_pose_offline.py \
  --video data/golfdb/videos_160/0.mp4 \
  --model artifacts/pose/pose_landmarker_full-v1.task \
  --model-manifest artifacts/pose/pose_landmarker_full-v1.manifest.json \
  --output artifacts/pose/clip0-observations \
  --camera-view down-the-line --handedness right \
  --timestamp-basis reencoded \
  --rights 'Official GolfDB research clip; local research only, no redistribution grant' \
  --unmirrored --single-golfer-confirmed
```

For the clip0 smoke, down-the-line/right/unmirrored/single-golfer are explicit
operator metadata assumptions, not independently validated labels. The operator
must actually review camera view, handedness, mirror convention and
single-golfer input; the flags are declarations, not model validation. The example
values are not automatic labels for arbitrary clips. The verified launcher currently supports macOS only and applies a600-second
subprocess timeout. Linux isolation has not been verified and extraction is
unavailable there. Unsupported platforms fail rather than silently execute
without verified egress isolation. The native
extractor additionally requires OS-denied IPv4 and IPv6 numeric-loopback socket
probes before importing MediaPipe. Connection refusal, timeout or unsupported IPv6
is not accepted as proof of denial. The probes are guards; the launcher installs
the deny-all-network policy. No LLM, cloud model or API is used.

## Inputs, outputs and recovery

The input must have exactly one video stream and explicit square pixel aspect,
upright coded raster, no unverified display matrix, fixed dimensions at most
3840 by 2160, a file at most150MiB, at most3600frames and an observed PTS span at
most30seconds. Only supported local MP4/MOV/WebM/AVI demuxers and file protocol
are admitted. This trusted-local research CLI is separate from the app's
subprocess-limited upload worker. It is not a general untrusted-media service.

A nonblocking shared `artifacts/training/active-training.lock` prevents overlapping
expensive work. The output directory is created exclusively. Before native work,
`run.json` records unique job/PID/time, source SHA, explicit metadata, exact asset
manifest/hash, full environment lock/installed versions/platform/executable,
adapter/launcher/registry source hashes, CPU delegate, thresholds and limits.
Native library versions/build information are added before inference. There is no
automatic resume. If interrupted, inspect the process, lock and saved identity;
never relaunch into the same directory or assume a launched job completed.

The decoder first inspects every bounded frame and rejects invalid clocks before
constructing the VIDEO detector. A second sequential decode must match every
native integer PTS, rational time base and raster dimension. Source/model/runtime/
source-code fingerprints are checked again before completion. Fatal native or
provenance failures mark the ledger failed and publish no completed result.
Successful `result.json` publication is atomic and no-clobber; the completed ledger
binds its SHA256. A final-ledger publication failure removes the completed result.

Every source frame has one observation row, preserving:

- Native integer PTS and rational time base; unchanged float64 actual seconds.
- Separate relative integer API milliseconds using exact rational half-up rounding.
- Frame index, raw normalized coordinates, visibility and presence for every return.
- Explicit `no_pose`, `multiple_poses`, `malformed_pose` or `single_pose` status.
- Measurement-contract landmarks and any presence-filtered anatomical indices.

Integer API milliseconds never replace measurement time. Clock collisions,
non-increasing PTS, float64 precision loss and signed-int64 overflow fail closed;
there is no jitter, frame dropping, FPS reconstruction or silent IMAGE fallback.
API rounding is bounded by0.5ms mathematically, not a landmark-accuracy bound.

No/multiple/malformed poses yield all-null measurement points (NaN when converted
into `PoseSequence`), with explicit raw observations/status. Nothing is carried
forward or interpolated. A single returned pose does not prove that the scene
contains only one person or that identity stayed stable. Any multiple-pose frame
marks the entire sequence identity ambiguous: raw extraction is retained, but
`as_pose_sequence` refuses the whole spatial sequence even if another field claims
identity is unambiguous. Each landmark must have
finite coordinates and separate finite presence/visibility scores in[0,1]; no
score is invented. The0.5presence floor masks measurement points while retaining
raw presence and visibility. This is an engineering filter, not validated golf
reliability. Out-of-image x/y values are retained rather than clamped. Anatomical
IDs never change; no guessed left/right swap or calibrated3D/depth claim is made.

`backend.swingsync.pose_runtime.contracts.as_pose_sequence` constructs the existing
immutable `[T,33,4]` measurement contract plus actual PTS from a completed artifact.
This conversion is arithmetic, not an artifact authenticity verifier. Callers must
verify the completed ledger/result SHA and provenance first. Advice is always
empty and `validated_for_coaching` always false. Empirical validation, overlay/
browser correspondence and a permitted video-to-report demonstration are later
gates. CaddieSet feature equivalence remains closed. The actual CPU smoke logged an
upstream `landmark_projection` warning about `NORM_RECT` without
`IMAGE_DIMENSIONS`; successful coordinate output does not resolve that limitation
or establish landmark accuracy. Preserve the native log in the run evidence.
