# Local research event proposals from video

The research runner applies the already trained, hash-registered event heads to a generic local video without using labels or a GolfDBRecord in frame extraction. It does not enable the normal application's model availability, produce coaching advice, score swing quality, predict a shot outcome or validate accuracy on uploaded videos. The original annotations, split manifest and training catalog are still read by model admission to authenticate the selected checkpoints; they are not used as labels for the input recording. No outer-test cache or saved outer prediction is opened to choose a model.

## Run with the original ML runtime

Use the existing Python3.11 frozen ML environment and root dependency lock. This workstation has `.tools/ml-runtime/bin/python` for the immutable research environment. Do not install pose/tabular packages into it, change the root lock to make historical artifacts pass, or substitute a moving checkpoint. The runner verifies the exact selected `best_model` tensors, original completed training records, transform/backbone/runtime fingerprints and local upstream ImageNet weight file before inference. Weights are never downloaded during this command. Missing or changed artifacts produce an unavailable/failed state rather than fabricated proposals.

From the repository root, use a fresh output directory:

```sh
.tools/ml-runtime/bin/python scripts/propose_events.py \
  --video data/golfdb/videos_160/0.mp4 \
  --output artifacts/event-input/clip0-new \
  --contract experiments/heldout-v1.json \
  --annotations data/golfdb/golfDB.mat \
  --manifest artifacts/golfdb/preparation-002-all/fold-1.json \
  --protocol experiments/event-input-v1.json \
  --camera-view down-the-line --handedness right --club driver \
  --timestamp-basis reencoded \
  --rights 'Official GolfDB authors research clip; local research only, no redistribution grant. Metadata are operator declarations.' \
  --unmirrored --single-swing-confirmed
```

The example metadata are declarations about a reviewed input, not inferred anatomy, mirroring or identity validation. Supply truthful metadata for the actual recording. Clip1 is slow-motion footage; neither video PTS nor a successful proposal validates physical swing tempo. No automatic timing measurements are computed.

The supplied model contract bytes and root lock must match the pinned input protocol before any native decoder or model setup. The command runs a separate native process with a600-second limit; preflight decoding has a45-second child limit. If a process is interrupted or times out, inspect its PID, output ledger and shared expensive-work lock. Do not assume completion or automatically retry. The API intentionally has no resume path and refuses an existing output directory. The CLI is trusted-local research tooling, separate from the app's upload worker.

## Pixels and clocks

The input contract requires explicit supported view/handedness/club, unmirrored reviewed single-swing declarations, source rights and timestamp basis. It admits one upright coded video track, explicit square pixels, constant supported raster and bounded size/duration/frame count. Unknown/nonzero rotation, unverified display matrices, non-square/unknown aspect, missing/duplicate/decreasing/unrepresentable PTS or budget violations fail closed. See [event-input-contract.md](event-input-contract.md) for exact limits and geometry.

Every frame keeps zero-based decoded index, integer native PTS, rational time base and float64 actual seconds. Nothing is reconstructed from FPS, dropped, resampled or interpolated. The decoder first preflights the full recording. A second sequential decode must reproduce all PTS and dimensions. RGB pixels then follow the original deterministic tensor transform: resize short edge232, center crop224, bilinear antialiasing and ImageNet normalization. There is no new golfer ROI crop or annotation-derived bounding box. A bounded16-frame batch feeds the same frozen backbone; the final partial batch is retained. No model or preprocessing is fit.

Both the frame and temporal selected heads use the fixed **reference64-reset** context. Temporal recurrence starts afresh every64 frames, including the unpadded tail. Per-frame nine-class post-softmax values are retained; eight event proposals use the unchanged NumPy default argsort's last index independently per event. No chronological sorting, duplicate removal, order-constrained decoder or confidence calibration is applied. A selected event may share a frame with another event or be out of phase order, and background can exceed the selected event score. These are explicit diagnostics, not repaired results or validated reliability.

## Output and publication

The shared `artifacts/training/active-training.lock` prevents overlap with another expensive research job. An exclusive `run.json` is saved before model setup/inference with job/PID/time, input SHA, metadata, protocol and source identity. Full admitted model, source, frozen transform/runtime and dependency identities are added before forwards. Atomic no-clobber publication produces:

- `arrays.npz`: float32 features `[T,1280]`, float32 frame/temporal probabilities `[T,9]` and float64 native `pts_seconds[T]`; no labels or fake historical source indices.
- `result.json`: all native frame metadata, both heads' eight proposal rows with scores/background values, duplicate/order diagnostics, primary temporal head, exact context, selected checkpoint metadata, array checksum and explicit false availability/accuracy/domain/physical-time/serving/coaching-related gates.
- Completed `run.json`: binds the actual result and array SHA256 values. Source/media/model/runtime checks occur after forwards, after result publication and after completed-ledger publication. Any detected drift/fatal failure removes unbound arrays/result and records failure. A completed job is not empirical validation.

Scores are model quantities, not calibrated probabilities of correctness. Any renderer must authenticate the completed ledger/result/arrays and source bytes, independently check frame correspondence and preserve those warnings. A predicted event is not a manual correction. No prediction automatically creates descriptive tempo or advice.

## What parity can establish

The registered pilot compares the generic extractor with existing frozen **training** caches on identical decoded source images and the same CPU/runtime/batch configuration. Exact features, timestamps, selected-head probabilities and event peaks can establish software consistency. They cannot establish raw-phone-video crop equivalence, original-source frame mapping, camera/mirror correctness, independent golfer generalization, event-error confidence, browser/audio synchronization, physical tempo or coaching efficacy. The two pilot clips share a source/golfer and must not be counted as independent examples. Existing outer test results must not be reused for tuning.

The normal app remains model-unavailable until its separate admission requirements are satisfied. CaddieSet feature equivalence and outcome serving remain independently closed. All raw videos, checkpoints, predictions and generated artifacts stay ignored; source/media/model redistribution rights are separate from local research execution.
