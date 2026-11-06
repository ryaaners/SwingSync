# Architecture — increment 1

React/TypeScript client -> same-origin `/api` (Vite development proxy) -> FastAPI -> SQLite and local video files. One lifespan-owned worker processes persisted queued jobs. Interrupted processing is marked interrupted at startup, never automatically repeated. Start exactly one Uvicorn worker. No remote inference, accounts, queues or external service.

This increment inspects video, preserves presentation timestamps, and provides playback/history/deletion/HTML report. `completed` means inspection finished, not trained swing analysis. Models, event timelines, pose measurements and coaching are unavailable. No fabricated results are emitted.

## Shared API

GET/POST `/api/analyses`; multipart `file`, `camera_view` (face-on/down-the-line), `handedness` (right/left), `club`, `mirrored`. GET `/api/analyses/{id}`, `/video`, `/report`; DELETE `/api/analyses/{id}`. GET `/api/health`.

Records contain id, status, progress, error, metadata, video, empty events/measurements/observations, and explicit model availability. Source PTS and playback-relative times must remain distinct. Browser video codec support is independent from decoder support; rejected playback must have a useful message.

## Planned research boundaries

Audited GolfDB loader -> frozen shared MobileNetV2 features -> independent frame head and bidirectional LSTM head -> standard PCE evaluation. Optional ordered decoding reported separately. Pose feature extractor and templates must suppress unreliable results. Tabular outcome research stays separate until paired feature equivalence and golfer-held-out performance pass.

## Local trust boundary

Bind to loopback only. This local single-user application has no authentication and must not be exposed to a network. Upload limits and decode limits bound resource use; use no shell-based video commands. Local files persist until deleted through history; no automatic upload elsewhere. Do not run multiple backend instances against one storage directory. More robust worker isolation/timeouts are required before broader use.

## Offline research preparation (increment 002)

MAT loader -> immutable annotations + exact bytes fingerprint -> deterministic source-grouped inner partitions preserving each official outer fold. Read-only evaluation helpers implement the upstream full-clip protocol, including chronological target-slot behavior on anomaly677. Timeline inspection decodes exact contiguous frames without FPS guessing, records real PTS and provenance, and never calls re-encoded clip time original-source time. This module is not wired to user-facing model inference. Padding is excluded in evaluation; model/augmentation/feature-cache contracts remain future work.

## Model smoke infrastructure (003a)

Optional ML dependencies support FrozenMobileNetV2 -> provenance-checked full-clip feature cache -> independent frame or packed BiLSTM head. Weighted CE uses valid-prefix masks; no padding loss or backward-context leakage. CPU checkpoints restore model/Adam/RNG/provenance with weights_only loading and atomic no-overwrite publication. A durable smoke-run record precedes work and a training-root lock prevents duplicates; failed/unfinished state is not automatically called complete. The user-facing backend remains disconnected from these smoke artifacts until full training/evaluation and inference gates pass.

## Dataset preparation and full-head driver (003b)

A resumable feature job validates an immutable full media inventory and selected source partitions, atomically publishes each cache+ledger progress, and explicitly distinguishes partial/interrupted/failed/completed. Registered training admission requires complete matching caches; the CPU driver shuffles whole clips deterministically, uses all training labels for weights, and selects only on complete inner-validation epochs. Data cursor/order, RNG, optimizer and selection survive checkpoint resume. One repository lock serializes cache, smoke and head CLIs. No feature-preparation state or synthetic validation score enables user inference.

## Movement contract and calculations (004a)

An independent NumPy module accepts explicitly supplied MediaPipe-anatomical landmarks and event timestamps, produces six descriptive values with evidence or unavailable reasons, and always suppresses advice. It does not load a pose model or connect unvalidated calculations to the user app. Geometry is aspect-correct, view-scoped, normalized to an address torso reference, and does not use monocular z. See movement-measurements.md for exact formulas, quality policies and empirical-validation gates. It is isolated from the frozen event-training source and dependency lock.
