# Architecture

React/TypeScript client -> same-origin `/api` (Vite development proxy) -> FastAPI -> SQLite and local video files. One lifespan-owned worker processes persisted queued jobs. Interrupted processing is marked interrupted at startup, never automatically repeated. Start exactly one Uvicorn worker. No remote inference, accounts, queues or external service.

This increment inspects video, preserves presentation timestamps, and provides playback/history/deletion/HTML report. `completed` means inspection finished, not trained swing analysis. Automatic event models, pose measurements and coaching are unavailable; manual event review is now provided below. No fabricated results are emitted.

## Shared API

GET/POST `/api/analyses`; multipart `file`, `camera_view` (face-on/down-the-line), `handedness` (right/left), `club`, `mirrored`. GET `/api/analyses/{id}`, `/video`, `/report`; DELETE `/api/analyses/{id}`. GET `/api/health`.

Records contain id, status, progress, error, metadata, video, explicit model availability, manual events/recording-time measurements when marked, and empty coaching observations. Source PTS and playback-relative times must remain distinct. Browser video codec support is independent from decoder support; rejected playback must have a useful message.

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

## Fixed held-out evaluation and latency (006)

A prospective hash-bound contract admits completed selected heads from the original immutable training catalog. An independent test-only feature job preserves the same pipeline. Evaluation checks exact source-disjoint coverage and fresh decoded PTS, persists both heads/contexts per clip, and publishes overall/per-event results only after complete coverage. Explicit resume validates prior input/result hashes and arithmetic without repeating durable clip inference. A separate train-only CPU timing tool measures actual streaming extraction and selected heads, with initialization/warmup separated from measured stages. Neither tool changes the application's unavailable-model state. See heldout-evaluation.md for the fixed protocol.

## Manual annotations and canonical evidence (007)

User-selected decoded frame indices map to inspected relative/native PTS and produce only three descriptive tempo quantities. Each explicit save transactionally checks a revision and stores the full manual event snapshot, preventing stale overwrite/deletion resurrection. React retains dirty drafts across polling, explicitly reloads conflicts and prevents old responses undoing newer saves. Canonical frame PNGs and report images verify original upload SHA256 and full decoded PTS through a bounded subprocess, use coded orientation and two shared decoder slots, and never come from an unverified browser canvas. Reports retain current/superseded revisions, images and recording identity. No derived images persist server-side. See manual-event-review.md for API, storage, timestamp and reliability contracts.

At execution007, pose runtime selection was separately audited in pose-runtime-audit.md without installation or model acquisition. The later isolated adapter is described below; root uv.lock and historical feature/training implementation fingerprints remain unchanged.

## Completed research evaluation and offline report packaging (008)

The fixed selected heads have completed one official held-out fold with independent all-clip replay. This creates research evidence, not automatic app availability. An stdlib-only report packager consumes an expected-SHA completed run and bound contract/manifest, checks every saved row/metric/coverage/aggregate and exclusively emits a small provenance report. It does not load checkpoints, caches, video or Torch. Post-hoc ordering diagnostics only describe saved peaks and never change decoding/selection. The raw operational run and replay remain ignored. See experiments.md and model-card.md for actual metrics and integration limits.

## Isolated offline pose observations (009)

An independent non-workspace `pose_runtime/` lock installs MediaPipe and native dependencies outside the event environment. The macOS launcher invokes that explicit interpreter under an OS deny-all-network profile because the wheel declares utilization telemetry. The child checks denied IPv4/IPv6 socket access before importing MediaPipe. Linux native imports are a separate CI check; Linux extraction awaits verified isolation. No product endpoint invokes this CLI.

The offline extractor holds the shared expensive-work lock, saves an exclusive job identity before work, verifies independently pinned model/member hashes and license policy, preflights every rational native PTS and derives a separate collision-checked integer-ms API clock. It then decodes again for synchronous CPU VIDEO inference. Full raw presence/visibility and per-frame statuses remain distinct from filtered measurement points. Missing/malformed observations stay missing; multiple detections prevent whole-sequence spatial conversion. Source, code, lock, runtime and asset identities are checked before completion. Atomic result publication is bound by a completed ledger; a final-ledger failure removes the result. No automatic resume, inferred identity reliability or advice is provided. See pose-extraction.md for limits and recovery, pose-assets.md for separate licensing/privacy evidence.

## Offline pose review and reference collection (010)

A separate completed-artifact loader admits an independently known ledger SHA, its bound result and exact media bytes. It checks registered pose/model/config/source/lock identities, raw-to-filtered semantics and all native PTS/raster metadata in a bounded decode. This authenticates historical observations and fresh frame correspondence; it does not claim fresh reproduction of the historical native runtime.

A second bounded renderer emits exact-raster clean/overlay PNG evidence, a diagnostic HTML report, and an independent clean-only reference page. The clean page contains no model coordinates and starts with null/unreviewed anatomical slots and unchecked reviewer declarations. Pointer positions map back to original image-edge pixels. A strict manual-reference contract and separate CLI report conditional agreement, coverage and abstentions; empty drafts produce no score. All product/coaching/outcome gates remain false. Renderer, measurement modules, protocol, root lock and generated files are fingerprinted with pre-publication drift checks. No pose inference or benchmark evaluation is repeated. See pose-review.md, pose-validation.md and remaining-work.md.
