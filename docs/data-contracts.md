# Data contracts

Audit date: 2026-11-04 (America/Toronto). Contracts below separate inspected facts from required future validation. No event or outcome model has been trained in SwingSync.

## GolfDB

Inspected upstream revision: `a63b4ce8c0900d09abbe92519223efe16810f5f4`.

Sources: [repository](https://github.com/wmcnally/golfdb), [MAT conversion and official splits](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/data/generate_splits.py), [reference loader](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/dataloader.py), [preprocessing](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/data/preprocess_videos.py).

The downloaded `data/golfDB.mat` has SHA256 `36523fcef016d93334691ea058d643bcf0d3a4a8c70b733f3b4cc0b8e5e0cf0a`, 1,400 annotations and 580 unique `youtube_id` values. MATLAB structured fields are `id`, `youtube_id`, `player`, `sex`, `club`, `view`, `slow`, `events`, `bbox`, `split`. Each events array contains ten entries. The first and last delimit trimming; the interior eight are the target events. `bbox` supplies normalized x, y, width, height for the source image. The split field contains 1–4, with exactly 350 rows each. Actual metadata audit found zero shared YouTube IDs between any pair of outer splits.

| Class | Event |
| --- | --- |
| 0 | Address |
| 1 | Toe-up |
| 2 | Mid-backswing |
| 3 | Top |
| 4 | Mid-downswing |
| 5 | Impact |
| 6 | Mid-follow-through |
| 7 | Finish |
| 8 | Background / no event |

The reference loader subtracts `events[0]` and labels `events[1:-1]`, in that order. Never treat the trim boundary as Address. Copy annotation arrays before subtracting offsets; the reference loader mutates its local array in place. For the first row, original events are `[408,455,473,476,490,495,498,501,514,545]`; trimmed event indices are `[47,65,68,82,87,90,93,106]`.

The preprocessing code increments its source-frame counter before testing the inclusive trim interval. Therefore the first preprocessed frame maps to source decoded index `max(1, trim_start) - 1` (the counter begins at one even for a zero trim boundary) under a zero-based decoder. A new source-video loader must explicitly test this conversion; blindly subtracting the trim start from a zero-based source index would introduce a one-frame error. Benchmark clip indices should reproduce the reference loader; source frame PTS must be retained independently. Existing preprocessed clips were re-encoded at reported FPS; they cannot prove preservation of the original source's variable timestamps.

### Split and sequence requirements

For official outer fold k: test rows have `split == k`; training candidates have `split != k`. The reference filenames call the test rows `val_split_k`, but SwingSync must not use them for tuning. Derive internal validation by grouping candidate rows on `youtube_id`, save row IDs and seed, assert group disjointness, and never split frames. Player identity may recur across recordings, so official source-disjoint evaluation is not evidence of unseen-golfer generalization.

Contract for future samples: RGB frames `[T,3,H,W]`, integer labels `[T]`, valid mask `[T]`, original source frame indices `[T]`, source PTS seconds `[T]`, clip ID, source recording ID, and transform metadata. Padding must never contribute loss, argmax, or recurrence context. Use packed sequences or equivalent masking for bidirectional recurrence. Apply one randomly sampled spatial transform consistently over the sequence. Frozen-backbone feature caches must include decoder/transform/backbone/version fingerprints. Do not wrap end-of-clip frames into a false contiguous sequence merely because the historical loader did so.

### Official PCE contract

Sources: [correct_preds](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/util.py), [evaluation](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/eval.py).

For one full clip, the reference recovers eight occupied event positions in chronological order with `np.where(labels < 8)[0]`, ignoring class identities. Tolerance is `int(max(np.round((events[5] - events[0]) / 30), 1))` using that chronological vector. On normal ordered annotations these are Impact and Address. This uses NumPy rounding, not a fixed percentage of video duration or an FPS-to-seconds conversion. Each event prediction independently maximizes its class probability over clip frames. An event is correct when its absolute frame error is at most tolerance. Overall PCE is the mean of the per-clip eight correctness indicators. Report per-event means too. The reference evaluates chunks of 64 frames and resets recurrent state per call; changing temporal context must be recorded. Tied probabilities need a documented, tested policy to reproduce the reference `argsort(...)[-1]` behavior. Report order-constrained decoding separately from this benchmark. Millisecond error must use actual PTS of prediction and annotation frames, and disclose when only re-encoded clip timestamps are available.

The reference training code uses weighted cross entropy: each event class weight 1/8 and background weight 1/35. SwingSync should compare its train-only frequency-derived weighting or preserve this explicit reference configuration; test data must not set class weights. Both planned models share MobileNetV2 image features. Start with frozen features, then a linear frame classifier versus a BiLSTM head. Record any departure from the historical architecture or preprocessing.

## CaddieSet

Inspected revision: `3c73d9d40580bb8a5a10711ad1fa10735a205ffe`. Sources: [repository](https://github.com/damilab/CaddieSet), [CSV](https://github.com/damilab/CaddieSet/blob/3c73d9d40580bb8a5a10711ad1fa10735a205ffe/data/CaddieSet.csv), [paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Jung_CaddieSet_A_Golf_Swing_Dataset_with_Human_Joint_Features_and_CVPRW_2025_paper.pdf).

The downloaded CSV SHA256 is `646459b081863e244d75efe6f09fad4d34750d2fed7fde44db654e7720a9a306`. It has 1,757 records, 80 columns, and eight independent golfers. Of these columns, 69 are event-prefixed swing features, two are metadata (`View`, `ClubType`), one is grouping (`GolferId`), and eight are ball outcomes. There is no shot identifier, timestamp, source-video filename, handedness field, or raw-landmark coordinate table. The complete GitHub tree contains only README, LICENSE, and the CSV; no videos, feature-extraction code, or model weights were present.

| Field group | Treatment |
| --- | --- |
| `DirectionAngle` | First research target; signed degrees of initial departure relative to target line; negative left, positive right. Not slice/hook curvature. |
| `Distance`, `Carry`, `LrDistanceOut`, `SpinBack`, `SpinSide`, `SpinAxis`, `BallSpeed` | Exclude from every predictor set, including transformations or missingness indicators of these outcomes. |
| `GolferId` | Grouping only; never predictor or embedding. |
| `View` | FACEON / DTL; evaluate separately initially because feature support differs substantially. |
| `ClubType` | W1, W3, I4–I9; candidate metadata with explicit unseen-club fallback. |
| `0-...` through `7-...` | Event-specific candidate features only after numeric cleaning and train-only preprocessing. |

The paper describes the original 924 FACEON shots and 40 features, while the current CSV additionally includes 833 DTL records and 69 feature columns. Treat paper/current-release mismatch explicitly. The paper establishes DirectionAngle degrees, Distance/LrDistanceOut yards, and feature angles/ratios. Do not assume units of unneeded ball fields from their magnitudes. It describes joint extraction using fine-tuned SwingNet, Faster R-CNN and HRNet's 17 keypoints, and scaling coordinates by bounding-box width. Thus these are automatically selected events, not guaranteed manually annotated event frames. MediaPipe's 33 landmarks and a different normalization are not interchangeable inputs.

Numeric audit: 58,917 empty cells; 11 nonnumeric `#NAME?` feature cells; zero missing DirectionAngle values (range -18.4 to 20.9 degrees); six exact duplicate rows. Missingness is strongly structural by view: several columns have 833 or 924 missing cells. Convert known spreadsheet errors to missing, reject unexpected tokens, and fit imputation/drop decisions only in training partitions. Retain audit counts rather than silently dropping rows. Do not infer feature availability from an all-data missingness selector.

| GolferId | FACEON | DTL | Total |
| --- | ---: | ---: | ---: |
| 1 | 112 | 123 | 235 |
| 2 | 165 | 165 | 330 |
| 3 | 230 | 230 | 460 |
| 4 | 74 | 61 | 135 |
| 5 | 182 | 170 | 352 |
| 6 | 109 | 33 | 142 |
| 7 | 7 | 6 | 13 |
| 8 | 45 | 45 | 90 |

Club use is confounded with golfer: golfers 6 and 8 have only W1 records; golfer 5 has 340/352 W1 records; golfer 3 supplies 36/46 W3 and 36/45 I8 records. All golfers have both views, but golfer 6 has 109 FACEON versus 33 DTL. Report these imbalances; eight golfers do not support broad population claims. There are 803 groups sharing golfer, club, and all eight ball outcomes across both views. This is evidence consistent with alternate views, not an authoritative shot key. Holding the entire golfer together prevents these potential pairs crossing partitions. Do not publish row-random estimates.

### Outcome integration gate: CLOSED

Unblock conditions: obtain authorized source videos or paired landmark examples, exact feature formulas (including signs, head point, denominator, angle convention and missingness), camera and handedness conventions, event mapping and extraction configuration, and authoritative shot IDs. Build paired numerical agreement checks between the upstream extractor and the proposed video extractor. Quantify error caused by predicted versus reviewed event timing. Then require held-out-golfer outcome improvement over the club-conditioned mean baseline. Until all pass, keep outcome experiments a separate tabular research module. Neither shared feature names nor feature importance can justify causal coaching suggestions.

## Licensing and redistribution

GolfDB README explicitly licenses repository code CC BY-NC 4.0. The linked data videos and weights need separate provenance/terms review; that code statement alone is not evidence of video copyright permission or a distinct weights grant. Do not vendor reference code or redistribute data/weights in ordinary Git. Cite McNally et al., GolfDB, CVPR Workshops 2019. The reference repository's reported results are not SwingSync results.

CaddieSet root LICENSE is MIT, copyright 2024 damilab, with notice retention. There is no separately scoped video/weights license or released video/weights artifact in the inspected tree. Record the repository license for its provided files without assuming rights to unreleased recordings. Cite Jung et al., CaddieSet, CVPR Workshops 2025. Original SwingSync code must not automatically inherit either upstream license; the project owner still chooses its license.

## Increment 002 implementation and newly discovered anomaly

`backend/swingsync/research/golfdb.py` reads immutable records from the actual MAT bytes and saves deterministic source-grouped partitions. `evaluation.py` implements the standard reference evaluator; `timeline.py` validates inclusive trim counts and maps decoded PTS with explicit source versus re-encoded provenance. `scripts/golfdb_prepare.py` creates all four manifests and optional archive/clip integrity records, refusing to overwrite a run directory.

Actual clip **677** has original annotations `[794,886,908,925,940,952,962,961,1036,1071]`: Impact and mid-follow-through are reversed. Preserve these labels. The loader flags the anomaly; no repair or exclusion is performed. The evaluator preserves the class-specific input `(92,114,131,146,158,168,167,242)` but compares against chronological `(92,114,131,146,158,167,168,242)` to reproduce upstream. Its tolerance is two frames, versus three if class-specific Impact were substituted. Per-event benchmark names denote official slots; they are not guaranteed semantic target names on anomalous rows. Standard independent prediction may itself be nonchronological; this is not order-constrained decoding. Tie behavior uses default NumPy argsort and records NumPy2.2.5; equality across other NumPy builds is not promised.

The authors' linked `videos_160.zip` was acquired for local research only, SHA256 `e99d4dfc8516656b56d62b95042f959f8a1f892c657093bd8e541229e0f2f053` (698,989,423 bytes). Archive inventory contains exactly 1,400 matching MP4 members. Files remain ignored under data/golfdb; no clips, contact sheets, annotation bytes or weights are redistributed. Public provision for research does not establish unrestricted media redistribution rights, and no separate data/weights license was found in the inspected README. Original-project licensing remains unchanged.

All 1,400 reference clips passed exact decoded frame-count and strictly increasing PTS checks. Real clips 0 and 677 decode to 138 and 278 frames respectively, exactly their historical inclusive trims. Event contact sheets were inspected locally: clip0 is coarsely plausible in event order, clip677 retains its known inversion. This is a basic visual sanity check, not qualified coaching review, exact source-frame equivalence or annotation revalidation. Their actual decoded PTS belong to the re-encoded clips. Original YouTube PTS cannot be recovered from that archive. Browser event synchronization remains separately unverified.

## Frozen feature and smoke-training contract (003a)

Feature caches keep float32[T,1280] image embeddings, class labels, actual re-encoded-clip PTS and historical source indices together. Identity includes video/annotation hashes and unchanged events, exact backbone tensor/buffer bytes, pretrained initialization provenance, deterministic transform metadata, Torch/torchvision/NumPy/PyAV/FFmpeg/Python versions, extraction device, and local pipeline/backbone runtime-code hashes. Pickle-free loads check schema, checksums and semantic label/index consistency; these detect accidental corruption, not an attacker rewriting both data and metadata. Cache publication is atomic and never overwrites a completed cache. Decode keeps only a bounded RGB batch in memory. No augmentations are present; introducing them requires consistent per-sequence transforms and distinct cache identity.

Frame and temporal heads share exactly those features. Heads and weighted CE exclude suffix padding; the temporal model uses packed sequences so backward context cannot see padding. Class counts must be fit on the relevant training partition only. Explicit CUDA/MPS/CPU detection does not establish latency or numerical equivalence across devices.

Torch2.9.1/torchvision0.24.1 are an officially paired Python3.11-compatible stack, optional via --extra ml. CPU wheels are selected for non-macOS in the committed lock to keep CI/demo practical; macOS uses PyPI wheels with MPS capability. CUDA is detected by code, but this CPU lock does not install a CUDA-enabled distribution. A future CUDA environment needs its own deliberate, tested dependency resolution. ImageNet V2 weights were acquired from the official torchvision URL for local research and fingerprinted; no weights are redistributed or described as SwingSync-trained. Upstream torchvision code is BSD-3-Clause; pretrained dataset/weight terms remain distinct, not a license for original SwingSync code.

## Dataset-job and registered experiment boundary (003b)

A cache job inventories exactly the requested source-disjoint partitions and fingerprints the media before extraction. Every clip identity must match that initial inventory, including late queued clips. Resume validates actual cache bytes/labels/PTS against saved identities and rejects configuration, source, weights, pipeline or dependency changes. A valid cache published before ledger update may be recovered without recomputation. Null/unknown completed entries do not count as completed work. All operational CLIs share one absolute repository lock; test injection of isolated locks is not a production CLI override.

The full head-training CLI requires a completed train+inner-validation job with the hash-registered ImageNet MobileNetV2 state, default transform and runtime feature signature, then independently checks actual NPZ signatures. Merely having 1280 columns or similar feature names is insufficient. The generic head-driver API remains flexible for explicit synthetic fixtures; the operational CLI fails closed on unregistered feature pipelines. It has no outer-test evaluation path. Exact CPU continuation is bounded to identical code/config/data/runtime; cross-device/version equivalence is not assumed.

## Manual uploaded-video annotation boundary (007)

Manual marks accept only named zero-based decoded frame indices and an expected revision; recording/native PTS are read from SHA256-bound inspection evidence. UI human frame numbers are one-based, explicitly distinguished in exported reports. Strict chronological ordering is a user annotation contract, not a repair of GolfDB benchmark labels. No timestamp is guessed from FPS or browser playback, and no pose is fabricated. Canonical coded-orientation PNG evidence and manual corrections are described in manual-event-review.md. Manual timing is unvalidated and yields no advice; benchmark, pose and outcome gates remain independent.

## Offline pose observation boundary (009)

The separate research extractor preserves integer decoder PTS/time base and float64 recording seconds for every frame. Exact rational relative times generate a separate half-up integer-ms VIDEO API clock; collisions/overflow are rejected before task construction, never repaired by jitter, FPS reconstruction or dropped frames. Raw normalized x/y/z/visibility/presence are retained, with absent/malformed/multiple detections explicit. Presence filtering is an engineering policy; anatomical IDs remain unchanged, z is unused for measurements, and any multiple detection refuses whole-sequence spatial conversion. Operator view/hand/mirror/identity declarations are assumptions, not calibrated reliability. The existing immutable PoseSequence conversion requires authenticated completed provenance from the caller and does not establish accuracy.

Artifacts bind exact source SHA, reencoded/original timestamp basis, image geometry, both clocks, model/member registration, isolated lock/runtime/native libraries and adapter/launcher hashes. The network-denied macOS launcher and pre-import denial guard address the package's declared utilization telemetry. No advice or outcome predictions are emitted. Exact model-member redistribution attribution, empirical golf landmark/measurement accuracy, identity continuity, overlay/browser correspondence and CaddieSet equivalence remain separate gates. See pose-extraction.md and pose-assets.md.
