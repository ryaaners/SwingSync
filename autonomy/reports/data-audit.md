# Initial data audit — 2026-11-04

Scope completed: read official GolfDB reference loader, split generator, preprocessing, training, model and evaluation source; downloaded and inspected official MATLAB annotations; downloaded current CaddieSet CSV and paper; checked complete CaddieSet GitHub tree and root license. No training launched. No source code or dataset artifacts vendored.

## Evidence

Public read-only requests through curl required network escalation; ordinary sandbox DNS failed. Downloads were stored in `/tmp/swingsync-audit`, which is temporary and is not a reproducible artifact store. Persistent fingerprints and facts are in `docs/data-contracts.md`.

GolfDB revision `a63b4ce8c0900d09abbe92519223efe16810f5f4`; 1,400 rows; 580 source IDs; four folds of 350 rows; no source IDs shared across folds; every events array length ten. Verified the trim boundary/8-event distinction and source-index offset from preprocessing source. The exact official tolerance is rounded (Impact−Address)/30 with a minimum of one frame, rather than duration or FPS scaling. Standard prediction is independent class argmax. Videos still required to validate annotation-to-decoded-image alignment.

CaddieSet revision `3c73d9d40580bb8a5a10711ad1fa10735a205ffe`; CSV SHA256 `646459b081863e244d75efe6f09fad4d34750d2fed7fde44db654e7720a9a306`; 1,757 rows × 80 columns; 924 FACEON / 833 DTL; eight golfers; 69 event features; 58,917 empty cells; 11 `#NAME?` cells; six exact duplicate rows; DirectionAngle fully populated. Found 803 groups with identical golfer/club/ball outcomes across both views, consistent with repeated shot views. No authoritative shot identifier exists. All-golfer grouping is mandatory.

CaddieSet paper downloaded from the CVF URL in the repository; SHA256 `2cc1bc77850957e21561736e54b97f30e9a97d419c3ee115643c5d8e55adc313`. Text inspected using temporary pypdf 6.1.1. Paper describes 924 FACEON records/40 features, automatic fine-tuned SwingNet events, Faster R-CNN detection, HRNet 17-point poses and bounding-box-width scaling. It defines DirectionAngle as initial departure in degrees, negative left / positive right. Current CSV extends this release with DTL and more feature columns. Thus neither paper feature count nor paper preprocessing can silently define the full current CSV.

## Blockers recorded once

1. Event training: permitted real video clip inventory and frame-alignment validation absent. Next action: acquire referenced preprocessed clips under applicable terms; fingerprint and inspect a real clip before training. Public annotation access itself works.
2. Outcome integration: complete upstream tree has CSV/README/LICENSE only, no videos/extractor/weights. Need authorized paired video/extractor data, full numerical feature definitions, camera/handedness/event conventions and paired equivalence validation, then successful grouped baseline comparison. Research-only CSV experiments can proceed independently after pipeline implementation.
3. Redistribution: GolfDB code explicitly CC BY-NC 4.0; do not infer separate rights for third-party media or weights. CaddieSet MIT notice covers repository provision without proving rights to unreleased media. Original-project license remains an owner choice.

## Suggested acceptance for next data increment

Implement a non-mutating MATLAB annotation loader and saved source-grouped inner split manifest. Test first-row example, all 1,400 annotation shapes, disjointness and official PCE rounding boundaries. Decode one permitted real clip to verify the historical inclusive one-based source counter versus zero-based decoder mapping. No benchmark can be claimed from these software checks.
