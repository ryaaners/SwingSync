# Increment 002 validation — 2026-11-04

## Completed capability

GolfDB immutable MAT loader, source-grouped inner split manifests preserving official outer folds, explicit trim/source-index mapping, real decoded PTS timelines, and full-clip standard benchmark evaluator. Preparation CLI refuses overwrite and can validate a whole extracted archive with --all-clips. This is research preparation; it adds no unvalidated inference or coaching to the app.

## Actual data evidence

Audit fingerprints, archive provenance, split counts and artifact hashes: data-preparation-002.json. Actual annotations:1,400 clips,580 sources. Archive acquired from authors' public README link,698,989,423 bytes, exactly1,400 matching MP4 members. All clips were extracted using validated member names, decoded, and matched historical inclusive frame counts with finite strictly increasing PTS. Initial integrity pass took16.536seconds including local extraction/decoding; this is not model latency or a minimum-hardware benchmark. Reproduced independently through the committed preparation CLI with --all-clips.

Local artifact paths (ignored): data/golfdb/{golfDB.mat,videos_160.zip,videos_160/}; artifacts/golfdb/preparation-002-all/{fold-1.json,…,fold-4.json,summary.json}; artifacts/golfdb/inventory-002.json. No dataset, images, clips or weights redistributed. Hash ledger committed as aggregate evidence only. Archive data provision is not a grant of unrestricted third-party-media rights.

All four partitions cover every clip once and keep sources disjoint. With seed20261104 and20% inner source fraction, train/validation/test clip counts:840/210/350,836/214/350,839/211/350,853/197/350. Source-level fraction need not equal clip-level fraction.

Annotation677 preserves reversed Impact/follow-through indices. Standard evaluator matches official chronological occupied-target slots and tolerance2, rather than silently substituting semantic Impact (tolerance3). Raw class annotations and anomaly flags remain in results. No order-constrained improvement is mixed into standard PCE.

## Software checks and review

Full pytest suite:56 passed, one pre-existing Starlette/AnyIO deprecation warning. Ruff backend/tests/scripts passed. Locked offline uv sync passed with SciPy1.15.3. Frontend clean install (npm ci --ignore-scripts) and strict TypeScript/production build passed; npm audit reported zero known vulnerabilities. Original first-increment CI run41967665483 passed. New-checkpoint CI will be reported after push; do not infer remote success from local checks.

Independent fresh review: review-002.md. Complex timestamp conversion issue fixed and covered by rejection tests before float coercion. Reviewer independently rechecked evaluator, loader, timelines, CLI and actual sample media. CLI serialization/no-overwrite and malformed inventory regressions included.

## Remaining gates

No training launched, no checkpoint/model accuracy, no pose/coaching/outcome predictions, no model inference timing. Real event contact sheets0/677 were inspected locally for coarse visual sanity only. Exact source-video image equivalence, original YouTube PTS and qualified annotation review are not established by the re-encoded archive. Browser seeking with leading audio/container offset remains unverified. These limits are preserved in status/backlog/model card and data contracts. Research using identified preprocessed clips can proceed with its actual re-encoded PTS disclosed; original-source precision claims remain gated.
