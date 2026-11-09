# Independent review — execution 009

Reviewed 2026-11-09 in fresh reviewer context. Scope: increment 004d's isolated pose environment, acquired model identity, offline launcher, timestamp-preserving CPU extraction, synthetic checks, documentation, and coordinator-produced actual extraction artifacts. No implementation edits, Git operations, training, or duplicate native inference were performed by this reviewer.

## Disposition

No unresolved material finding in the reviewed implementation. Three P2 findings raised during development were corrected and independently rechecked:

1. **Self-consistent forged model manifests could replace the reviewed model.** Initially, `validate_model` trusted caller-supplied bundle/member hashes and any truthy license claim when the URL matched. Reproduction: create a ZIP with the expected two member names, set its own hashes/size in a manifest with the official URL and an arbitrary license dictionary; admission succeeded. `backend/swingsync/pose_runtime/extraction.py:63` now checks separately pinned bundle size/hash, member identities and complete code/model/privacy evidence from the acquisition registry. Regression tests reject jointly changed model+manifest bytes and invented license claims. The real acquired asset passes the stricter validator.
2. **A final ledger publication failure could leave a completed result beside a failed job.** Reproduction: permit result publication, then inject `OSError` only when publishing `run.json` with state `completed`. The exception path now removes `result.json` before recording failure (`extraction.py:306`). Independent injected-failure execution leaves a failed ledger and no result. The committed regression test passes.
3. **A multi-pose frame left other frames available as an apparently unambiguous spatial sequence.** Reproduction: a completed artifact containing one single-pose row and one multiple-pose row could previously be converted with `as_pose_sequence`. Extraction now records sequence identity ambiguity, and conversion rejects any explicit ambiguity or multi-pose row (`contracts.py:114`), preserving raw observations for inspection. The regression test confirms whole-sequence spatial suppression.

The public presence-floor helper also now rejects invalid configuration values, including NaN. Camera/mirror/handedness/single-golfer metadata remains an explicit operator declaration rather than empirical validation.

## Independent software checks

- `.venv/bin/python -m pytest tests/test_pose_runtime.py tests/test_pose_launcher.py /tmp/review_pose009.py -q`: **61 passed** (56 repository tests plus five independent temporary review tests). The temporary tests inject source/model/runtime identity drift during synthetic inference and completed-ledger publication failure; all reject completed output. These tests use fake native objects and demonstrate software behavior only.
- Targeted Ruff over the pose adapter, both test modules and all three new scripts: **all checks passed**.
- Independent seeded `Fraction` oracle: **1,000 timestamp sequences** plus a mixed-time-base/nonzero-origin boundary matched exact half-up relative integer milliseconds and preserved actual source seconds. Rejected cases matched rounded-clock collisions. No FPS reconstruction or jitter was used.
- Independent synthetic acquisition probe, with in-memory fake download and separately patched fixture pins: initial exclusive publication, byte-identical reuse without a second fetch, corrupt-existing-file refusal, and oversized invalid-download cleanup all passed. No network request or model inference occurred.
- Reviewed geometry admission for unknown/nonzero rotation, display matrices, missing/non-square pixel aspect, changed dimensions, PTS/time-base drift and decode budgets. Native detector construction follows the complete clock preflight, and missing/multiple/malformed observations retain rows without fabricated points.
- Reviewed the launcher: fixed isolated interpreter, fixed extractor path, argument-array subprocess invocation, macOS `sandbox-exec` profile `(version 1)(allow default)(deny network*)`, and unsupported-platform refusal. The extractor requires OS-denied numeric IPv4/IPv6 socket probes before MediaPipe import; refusal, timeout and unavailable IPv6 are not accepted as denial. These probes are guards, not proof that arbitrary firewall rules deny egress. No packet-capture or telemetry-payload claim is made.

## Environment and asset evidence

Independently matched root and pose lock hashes to `pose-environment-009.json`; the installed package inventory's exact versions occur in the pose lock. The root lock remains `8ce1d6674a07bf122541f555b90ad81532029b7ce343af54a78cd057fc2b26fc`. The isolated lock is `edc1d9c8c3e6e3a794bbb0e582e25f292ce5bf3b640f11026c3cce6df9135919`.

Rehashed all seven upstream evidence files, all **14 current** installed license/notice inventory entries (including file sizes), the MediaPipe native library, acquired bundle, both members, and full manifest. They agree with the report. The reviewed manifest distinguishes code notices, model-card family evidence, absent exact-member metadata licenses and disabled redistribution. This review does not resolve exact-member redistribution rights.

`pose_runtime/` has its own non-workspace Python 3.11 project/complete lock and uses the explicit separate `.tools/pose-runtime` interpreter. Review did not reinstall or alter either environment. The environment report records macOS ARM import/native conversion; Linux/aarch64/Windows extraction is not established by that evidence.

## Actual completed extraction audit

The coordinator alone ran native CPU extraction under the offline launcher. Reviewer independently inspected ignored `artifacts/pose/smoke-009-train-0/{run,result}.json` and decoded the source using PyAV, without rerunning pose inference:

- Both durable artifacts say completed; the ledger binds the actual result SHA.
- **138/138 frames** match independently decoded native integer PTS, rational time bases, float64 seconds, raster dimensions, and a separately recomputed rational VIDEO clock.
- Actual source time spans `0.0` through `4.570018013209687` seconds; separate API clock spans `0` through `4570` ms.
- **134 single-pose / 4 no-pose** rows. All missing rows contain null measurement points; raw-score-to-landmark/presence-filter mappings match for every single-pose row. Counts agree across ledger and result. Immutable `PoseSequence` conversion succeeds for this non-multiple artifact.
- Current source SHA, acquired model+manifest, pose lock, and all recorded adapter/launcher/registry source hashes match. Advice is empty and coaching validation is false.
- Source SHA256: `c4f17b019859c62b3745fd7f48c2a6e79892ad90be44ebf86cd759e72fa084d0`.
- Completed run SHA256: `772ab832647ccede39291b95b0179e54f5cf84e67c4af86980e1c1bf5691fb65`.
- Completed result SHA256: `e972b761bc519b7544f85fc1a19224d6e9c8d39d3c2caa6d246de1ec9c432ba1`.

The input is an official preprocessed TRAIN clip, with **reencoded** timestamps. Saved provenance labels hand/mirror/single-golfer declarations as operator assumptions after frame inspection. The native `NORM_RECT`/`IMAGE_DIMENSIONS` warning reported by the coordinator remains a limitation; no golf coordinate-accuracy inference follows from successful execution. No held-out model comparison was repeated or tuned.

## Remaining gates and review limits

This is actual extraction/software correspondence evidence, not empirical identity continuity, landmark accuracy, calibrated score reliability, coaching efficacy, browser-overlay correspondence, or uploaded-video generalization. Single-pose returns do not prove one person or stable identity. Known mirror/unknown view/unsupported transforms remain refused; externally suspected identity switches require suppressing spatial use. Advice, event serving, outcomes and model redistribution remain disabled.

Reviewer process listing was sandbox-denied; coordinator reported independent process/shared-lock inspection before its sole actual run. No independent packet capture or native inference replay was performed. Full repository integration checks, final status/backlog, Git checkpoint and remote/CI verification remain coordinator-owned and are not claimed by this report.

## Final integration documentation follow-up

Reviewed the final README, architecture/data-contracts/experiments/model-card updates and `pose-smoke-009.json`. No additional material finding. The documents consistently separate ordinary app unavailability from the completed offline research extractor, disclose pretrained upstream provenance, one TRAIN/reencoded clip, operator assumptions, unvalidated accuracy, timing exclusions/native thread limits, upstream projection warnings, macOS isolation limits and unresolved model-member redistribution attribution. The configured Linux CI job is import-only; no Linux extraction result is implied.

Independently matched the final smoke aggregate to the actual completed ledger/result: artifact and verification-witness SHA256 values, complete configuration, status counts, timing, identity/coaching flags, advice count, source/view/timestamp/dimension metadata, native versions and adapter provenance all agree. The aggregate contains no raw coordinate rows or media. The verification witness was inspected and hashed without executing its report-writing side effect. No extra inference or production change was made during this follow-up.
