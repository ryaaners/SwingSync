# Independent review — execution015

Reviewed 2026-11-12 as the reviewer role. Software acceptance and the coordinator's completed1400-cache audit evidence passed independent review. No new training, backbone/head forward, video decode, prediction, outer PCE or human reference label was produced by this review. No Git operation or implementation edit was performed. This reviewer owns this report only.

## Scope and result

Reviewed `backend/swingsync/research/fold_cache_audit.py`, `scripts/audit_fold_caches.py`, `backend/swingsync/research/fold_protocol.py`, `experiments/additional-folds-v1.json`, both new test modules and both new documentation pages against the unchanged feature/cache, GolfDB split, training and official-evaluator contracts. Final protocol SHA256 is `b9c7e29b2afa63e6733d0a26c5e335f7ad31463143b97300a2ea3e3d91107a53`.

No unresolved material findings remain in the reviewed software or completed audit evidence. At the registration checkpoint, software acceptance was complete and the actual scan was explicitly pending; the post-checkpoint evidence below now establishes its completion. Acceptance covers the read-only audit, prospective registration and saved compatibility evidence. It does not admit a new training loader, launch fitting, or validate uploaded-video/coaching accuracy.

## Findings resolved before acceptance

1. **P2 — incomplete dependency drift coverage.** The initial audit validated original head configs, the held-out registration and additional frozen sources during protocol admission, but omitted some from later publication checks. Changing a protocol-declared config after admission could therefore leave an apparently completed audit. Reproduce with `test_protocol_declared_config_drift_after_admission_refused`: the synthetic clip-check hook changes a registered temporary config after admission. The final implementation collects every declared path/SHA binding, frozen-source binding and dependency lock into its watched inputs, and checks them before report publication, after report publication and after the completed ledger. The regression now rejects the change and removes the success report.
2. **P2 — published bytes needed a precomputed expected digest.** Worker self-review found that calculating the report digest only after writing could authenticate a report altered during publication. `test_modified_published_output_cannot_be_bound_as_valid[audit.json]` and `[run.json]` reproduce this with write hooks. The final implementation computes expected serialized bytes before publishing, verifies both report and final-ledger bytes, marks failure and removes the success report on drift. Both cases pass independently.
3. **P2 — synthetic fixtures depended on the original checkout/runtime.** Coordinator review found the fixture initially called strict production runtime admission using historical Mac versions and marshaled source-path identity. This could fail every fixture in a fresh checkout or Linux CPU build before exercising the intended behavior. The final fixture writes a clearly synthetic registration from its current runtime/code identity. Strict production checks remain unchanged; separate tests reject a changed Torch build suffix and changed code filename. A fresh source-only temporary copy independently passes all88 focused cases.

## Independent verification

- Read `AGENTS.md`, reviewer role, current status and backlog, plus historical completed-cache evidence. The read-only descriptor probe found the shared expensive-work lock available during initial inspection. Direct `ps` inspection was sandbox-denied; process absence is the coordinator's responsibility and was not independently claimed here.
- Independently validated actual audited MAT bytes and all three new saved manifests with `validate_protocol(dataset=load_annotations(...), manifests_dir=...)`; exact source-grouped assignments, counts and protocol bytes passed without opening caches or videos.
- Ran `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_fold_cache_audit.py tests/test_fold_protocol.py -q -p no:cacheprovider --basetemp=/private/tmp/swingsync-review015-folds-r2`: **88 passed in2.86s**. These are synthetic software checks, not model results.
- Copied only backend source, experiment registrations, dependency lock, audit CLI and selected tests to `/private/tmp/swingsync-review015-source-5y2j1akv`, excluding bytecode, datasets, weights and real caches. Ran the same focused suite using the existing Python environment from that new source directory: **88 passed in3.07s**. This verifies source-path independence of software fixtures; it does not reproduce strict historical real-cache runtime admission on Linux.
- Independent Ruff check of all five new code/test files passed. Reviewed coordinator check logs: locked base653passed/14skipped, fullML778passed/1skipped; coordinator reports frontend9tests/build and complete Ruff passed. Existing AnyIO deprecation warning remains.

The reviewed cache path authenticates both independently known historical catalog SHA values, exact old membership/disjoint coverage, complete MAT identities, cached array hashes/schema, inclusive historical indices, finite float32[frames,1280] features and float64 increasing PTS. Fresh source-video hashes bind the existing cached timestamps to previously verified source bytes; no fresh decode or original-source timing equivalence is claimed. New per-fold train-only class counts come solely from each regenerated train partition. Audit success leaves training/evaluation/serving gates false.

The registration preserves original20-epoch configs and seeds, fresh per-fold heads, no warm start, inner-only earliest-tie selection, fixed reference64-reset and separately reported full-clip contexts, and unchanged official chronological-target PCE including anomaly677. It truthfully discloses previously observed fold1 results and historical use of the new outer clips in fold1 train/inner partitions. Neither source-disjoint folds nor frozen-feature reuse proves unseen-golfer or arbitrary-upload reliability. Future launcher/smoke, fitting/evaluation and empirical coaching gates remain separate unfinished work.

## Actual evidence addendum

Coordinator reports registration commit `2f1fbebf6c5eb776182383af63a28ec98b161afe` was pushed and its remote SHA verified before the single actual scan. Remote Git/CI state is coordinator-supplied evidence; this reviewer performed no Git operation.

Completed ignored audit: `artifacts/additional-folds-015/audit-v1/`, job `a0bd16f8d12a458594ab23acde8333b4`, recorded PID1301, created06:50:28Z and completed06:50:46Z on2026-11-12, measured18.360005seconds. Run SHA256 `fedd0ab2d2e51cf3e22c6c4eb453cd0629e4fb6fc5fc29db03b677070109951f`; report SHA256 `282f540880ca111cae2723c971951f7e577d67858ce6762beea1301f49b9666e`. Both canonical serialized byte digests independently match their files, and the run binds the report digest.

Independent post-completion checks passed:

- Authenticated both known historical catalog SHA values, actual MAT and protocol bytes; regenerated all three new manifests. Cross-checked all1400 report mappings against immutable historical clip/source/split/cache/key/hash/video-hash/old-partition identities and MAT-derived inclusive frame counts. Coverage is exact with no overlap; total394177frames.
- Recomputed every fold's complete train/validation/test clip/source membership and train-only class counts from actual MAT records. Counts below match the report exactly; no validation/test labels contribute to training imbalance counts.
- Rehashed all23 watched input paths and all12 implementation paths from the completed run; every expected SHA still matches. Runtime-only introspection independently verified exact historical pipeline versions and code identity; recorded Python3.11.14,NumPy2.2.5,PyAV16.0.1 agree. No model was constructed or executed.
- Reopened only representative NPZs0,10and677 with the strict existing semantic/array-hash loader, rehashed those three source-video files without decoding, and matched file hashes/identities/native-PTS digests/shapes/dtypes. Clip677 retains `[92,114,131,146,158,168,167,242]`; no anomaly repair occurred. This was not a duplicate full cache scan.
- Rechecked all false gates in both run and report: no training, inference, accuracy metrics, training/evaluation admission, normal availability, serving or advice; fresh decode remains false. Native PTS remain authenticated cached reencoded timestamps, not original-source or physical-time validation.

| Fold | Train / validation / test frames | Train event counts0–7 each | Train background count |
| --- | --- | --- | --- |
| 2 | 237301 / 61014 / 95862 | 836 | 230613 |
| 3 | 237638 / 61200 / 95339 | 839 | 230926 |
| 4 | 236651 / 53506 / 104020 | 853 | 229827 |

No material finding remains in the completed raw audit evidence. A future reviewed launcher/train-only smoke and actual fixed fold training/evaluation remain unfinished; these read-only compatibility results do not replace them. The coordinator's final aggregate, validation report and status/documentation claims were separately checked as recorded below.


## Final package claims acceptance

Reviewed `autonomy/reports/additional-folds-audit-015.json`, `validation-015.md`, final `CURRENT_STATUS.md`, the003hbacklog family, appended architecture/data-contracts/experiments/model-card material, remaining-work and audit instructions. Aggregate raw paths/SHA values, all three manifest SHA values, partition summaries, training-only count vectors and23input/12implementation counts match the independently checked raw evidence. Backlog parent003h remains in progress,003h1 is complete, and003h2/003h3 remain pending. The status and remaining-work text identify reviewed launcher admission and a registered training-only smoke as the next increment.

The final claims retain historical fold1 exposure, source-recording rather than unseen-golfer scope, strict historical real-audit runtime, separate synthetic software fixtures, cached reencoded timestamps without fresh decoding/original-source equivalence, no new fit/forward/head load/PCE, false training/evaluation/normal-model/serving gates and empty advice. No four-fold model accuracy or automatic training-token claim is made. Registration remote SHA and six successful CI jobs are coordinator-verified evidence; no Git was performed by this reviewer. One minor status wording suggestion was sent to replace ambiguous “original source bytes” with “recorded preprocessed-video bytes”; the source-original gate is consistently closed in the underlying evidence.

Final disposition: accepted with no unresolved material findings. All reviewer writes are complete; no further report or implementation changes are planned before the coordinator's final checkpoint.
