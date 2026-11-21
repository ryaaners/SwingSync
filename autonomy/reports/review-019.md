# Independent review 019 — additional-fold evaluation software

Reviewed 2026-11-20 in fresh reviewer context, before real additional outer evaluation or timing. Read AGENTS.md, autonomy/roles/reviewer.md, CURRENT_STATUS.md and backlog.json. Scope: additional_study.py, additional_evaluation.py, additional_latency.py, both new CLIs, synthetic tests and protocol/docs. Reviewer changed no implementation, performed no Git operations, and ran no real model construction, checkpoint tensor load, fitting, outer forward or latency measurement.

## Decision

No unresolved material correctness finding remains in the reviewed source snapshot. The metadata admission, outer runner and registered TRAIN-only latency runner are acceptable for coordinator integration, publication and CI. This is software review only; remote publication/CI and actual run completion remain coordinator gates. Source/runtime/input identities must remain frozen before and throughout real runs. No additional-fold accuracy, hardware latency, serving, advice or upload-domain reliability result is established by this review.

## Findings resolved before acceptance

1. **P2 — interrupted outer forwards could silently repeat.** Originally `pending` was first persisted after all four head/context forwards and serialization. Reproduction: use the synthetic harness, make `predict_probabilities` return once then raise, simulate the dead predecessor and resume. The original ledger had no clip or pending marker, allowing the first variant to run again. Resolution: persist exact `in_flight` input/fingerprint before constructing/forwarding a selected head; refuse resume when a started clip lacks known durable probability bytes. Known authenticated NPZ orphans are adopted without forwards. Regression: `test_crash_after_first_forward_never_repeats_unknown_predictions`, `test_known_orphan_recovered_without_repeating_forward`.

2. **P2 — a report publication crash stranded completed probability evidence.** Originally `report.json` appeared before a durable completed ledger, so interruption in that interval made it an unknown orphan. Reproduction: make `_publish` raise immediately before or after publishing the final report, then resume with the synthetic dead predecessor. Resolution: persist `pending_report` path/SHA derived from complete independently replayed probabilities before publication; authenticate and reuse an exact existing report, or reconstruct an absent deterministic report without loading/forwarding models. Regression: both cases of `test_deterministic_report_orphan_recovers_without_forward`; existing report bytes/stat identity remain unchanged.

3. **P2 — initial stat pinning invalidated every fresh publication.** In the intermediate optimization, `_publish` captured inode ctime while the temporary hard-link alias still existed; deleting that alias changed the same published inode ctime. Reproduction: invoke the existing synthetic harness; initial ledger `save()` failed its next stat guard before model construction. Resolution: remove temporary alias before capturing published identity. Final fresh/bounded/resumed/completed tests pass; stat tests also reject restored bytes/mtime, inode replacement and symlinks.

The owner also added conditional final-ledger rollback: post-publication input drift retracts only the unchanged, owned successor to its prior pending-report ledger; an outside ledger edit is preserved. Both synthetic branches pass. Latency report cleanup likewise records the published inode at link time and preserves an external replacement; externally deleted ledger predecessors produce a separate conflict witness.

## Independent validation

- `.tools/ml-runtime/bin/python -m pytest tests/test_additional_study.py tests/test_additional_evaluation.py tests/test_additional_latency.py -q -p no:cacheprovider`: **159 passed in 6.85 s** (84 study, 29 evaluation, 46 latency).
- Previous stable outer/gate snapshot plus unchanged `tests/test_evaluation.py tests/test_heldout.py tests/test_latency.py tests/test_fold_protocol.py`: **238 passed in 12.52 s**. The subsequent outer changes added two publication tests and were included in the final 159-test run above.
- Direct root `.venv/bin/ruff check --no-cache` on all three new research modules, both CLIs and all three new test files: **All checks passed**.
- Process inspection showed no active SwingSync training/evaluation/benchmark worker. Only the inspection command itself matched.
- Synthetic tests intentionally mock production study admission and model calls, except the explicit synthetic selected-versus-last checkpoint loader test. They are software evidence, never accuracy or hardware-performance evidence.

## Gate and protocol checks

The production admission rejects caller-substituted receipt SHA before downstream admission. It binds independently known 018 receipt SHA `7c37a7d1b04e618dda41673f2a0b5e252b23f1822e6d3fc394590cbc0b06b407`, all six independently known completed ledger hashes, 120 immutable checkpoint byte bindings, fixed original configurations and runtime, train-only class weights, earliest maximum inner-PCE selection, both 016 smokes, the known 015 audit, and regenerated source-grouped manifests. Admission loads no tensor and constructs no model. Evaluation loads the final checkpoint's `best_model` through the unchanged selected-head loader, with no last-model fallback.

The outer preflight covers exactly 350 official outer clips per fold 2/3/4 and both heads. The unchanged evaluator retains NumPy default argsort-last peaks, chronological occupied target slots while exposing raw class-order anomalies including the 677-style case, rounded official tolerance with inclusive equality, separate reference-64-reset/full-clip contexts, and errors from authenticated float64 actual cached PTS. No chronological repair, selection on outer results, calibration, fitting or product integration appears in the new path. Reports disclose historical fold1/data reuse and source-recording rather than verified unseen-golfer scope.

Stat guards pin dev/inode/size/mtime/ctime/mode after stable full-byte authentication; full SHA checks repeat at invocation completion boundaries. Per-forward checks reject writes including restored bytes/mtime. This relies on the trusted local OS filesystem, as documented in code. Shared flock, predecessor comparisons, exact prefix replay, authenticated probability arrays, known-orphan adoption, no-forward byte-identical completed resume, and outside-edit preservation are covered synthetically.

Latency admits exactly registered TRAIN clips fold2=0/1, fold3=6/7, fold4=0/1, CPU, batch16, 16 warmup frames per fold and one measured repetition. Fresh decoded PTS must equal cached native PTS. Exact local ImageNet weight bytes are required before construction, and the unchanged loader uses weights=None plus those verified bytes; missing/wrong weights fail without downloads. All six selected heads, frozen-state checks, one-shot refusal, interruption/unavailable evidence and shared-lock exclusion are covered. Stage timing excludes provenance verification; composed pipeline timing shares one extraction pass and is explicitly scoped. No accuracy calculation or minimum-hardware claim occurs there.

## Real metadata witness checked without model work

Read coordinator-produced `artifacts/additional-folds-019/admission-check.json`, SHA `7a11dbd8f4b4d046d12b5bbdec4bad3934d36ba5a7e425f61f9dfae9da9e8dcf`. Independently rehashed the actual fixed receipt and all six run.json ledgers against compiled known constants and receipt values. Independently recomputed earliest maximum inner-PCE epochs from actual saved histories: fold2 frame/temporal20/18, fold3 20/18, fold4 19/19. Selected and final checkpoint SHA fields agree with the corresponding ledger epoch rows. Artifact flags for construction/tensor loading/forward/accuracy/normal model are all false, with evaluation_state_key=best_model.

The metadata artifact reports 1,583 watched files and 18.961426165999 seconds for admission plus recheck. Those counts/time are coordinator gate evidence; this reviewer did not duplicate the 120 raw checkpoint rehash or reconstruct the full production admission independently. The reviewer read JSON metadata only. No real outer probability or timing artifact existed within this review's acceptance scope.

## Reviewed source SHA256

- additional_study.py: `ed39268bff22d80b25a8b738ffcbfa75e04167bfa06efe7aca22e07538bda93c`
- additional_evaluation.py: `006d74d1c100809970df8832197fb881c44d446f617bd9a8a357f7e4724f55f8`
- additional_latency.py: `317e85aaa5d555a828b4d31b265a9b28e52505f4f62de70285f4c3cff2649d4b`
- evaluate_additional_folds.py: `6174c06fb6efd08056d37e618668bc5044d59a165fa3851c3b20ece9b21d0dc4`
- benchmark_additional_folds.py: `ad0997ef3e27392e142226deb8b0aea7c9f3a79487d51a9666b6445e06e89d47`

Normal serving/model availability, outcomes/pose availability and advice gates remain closed. Any later real result requires independent saved-probability replay and provenance/completion verification; this report does not pre-approve unobserved results.
