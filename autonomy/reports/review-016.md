# Independent review — execution 016

Scope: `003h2` additional-fold admission and five-update training-only smoke. Reviewed `fold_admission.py`, `fold_training.py`, CLI, focused tests and training guide against the unchanged registration, audit implementation and `HeadExperiment`. No implementation edits or Git operations were performed by the reviewer. No actual smoke or other real-data fit was launched by the reviewer.

## Findings

**P2 — fixed before launch: completed resume could miss checkpoint drift during its final dependency verification.** In the original completed-resume branch, checkpoint hashes were checked before `_recheck`, which can stream the full admitted input set, but were not checked afterward. Reproduction: complete the synthetic smoke; wrap `_recheck` so that its second call during `resume=True` first performs the normal dependency check and then appends bytes to the last saved checkpoint; forbid `HeadExperiment` construction. Resume returned completed successfully despite the changed checkpoint. An independent temporary regression failed with `DID NOT RAISE` (1 failed, 4.00s). The author added `_checkpoint_rows` after the final dependency boundary. The same independent regression now passes, and an equivalent repository regression is included. The completed ledger remains byte-identical and no model is constructed.

**P2 — fixed after coordinator entry preflight, before any actual fit: documented direct script could not import `backend`.** The initial tests imported the Python API under pytest's configured path and missed the executable entry. The coordinator's first documented command stopped with `ModuleNotFoundError: No module named 'backend'` at CLI line 6, before admission, job-directory creation or head construction. Independent reproduction with `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 .tools/ml-runtime/bin/python scripts/train_additional_fold.py --help` likewise exited 1. The author added an explicit repository-root import bootstrap and two subprocess regressions that clear `PYTHONPATH` and run from a temporary working directory. Independent verification: the documented relative `--help` entry now exits 0; an absolute-script invocation from `/private/tmp` with missing input files and `--mode full` exits 2 with the intended full-training refusal, and creates no run directory. The two subprocess regression cases independently passed (2 passed, 31 deselected in 11.97s). The failed coordinator command is retained in ignored `artifacts/additional-folds-016/frame-partial-stdout.txt`; the intended actual run directory did not exist after that preflight. This is a second P2, discovered by coordinator actual-entry preflight, not by the initial independent API-level review.

No unresolved material finding in the reviewed code. Coordinator-origin hardening also added checks of every durable checkpoint before/after successful ledger publication, precomputed expected ledger publication bytes, and rejection of duplicate/nonfinite resume JSON with typed progress fields. These are not attributed to the independent P2 finding above.

## Verified behavior and checks

- Production admission pins independently known completed audit run SHA `fedd0ab2d2e51cf3e22c6c4eb453cd0629e4fb6fc5fc29db03b677070109951f` and report SHA `282f540880ca111cae2723c971951f7e577d67858ce6762beea1301f49b9666e`; it does not trust a self-asserted report hash alone.
- Completed state, closed evidence gates, original frozen dependencies/runtime, full regenerated source-grouped partitions and exact selected cache hashes are checked. Returned cache coverage is exactly new train plus inner validation; outer cache arrays are not opened.
- Shared expensive-work lock is acquired before admission. Immutable runtime/config/seed/input/source/process/job identity is written before head construction. The original configs and historical pipeline remain frozen; heads start freshly seeded.
- Smoke is restricted to fold 2, exactly five cumulative optimizer batches per head. The epoch-boundary preflight excludes inner validation; summary history/selection stay empty, and no outer forward, score or accuracy metric is produced. Full mode fails closed and is deferred to `003h3`.
- Synthetic tests exercise both head types, two-update partial plus explicit continuation to five, continuous/resumed tensor equality, orphan-checkpoint recovery, interruption, input/runtime/checkpoint/publication drift, shared lock, rejected launch inputs, and completed resume without head construction or ledger mutation.
- Independent admission run: `PYTHONDONTWRITEBYTECODE=1 .tools/ml-runtime/bin/python -m pytest -q tests/test_fold_admission.py -p no:cacheprovider`: **41 passed in 9.54s**.
- Independent launcher plus separate reviewer regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests .tools/ml-runtime/bin/python -m pytest -q tests/test_fold_training.py /private/tmp/swingsync_review016_extra.py -p no:cacheprovider`: **32 passed in 19.37s** (31 repository tests plus one reviewer test). All fixtures were synthetic software checks, not GolfDB training or accuracy evidence.

- Independent source-only reproduction from `/private/tmp/swingsync-review016-xvi2o74d`, containing copied source/tests/configs but no Git metadata, data or checkpoints: the existing locked ML interpreter ran both new test modules with **72 passed in 43.03s**. This checks portable synthetic software fixtures; production actual-cache admission continues to require its historical source path/runtime identity.

Git publication/review/CI is explicitly a coordinator workflow prerequisite, not an invented programmatic remote-proof claim. The launcher records its local base revision and exact source bytes. The coordinator must verify push/review/CI before any actual fitting.

## Reviewed implementation identities

- `backend/swingsync/research/fold_admission.py`: `ac1fb8c95b04a04576ecfa63010212bca4a2ccbc1599b516b0f8f0916c1e264d`
- `backend/swingsync/research/fold_training.py`: `5f136dd6d10528afb52cfa64f29f577f2690286770e4d94fe2cf0ea4e688513f`
- `scripts/train_additional_fold.py`: `e75cd6a96656187573333f1ad940265084cd986f72784fc329fbd735b48238fa`
- `tests/test_fold_admission.py`: `5a65d4b1d1d2ba8ce6edb9b2d44e1d9c7cad02174781832cf502dd3548cceb34`
- `tests/test_fold_training.py`: `14b477919ac594a49bb5d380399e4bff12ca6cf69013026f87442c067fe38b54`
- `docs/additional-fold-training.md`: `72f3c1b63358878d72cb30c9734cc3964ec4a9821ba6338ebc02d4ef507ca148`

## Actual evidence

Both coordinator-run actual fold-2 training-only smokes genuinely completed after the corrected implementation `47667360fd4eda70f5478b5c9d2f4c47695daa87` was pushed and coordinator-recorded CI run `43774822790` passed all six jobs. The saved corrected launch requests bind that commit/CI and the prior reviewed-source report before fitting. The original `c851a94` command failed before any job/admission/head/update; its request, failure metadata and log remain separate and are not represented as partial training.

| Head | Completed job | Independently verified completed ledger SHA256 |
| --- | --- | --- |
| Frame | `7890727c123743c3b93a13c218f03c56` | `eb0728446686ad3db71960da17a131722f103cff0cb575589152795c70294657` |
| Temporal | `037d68b3764f49ea9fa8383653fb69a9` | `9b0bc9acc8c65a5a1898e9bebc415bdbed8b8c9c49e9267ec78e6d8b2ed41736` |

Independent actual-artifact verification was read-only. The reviewer loaded each of the ten saved checkpoint payloads using `torch.load(..., map_location='cpu', weights_only=True)` and checked every file SHA, model tensor finiteness, saved optimizer steps 1–5, epoch 0, cursor 4–20, RNG state presence, empty validation history and null selection/best-model state. Partial snapshot checkpoint rows 1–2 remain exactly the prefix of each completed ledger. Saved partial/final model tensor hashes match independently recomputed payload hashes; both differ from the recorded fresh-initialization hash. Each job has exactly two attempts, first partial and then completed, totaling five optimizer updates rather than five additional resume updates. Frame completion precedes temporal start, consistent with sequential shared-lock operation.

The reviewer independently regenerated fold 2 from the saved authenticated annotations: 836 training clips, 214 inner clips and 350 outer clips, with exact source-grouped membership. Every checkpoint's 1,050 cached-input fingerprint entries exactly match train plus inner and the known audit's cache hashes/keys; no outer ID is present. Independently recomputed train-only counts are `[836, 836, 836, 836, 836, 836, 836, 836, 230613]`, matching both ledgers, both checkpoint provenances and the known audit. Original config bytes, frozen pipeline plus explicit reencoded-PTS provenance, source hashes, runtime and independently known audit run/report metadata agree. The admission watched-map digest was reconstructed from those metadata bindings without rescanning cached arrays.

The first five batches use deterministic training clip IDs `330,1127,1336,1359,709,1213,1123,258,688,997,230,786,1074,212,311,25,899,350,999,296`. Independent annotation arithmetic gives 5,877 unpadded training frame rows from 20 source recordings per head. These are software-smoke exposure counts, not golfers, validation samples or inference-latency/accuracy measurements.

The coordinator's saved `frame-noop-witness.json` and `temporal-noop-witness.json` record byte-identical completed resumes for all six files per job (ledger plus five checkpoints), unchanged two-attempt counts and five updates. The reviewer independently rehashed every current file against both before/after maps and the known completed ledger hashes. The reviewer did not rerun completed resume, acquire the shared lock, call the launcher, construct a model, perform a forward/refit, open cache feature arrays, decode media or repeat the expensive full audit.

Both jobs retain `full_training_completed`, `inner_validation_evaluated`, `outer_test_evaluated`, `accuracy_metrics_computed`, `normal_model_available` and `validated_for_serving` as false, with empty advice. Completion establishes the registered actual training-only smoke and resumable software behavior. No 20-epoch training, selected model, validation/outer accuracy, uploaded-video generalization, coaching reliability or serving admission is established. Full training/evaluation remains explicitly deferred to `003h3`.

## Final claim check

Reviewed `additional-folds-smoke-016.json`, `validation-016.md`, final status/backlog and the architecture/data-contract/experiment/model-card/remaining-work documentation against the completed artifact review. Aggregate ledger/checkpoint/partial-snapshot/no-op hashes, per-attempt elapsed values, runtime, optimizer/epoch/selection fields and closed gates match the saved evidence for both heads. Counts and remaining `003h3` scope agree with the independently verified metadata. The source-only synthetic result is explicitly labeled as preceding the CLI fix, with the final two subprocess checks recorded separately. Full training, accuracy, independent-golfer generalization, uploaded-video/pose reliability, outcome integration and coaching remain unclaimed.

One wording clarification was corrected before acceptance: elapsed smoke time includes head construction, cached-feature reads and checkpoint verification, not creation of new caches. Historical parent-backlog no-launch evidence is now labeled as pre-015. No further material unsupported claim was found. This final claim check performed only metadata/hash reads; no new test suite, model construction, fit or forward was run. Final claim review accepted.
