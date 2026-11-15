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

Pending coordinator notification. Registration/implementation checkpoint `c851a94` and its six successful CI jobs preceded the failed direct-entry command above. That command stopped before any job, admission, head or optimizer update; it is not a partial training result. The entry fix must be pushed and checked before a subsequent actual fit. At this review stage, no actual new-fold fitting had launched. Synthetic tensor results establish software behavior only. No full training, validation, outer evaluation, model quality, uploaded-video accuracy, coaching reliability or serving admission is established.
