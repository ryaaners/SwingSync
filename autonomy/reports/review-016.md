# Independent review — execution 016

Scope: `003h2` additional-fold admission and five-update training-only smoke. Reviewed `fold_admission.py`, `fold_training.py`, CLI, focused tests and training guide against the unchanged registration, audit implementation and `HeadExperiment`. No implementation edits or Git operations were performed by the reviewer. No actual smoke or other real-data fit was launched by the reviewer.

## Findings

**P2 — fixed before launch: completed resume could miss checkpoint drift during its final dependency verification.** In the original completed-resume branch, checkpoint hashes were checked before `_recheck`, which can stream the full admitted input set, but were not checked afterward. Reproduction: complete the synthetic smoke; wrap `_recheck` so that its second call during `resume=True` first performs the normal dependency check and then appends bytes to the last saved checkpoint; forbid `HeadExperiment` construction. Resume returned completed successfully despite the changed checkpoint. An independent temporary regression failed with `DID NOT RAISE` (1 failed, 4.00s). The author added `_checkpoint_rows` after the final dependency boundary. The same independent regression now passes, and an equivalent repository regression is included. The completed ledger remains byte-identical and no model is constructed.

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
- `scripts/train_additional_fold.py`: `3dd83f6e9a0285449a570c403e24d183ee93b6178b16aaaab30f89771b0c9f58`
- `tests/test_fold_admission.py`: `5a65d4b1d1d2ba8ce6edb9b2d44e1d9c7cad02174781832cf502dd3548cceb34`
- `tests/test_fold_training.py`: `0a64fdc9b6191e9b27bed0382b7c2d7b426ed5d24d7430b2b5498e6ce780f866`
- `docs/additional-fold-training.md`: `166cee817892a78ad0da2565f65de5ecadc77d2eeb242788d24574c9d88114a9`

## Actual evidence

Pending coordinator notification. At this review stage, no actual new-fold fitting had launched. Synthetic tensor results establish software behavior only. No full training, validation, outer evaluation, model quality, uploaded-video accuracy, coaching reliability or serving admission is established.
