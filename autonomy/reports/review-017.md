# Independent review — execution017

Pre-launch review completed 2026-11-16. No open material findings in the reviewed full-training increment. This review admits the software for coordinator checks/publication/CI; it does not assert actual full training, outer evaluation or completion of backlog003h3.

## Scope and isolation

Read AGENTS.md, autonomy/CURRENT_STATUS.md, autonomy/backlog.json and autonomy/roles/reviewer.md. Reviewed the new full-fold admission, training launcher, CLI, focused tests and docs/full-fold-training.md against the frozen registration and historical HeadExperiment driver. No implementation edits, Git commands, actual-data fitting/forwards, video decode, checkpoint acquisition or expensive cache audit were performed. Synthetic arrays establish software behavior only. Reviewer owns only this report.

The sandbox denied process listing; the coordinator separately supplied its protected process inspection showing no relevant Python worker. Independently acquired and released the existing shared lock through a read-only descriptor: free. No actual full job had been launched at this review boundary.

## Resolved actionable findings

1. **P2 — Reject overflowing JSON numbers before resume.** The initial `_read_state` rejected literal NaN/Infinity but `json.loads` converted `1e999` to float infinity. Reproduce with `_read_state(b'{"attempts":[{"elapsed_seconds":1e999}]}')`; the original implementation returned nonfinite attempt metadata. A completed no-op checked only that attempts was a list, so it could accept this invalid numeric state. `full_fold_training.py` now supplies a finite `parse_float`; independent reproduction rejects overflow, NaN and duplicate keys. Focused regression: `test_strict_ledger_rejects_nonfinite_overflow_duplicates_and_nonobject`.

2. **P2 — Preserve a changed predecessor ledger.** The initial `_save_state` verified its output but silently replaced unrecognized predecessor bytes. Reproduce by changing run.json externally between advance and publication; the original path overwrote that edit. The new guard compares the last owned ledger digest before publication, preserves external bytes, writes a closed-gate conflict marker, and refuses automatic continuation while that marker exists. Independent direct reproduction confirmed rejection and byte-identical preservation. Focused regression: `test_predecessor_ledger_drift_is_preserved_and_blocks_continuation`.

The author also added upfront completed-resume checks for saved fresh initialization, train counts and class weights; these bind saved initialization to the known smoke INITIAL tensor hash without constructing a head. Original frozen modules remain unchanged.

## Evidence and verification

- Independently hashed both actual016 ledgers: frame `eb0728446686ad3db71960da17a131722f103cff0cb575589152795c70294657`; temporal `9b0bc9acc8c65a5a1898e9bebc415bdbed8b8c9c49e9267ec78e6d8b2ed41736`. Both are completed five-update smoke witnesses.
- Independently hashed all ten immutable smoke checkpoint files against additional-folds-smoke-016.json. Loaded all ten saved payloads with CPU `weights_only=True`, with no model construction/forward: exact original configs and train-only counts; epoch0, cursor4/8/12/16/20, steps1–5, empty history, null selection and no best_model.
- All ten `frozen_source_sha256` registration bindings and the root uv.lock still match experiments/additional-folds-v1.json. The original 20-epoch configs and HeadExperiment's strict-greater PCE/earliest-tie selection remain frozen.
- Independent existing protocol/admission/smoke suite: **135 passed in16.76s**. Independent new admission suite: **39 passed in6.52s**. Independent initial new launcher suite: **34 passed in9.07s**. Four final provenance regressions were added and included in the final source-only run below.
- Copied only backend/tests/scripts/experiments/pyproject.toml/uv.lock into a fresh temporary source directory; invoked the original isolated ML interpreter with PYTHONPATH unset and no repository installation/artifacts. Final current new suites: **77 passed in16.42s** (including the four final completed-provenance/fresh-initialization regressions). Their CLI subprocess cases exercised `--help` and invalid-fold refusal from another working directory; the direct `--help` invocation also passed independently.
- Additional independent synthetic test for BOTH frame and temporal with the ORIGINAL hidden64/20-epoch configs: pause after17 updates, explicitly resume to140 fixture updates, retain all20 epoch boundaries, force exact inner-score ties, and compare final best_model against both captured first-epoch tensors and the saved first-epoch checkpoint model. Both selected epoch1 exactly; selected weights differ from final model weights. Completed resume constructed no head and left ledger bytes identical. This uses small synthetic cached arrays and substituted admission, never GolfDB training or accuracy evidence.

Reproduce focused suites from the repository with:

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/ml-runtime/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/private/tmp/swingsync-review017-reproduce \
  tests/test_full_fold_admission.py tests/test_full_fold_training.py
```

## Reviewed behavior

Admission authenticates the independently known015 audit and BOTH known016 smoke ledgers, each smoke's five checkpoint bytes, original source maps/config/runtime, fixed pipeline and original fold2 smoke counts. New fold3/4 admission retains its own train/inner membership and train-only class counts; old fold2 smoke metadata is a prerequisite witness, never a replacement partition. No outer cache arrays or smoke checkpoint tensors are opened by this wrapper.

The launcher persists exclusive job/process/config/seed/runtime/source identity before constructing a fresh head. It accepts no warm-start parameter and requires the seeded initial tensor hash to match the known smoke INITIAL hash. HeadExperiment receives precisely train+inner paths and uses unchanged original20epoch configs, training-only class weights, whole-clip order and inner-only full-clip PCE selection. It contains no outer evaluator or product promotion path.

Checkpoint chunks stop at epoch boundaries (default operational limit500). Every intended next target is durably reserved before advance. Only one orphan at that exact reserved next target can be restored; provenance and typed progress must match. Ambiguous/unreserved checkpoints are refused. Continuations preserve optimizer/RNG/cursor/selected weights; completed no-ops recheck dependencies, all durable checkpoint bytes, runtime, summary and ledger without construction or mutation. Input/source/checkpoint/ledger drift around publication fails closed; conflicts preserve outside edits for inspection.

Full completion requires20 epochs and the exact fold-specific update budget. Final checkpoints retain both continuation model and selected best_model; future evaluation must use best_model. All outer/product/advice gates stay closed. docs/full-fold-training.md correctly separates actual fit completion from outer comparison, hardware latency, upload-domain reliability and coaching evidence.

## Reviewed file identities

```text
07534e96b504b4346c42d0bfc526599c8b82c82d880cb577705b17b3c8e9ca81  backend/swingsync/research/full_fold_admission.py
86833cb8b39da29931979c8b0fdf75fea8c11e4ffa6700a4f82ba6e971b34ffa  backend/swingsync/research/full_fold_training.py
3765d0d5daebf117c6fc6b567c22d4c6ce0b9ed4b6f934c93de9f260b4520ed8  scripts/train_full_fold.py
5d8a36a316363cca11d1cfcb2f9584fb18d946345e561929a4494da64411b375  tests/test_full_fold_admission.py
97f69438f3966c8134ed4fb1587cd3c52af03670c3999317dd9ba2e92d59c082  tests/test_full_fold_training.py
b12401d9c97ff1baff73f6a1ed5d87634985bded0a15ed8f336e3f80ba0bf3ba  docs/full-fold-training.md
```

## Remaining boundary

Coordinator must finish required checks and verify reviewed code pushed/CI passed before actual fits. Any actual jobs need a later read-only saved-artifact audit; launch/partial state is not completion. All six heads must finish before this training increment is complete; outer evaluation and the three-fold aggregate require a separate reviewed increment. Historical fold1 was already observed, source groups are not verified unseen golfers, and this review provides no empirical accuracy, serving, upload-domain or coaching validation.

## Actual saved-artifact audit — first completed head only

Read current status/backlog again at the actual-artifact review boundary. Coordinator reported implementation `cfcde5e79784131d40a027bd1920104244ca3e8e` pushed and all six CI jobs in43945570455 passed before fitting; remote/Git verification remained coordinator-owned. The fold2-temporal worker was active during this read-only audit. The reviewer did not acquire/touch the training lock, inspect unfinished payloads, construct a model, forward/fit, decode video or rehash the feature-cache inventory.

**Verified fold2-frame only; study remains in progress.** Job `56295ab1806e48fda5dea64cf8012963`, known runSHA `9ad9f31f7e5b516b186ee797ba62477eca340d8b2baf208175793ce75cd34041`, genuinely saved20 epochs/4,180 updates and20 epoch-end checkpoints. No material findings.

Read-only verification authenticated that known ledger before and after inspection, all20 checkpoint byte hashes and `weights_only=True` CPU payloads, both known016 smoke witnesses/all ten smoke checkpoint bytes, known015 ledger/report hashes, current training source/config/root-lock bindings, and the frozen registration sources. Runtime/config and recorded initialization bind to the original20epoch frame config and independently known fresh smoke INITIAL tensor hash. No earlier trained head/smoke-final tensor is an initialization input.

Regenerated fold2 from actual annotations and matched the saved manifest:836 train/214 inner/350 outer clips,350/88/142 source groups. Each checkpoint provenance covers precisely the1,050 train+inner cache fingerprints, matching authenticated015 metadata, with no outer IDs. Counts were independently recomputed from training annotations only:836 for each event and230,613 background; normalized inverse-frequency weights match ledger/checkpoint values. This checked existing fingerprint metadata rather than reopening all cache arrays/bytes.

For each epoch1–20, verified exact step209×epoch, zero cursor/loss accumulators at epoch end, deterministic next-epoch clip order, identical provenance, history matching the ledger prefix, earliest maximum inner-PCE selection, summary SHA,214-clip inner-only/full-clip context and metric aggregation arithmetic. Every checkpoint's best_model is bitwise identical to the model saved at its selected epoch. Final selection is epoch20; its final model tensor hash matches the ledger. All Adam parameter states have the corresponding exact update count, finite moments and original hyperparameters. Saved Python/NumPy/CPU-Torch RNG states are structurally restorable without constructing a model or performing any forward/update.

The saved one-epoch partial snapshot and final ledger share the same job/immutable identity and byte-identical first checkpoint `cc171cbe31787af07b65b12f8eef1b53f9523a9879d70e0e71a7f0aedddb4ec3`. Attempts are one partial209-update epoch followed by explicit same-job continuation to4,180 updates, not duplicate jobs. Recorded attempt elapsed22.952765s and192.248963s excludes earlier import/admission and is not inference latency. The final no-op witness's before/after maps are identical for all21 ledger/checkpoint files and match current bytes; attempt count remains2 and update count4,180.

All outer/normal-model/serving flags remain false and advice empty. The audit verifies saved training/selection provenance and arithmetic, not recomputed inner predictions, new outer accuracy, uploaded-video generalization, unseen golfers or coaching reliability. Remaining heads require their own completed-artifact verification; no aggregate additional-fold result is admitted here.
