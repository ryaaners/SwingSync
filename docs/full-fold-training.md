# Fixed additional-fold full training

This is the separate launcher for the training portion of `003h3`. The immutable
five-update smoke command remains smoke-only. This command trains one fresh frame
or temporal head on an official fold 2, 3 or 4, using the unchanged original
20-epoch configuration and frozen ImageNet MobileNetV2 features. It has no outer
evaluation or serving path. A completed single head is not the completed
three-fold comparison, uploaded-video validation or coaching evidence.

## Required admission

Use the original local `.tools/ml-runtime/bin/python` environment, repository
path, actual annotations and saved manifests/cache audit. The direct script
bootstraps its repository import path and needs no `PYTHONPATH`. Exact historical
pipeline/runtime/source identity still applies; a portable synthetic fixture does
not authorize a different actual extraction build or recording.

The coordinator must review, commit and push the new launcher and verify its CI
before any actual fit. The command records its local Git base and source hashes;
it does not make network requests or claim that a boolean proves external review.
Inspect actual workers, saved job identities and the shared lock first.

`full_fold_admission.py` authenticates the known 015 audit and both independently
reviewed completed 016 smoke ledgers and their five checkpoint files each:

- Frame run SHA256: `eb0728446686ad3db71960da17a131722f103cff0cb575589152795c70294657`.
- Temporal run SHA256: `9b0bc9acc8c65a5a1898e9bebc415bdbed8b8c9c49e9267ec78e6d8b2ed41736`.

Smoke is a prerequisite witness, never initialization weights. Before any update,
the fresh full head's seeded tensor hash must equal the witnessed *initial* hash.
Only a checkpoint from this same full job can subsequently restore optimizer,
RNG, model and data-cursor state. There is no warm-start input parameter.

The requested fold's complete train and inner-validation paths are regenerated
and authenticated. Permanent official split identity is preserved even when an
old catalog role changes for a new fold. Class weights use every relevant training
label, excluding inner and outer labels. Inner validation selects highest standard
PCE with the earliest exact tie; training and inner context remain full-clip.
Outer caches never enter the driver. Fixed baseline/temporal configs, seed,
no-augmentation policy, CPU one Torch thread and deterministic algorithms stay
unchanged. Reencoded native cache timestamps are not original-source clock proof.

## Run sequentially and explicitly resume

From the project root, after coordinator review and the verified pushed checkpoint:

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .tools/ml-runtime/bin/python scripts/train_full_fold.py \
  --annotations data/golfdb/golfDB.mat \
  --manifests-dir artifacts/golfdb/preparation-002-all \
  --audit-run artifacts/additional-folds-015/audit-v1/run.json \
  --smokes-dir artifacts/additional-folds-016 \
  --fold 2 --model frame \
  --run-dir artifacts/additional-folds-017/fold2-frame-v1 \
  --max-new-batches 209
```

The bounded invocation is partial, not completed training. Continue the same job
with identical arguments, removing `--max-new-batches 209` and adding `--resume`.
A bounded limit controls new updates in that invocation; it does not replace the
fixed total budget. Actual complete totals are 4,180 updates for fold 2, 4,200 for
fold 3 and 4,280 for fold 4, for each head. Never start a new directory to evade
an interrupted or refused resume. Train the other head/folds sequentially only
after inspecting completion/ownership. Run at most one expensive worker.

The immutable operational checkpoint interval defaults to 500 batches, capped at
every epoch boundary. Actual folds therefore normally save 20 epoch-end states;
a bounded partial pause adds a checkpoint. This does not change optimizer updates,
inner validation timing or selection. Each selected epoch can be checked against
the final payload's `best_model`; the final `model` may be a different epoch.
Future evaluation must use the selected `best_model`, not blindly the final state.
The checkpoint interval may be chosen explicitly before creating a job, but must
remain identical when resuming it.

Before head construction the ledger records job/PID/process start, config/seed,
source/input/runtime hashes and paths. Before advancing a chunk it persists the
exact intended checkpoint target. An orphan physical checkpoint may be adopted
only when it is the single exact reserved next target, with matching driver
provenance and progress. Ambiguous/unreserved files fail closed. Completed resume
validates saved summary, initialization/counts/weights, all checkpoints, sources,
runtime and ledger bytes and returns without constructing a head or mutating files.

Successful publication checks expected predecessor and output ledger bytes,
checkpoint bytes, runtime and dependencies. Per-batch cache checks remain in the
unchanged head driver; whole-admission checks occur at persistence boundaries,
not every optimizer update. An outside ledger edit is preserved rather than
overwritten; `run-conflict-*.json` records the failure and blocks further automatic
continuation until the saved evidence is inspected. Failed/interrupted jobs do not
claim completion. Keep all raw jobs/checkpoints under ignored `artifacts/`.

Attempt elapsed time starts after admission and includes head construction,
cached-feature reads, updates, internal validation, checkpoint verification and
publication. It excludes earlier package import/admission and is not inference
latency, an end-to-end app measurement or minimum hardware evidence.

## What remains after training

All six heads must finish the fixed budget before the registered additional-fold
comparison is complete. A separate reviewed evaluator must authenticate selected
weights, preserve official independent-event PCE and native-PTS timing errors,
report reference-64-reset as primary and full-clip separately, and measure the
registered training-only latency inputs on recorded hardware. Do not tune against
new outer results or the completed historical fold 1. Aggregate additional-fold
results only after all three folds complete; partial processes are not final data.

Historical fold 1 was already observed and new outer clips appeared in its train
or inner partitions. A four-fold summary is not wholly prospective or verified
unseen-golfer evaluation. Uploaded-video/crop reliability, independent pose
references, outcome equivalence/generalization, original-source timing and coaching
admission remain separate gates. `normal_model_available` and
`validated_for_serving` stay false and advice empty throughout this research job.
No actual training or performance result is asserted by this operational guide.
