# Additional-fold training smoke

This is the reviewed launcher for backlog `003h2`, separate from the historical
fold-1 driver. It admits the fixed folds 2–4 study and the independently known
completed cache audit, then runs **only fold 2's five-update training smoke** for
one fresh head at a time. A completed smoke is not completed 20-epoch training,
model selection, an accuracy estimate, or a serving checkpoint. `--mode full`
explicitly fails closed; a separately reviewed full-run launcher is `003h3`.
No training result is recorded by this guide.

## Prerequisites and evidence

Use the original local `.tools/ml-runtime/bin/python` environment and repository
path. The existing cache identity checks include exact versions, implementation
bytes and historical Python code identity. A source-only checkout elsewhere or
different CPU package build is not admitted merely because versions look similar.
No model/backbone download, video decoding, image-feature extraction or paid API
is used by the smoke.

The coordinator must independently review the launcher and commit/push it before
any real fit; verify that registration commit
`2f1fbebf6c5eb776182383af63a28ec98b161afe` was already pushed, inspect active
processes and the shared lock, and record the implementation commit/source hashes
and launch identity. These are workflow prerequisites. The launcher records its
Git base revision and exact source/runtime bindings; it does not make network
requests to prove a remote push or pretend that a local flag proves review.

Admission requires the unchanged `experiments/additional-folds-v1.json`, original
configurations/lock/model implementations, actual `golfDB.mat`, regenerated
source-grouped manifests, and completed immutable audit:

- Ledger: `artifacts/additional-folds-015/audit-v1/run.json`, SHA256
  `fedd0ab2d2e51cf3e22c6c4eb453cd0629e4fb6fc5fc29db03b677070109951f`.
- Audit report: `audit.json` beside that ledger, SHA256
  `282f540880ca111cae2723c971951f7e577d67858ce6762beea1301f49b9666e`.

These expected hashes are independently pinned in code, not taken from a
self-asserted current ledger. The old read-only report is not itself a training
catalog. Admission derives precisely the requested new train/inner partitions;
no outer feature arrays enter the driver. Metadata checks may inspect historical
catalogs and all fold manifests, but no outer prediction, PCE, tuning or selection
is performed.

## One expensive worker, explicit continuation

Run these commands **sequentially** from the repository root. Keep generated
checkpoints/ledgers under ignored `artifacts/`; never commit their media or weights.
Use a new output directory only for an intentional new job, not as a workaround
for a failed resume. The shared `artifacts/training/active-training.lock` rejects
concurrent expensive workers.

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .tools/ml-runtime/bin/python scripts/train_additional_fold.py \
  --annotations data/golfdb/golfDB.mat \
  --manifests-dir artifacts/golfdb/preparation-002-all \
  --audit-run artifacts/additional-folds-015/audit-v1/run.json \
  --model frame --fold 2 --run-dir artifacts/additional-folds-016/frame-smoke-v1 \
  --max-new-batches 2
```

This saves two updates and reports `partial`. Continue that same job explicitly
with the same arguments, removing `--max-new-batches 2` and adding `--resume`.
This performs the remaining three updates, reaching **five cumulative updates**,
not five additional updates. Then repeat the pair of commands with `--model temporal`
and `artifacts/additional-folds-016/temporal-smoke-v1`, after the frame process ends.
Both heads must genuinely complete before a future full run is admitted. An
additional `--resume` on a completed job revalidates inputs, runtime, ledger and
checkpoint bytes and returns byte-identical saved state without constructing a
head or performing further updates.

All original configuration fields stay unchanged: seed 20261105, CPU, one Torch
thread, deterministic algorithms, full-clip/no-augmentation input, batch size 4,
learning rate 0.001, and temporal hidden size 64. The model begins with a fresh
seeded initialization; there is no previous-head or smoke warm start. Class
weights are recomputed using **all fold-2 training labels only**, compared with
the independent audit counts. Inner arrays are authenticated by the unchanged
constructor, but no inner example is forwarded or scored. A preflight guard
requires more than five training batches before the first epoch boundary, where
the historical driver would otherwise run inner validation.

Before model construction, `run.json` records PID/process start, job ID,
configuration/seed, input and dependency hashes, runtime, source identities,
run directory and shared lock. Every update has an exclusive immutable checkpoint
with optimizer/RNG/data-cursor state. A crash after physical checkpoint publication
but before its ledger pointer can recover exactly the next update, after exact
provenance/step validation; ambiguous extra checkpoints fail closed. Resume refuses
changed input, code, runtime, configuration, model or data partitions. Dependencies,
all durable checkpoint bytes and ledger publication bytes are checked around
successful publication. Failed/interrupted state never claims smoke completion.

`state: completed` is scoped by
`evidence_type: five-update-training-only-smoke-not-completed-benchmark-training`.
`full_training_completed`, `inner_validation_evaluated`, `outer_test_evaluated`,
`accuracy_metrics_computed`, `normal_model_available`, and `validated_for_serving`
remain false, and `advice` remains empty. Counts and elapsed smoke time are not
accuracy or inference-latency measurements.

## What remains

`003h3` must implement/review full-run admission bound to both completed smoke
witnesses, fresh per-fold/per-model initialization, the unchanged 20-epoch budget,
resumable provenance and inner-only earliest-tie selection. It must separately
implement/review fixed outer evaluation, reference-64-reset primary context and
full-clip reporting, recorded-hardware latency, and aggregate results only after
all three additional folds complete. Never warm-start from these smoke weights.
Historical fold 1 was already observed; additional outer clips previously appeared
in historical fold-1 train/inner data. A future four-fold summary is neither wholly
prospective nor verified unseen-golfer or uploaded-video validation. Coaching and
normal model availability stay gated independently.
