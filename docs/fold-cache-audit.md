# Additional-fold cache audit

This read-only stage checks whether the existing frozen per-clip ImageNet features can be reused under newly registered official GolfDB folds. It does not fit a model, load a trained head, run a backbone, compute predictions or evaluate PCE. It does not enable the existing fold1 training/evaluation launcher for other folds.

## What stays fixed

The original MAT version, deterministic image transform, frozen ImageNet V2 backbone, two head architectures and original20-epoch configurations remain unchanged. Classifiers for any additional fold must start from scratch; fold1 head weights and its class weights are not transferable training inputs. New class weights may use only that fold's training labels. Inner validation remains grouped by source recording and drives checkpoint selection; outer outcomes cannot drive a sweep, context change or reselection.

Fold1 has already been observed. A later summary that includes it must disclose its historical status; it is not a wholly prospective four-fold study. Source-recording separation does not establish unseen-golfer or arbitrary uploaded-video generalization. See [the additional-fold protocol](additional-folds-protocol.md).

## Read-only admission

The audit authenticates the exact two completed historical catalogs (1050original training/inner clips and350original outer-test clips), checks disjoint coverage, regenerates the official source-grouped manifests and streams one immutable cache at a time. Each cache must retain its permanent clip/source/official-split identity, raw annotation event indices, inclusive historical indices, float32 features, labels and float64 native timestamps. Its pipeline must match the frozen original registration. The audit binds fresh video-file SHA256 values to the recorded source bytes.

The cache field `outer_split` is permanent annotation metadata. A new experiment's train/validation/test assignment comes from its regenerated manifest; no cache identity or label is rewritten to match that assignment. Reuse is justified only because the shared image backbone and preprocessing were frozen independently of GolfDB event-head fitting. Reusing a trained fold1 head or a dataset-fitted transform would cross this boundary.

No fresh video decode or image-feature recomputation is claimed by this audit. Timestamp values are authenticated through the previously verified cache/catalog/source-byte chain. They remain reencoded clip PTS, not original YouTube timestamps or validated physical swing time. The existing original-source precision gate stays open.

An exclusive output directory and the shared research-work lock prevent duplicate or overlapping scans. The run records identity before scanning and remains partial/failed until exact full coverage and final integrity checks pass. A completed audit publishes a read-only catalog and per-fold partition/class-count evidence. It is not an executable training admission token.

Future fitting still requires a published registration commit, an independently reviewed completed audit with its expected SHA, a reviewed launcher that binds both, a training-only smoke, and separately tracked resumable jobs/checkpoints. Inspect actual running processes before starting anything; one expensive job at a time. Models, media, full cache catalogs and raw audit artifacts remain outside ordinary Git commits.

## Run in the original research environment

This audit requires the original locked research Python environment; runtime/build and marshaled forward-code identities must match the historical cache pipeline. A different machine/source path may fail that strict check even though a software fixture can run there. Do not edit historical fingerprints to bypass it. The synthetic tests use their own runtime-aware registration and do not claim pretrained extraction occurred.

```sh
.tools/ml-runtime/bin/python scripts/audit_fold_caches.py \
  --annotations data/golfdb/golfDB.mat \
  --train-catalog artifacts/features/fold1-job-v1/completed-catalog-v1.json \
  --test-catalog artifacts/features/fold1-test-job-v1/job.json \
  --manifests-dir artifacts/golfdb/preparation-002-all \
  --clips-dir data/golfdb/videos_160 \
  --protocol experiments/additional-folds-v1.json \
  --output artifacts/additional-folds/my-audit
```

Use a fresh output directory. `run.json` records partial/failed/completed state, and `audit.json` is published only after full verified coverage. Every declared protocol dependency, source, cache, video and approved output byte sequence is checked around final publication; changed inputs produce a failed ledger and remove the report. There is no automatic resume or training mode. Inspect actual process identity and the shared lock before another attempt.

## Completed first audit

Registration/code checkpoint `2f1fbebf6c5eb776182383af63a28ec98b161afe` was pushed before the scan; its six CI checks passed. One actual audit completed1400caches/394177frames with no fitting, model execution, fresh decoding or PCE. Run SHA `fedd0ab2d2e51cf3e22c6c4eb453cd0629e4fb6fc5fc29db03b677070109951f` binds report SHA `282f540880ca111cae2723c971951f7e577d67858ce6762beea1301f49b9666e`. The18.360s interval measures file/array verification and publication, not inference latency. Original files remain unchanged.

Independent review checked all1400catalog/MAT mappings, regenerated partitions, train-only counts,23input/12implementation bindings and sample cache/source bytes0,10,677. The source-clock limitation and strict original runtime remain explicit. See [small machine evidence](../autonomy/reports/additional-folds-audit-015.json) and [review](../autonomy/reports/review-015.md). The next step is a reviewed launcher and registered training-only smoke; the audit report is still not a training catalog.
