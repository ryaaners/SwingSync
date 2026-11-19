# Independent review — execution018

Saved-artifact review completed 2026-11-18. **No open material findings in the six-head training completion evidence.** All six heads have durable original20-epoch training budgets and inner-only selection. This does not complete backlog003h3: registered outer evaluation, hardware latency and the additional-fold comparison remain unfinished.

## Scope and independence

Read AGENTS.md, autonomy/CURRENT_STATUS.md, autonomy/backlog.json, autonomy/roles/reviewer.md and review-017.md. Reviewer owns only this report. No Git operation, implementation edit, actual fitting, model construction/forward, video decode, outer evaluation, feature-array audit or expensive-work lock acquisition was performed. Checkpoint payloads were read on CPU using `.tools/ml-runtime/bin/python` and `torch.load(..., weights_only=True)`; no environment synchronization was performed.

The sandbox refused `ps`; coordinator independently reported old controller16452, old worker18746 and recovery5829 absent and shared lock free. Process/lock observations are coordinator-supplied evidence, not independent reviewer observations. The original017controller remains an interrupted producer snapshot: its saved `state=running`, five completed entries and dead active-child PID are not a completion receipt. It must be preserved and a new018verification receipt used for the aggregate completion claim.

## Durable results

| Fold/head | Job ID | Updates | Epochs | Selected inner epoch | Attempts |
| --- | --- | ---: | ---: | ---: | ---: |
| 2/frame | 56295ab1806e48fda5dea64cf8012963 | 4180 | 20 | 20 | 2 |
| 2/temporal | b2e6f4175c5546f4a192c61f17a036a4 | 4180 | 20 | 18 | 1 |
| 3/frame | b175351717fd47fdb114ec8df3b8b596 | 4200 | 20 | 20 | 1 |
| 3/temporal | a55c5b2c5b11427cafb13c47b12b5e20 | 4200 | 20 | 18 | 1 |
| 4/frame | 589d37cbba5a468f8b6de97f28e2fab1 | 4280 | 20 | 19 | 1 |
| 4/temporal | ba31e5bdebcc4565b2f1857ad56c872e | 4280 | 20 | 19 | 2 |

Total:25,320 durable optimizer updates,120 completed head-epochs and120 immutable epoch-end checkpoints. These are training completion counts, not an accuracy aggregate. Each head retains both continuation `model` and selected `best_model`; future evaluation must use `best_model`.

Authenticated final run.json identities under artifacts/additional-folds-017/:

```text
9ad9f31f7e5b516b186ee797ba62477eca340d8b2baf208175793ce75cd34041  fold2-frame-v1/run.json
746fb1519c75861a9fbdba6787904647106499e7cbf6de6c55845755fcbfcec5  fold2-temporal-v1/run.json
91718ef607e81568bc3399852fa6ab9100295416492c442375dc0359789c72bb  fold3-frame-v1/run.json
fabc1a4003114aa5debd748772e07178e0c710ae87e57a23ddeb93e2d75392af  fold3-temporal-v1/run.json
c57fb2e619af5d82dacd4be51cc821dda5e86ec17403a483f843cec699f4af85  fold4-frame-v1/run.json
08f0ce094785a80e9e331b3f2a5172c39295b917e5d5c5cd48e003a29ec2a33b  fold4-temporal-v1/run.json
```

Review017's independent20-payload fold2-frame audit is retained rather than repeated. This review reauthenticated its known ledger/no-op identity and overall metadata, and independently loaded the other100 completed checkpoint payloads, verified their file hashes and checked their ledger/history/provenance/selection relationships. Thus the combined017/018 independent review covers all120 completed payloads.

## Provenance, grouping and payload verification

- Independently authenticated known015 runSHA `fedd0ab2d2e51cf3e22c6c4eb453cd0629e4fb6fc5fc29db03b677070109951f`, reportSHA `282f540880ca111cae2723c971951f7e577d67858ce6762beea1301f49b9666e`, all recorded source/input metadata dependencies and both historical catalog bytes. Both known016 smoke ledgers and all ten smoke checkpoint file hashes remain exact. Smoke payload behavior was independently reviewed in016/017; this review did not repeat those payload audits.
- Original configs, registered frozen source maps, launcher/admission source maps and all three root/pose/tabular locks match their known identities. Saved runtime matches the isolated CPU Python3.11.14/Torch2.9.1/NumPy2.2.5 environment, deterministic policy and single-thread declaration. No new model initialization was performed for review. Each saved fresh-initial tensor hash equals that head type's independently known smoke INITIAL hash, not its trained final hash.
- Regenerated all three actual source-grouped manifests from the original MAT annotations: train/inner/outer clips836/214/350,839/211/350,853/197/350; source groups350/88/142,340/86/154,352/88/140. Clip and source partitions are disjoint; official outer assignments remain exact. Source recordings are not verified unseen golfers.
- Recomputed counts solely from training annotations: each event836/839/853 respectively, background230613/230926/229827. Normalized inverse-frequency class weights agree exactly with float64 normalization followed by float32 conversion. Saved ledger and checkpoint counts/weights match.
- Verified the1400 audited cache/source/frame metadata mappings against annotations and both historical catalogs without opening cache arrays or media. Each head checkpoint covers precisely its1050 train+inner cache fingerprints and no outer IDs; saved SHA/key pairs match authenticated015 metadata. Reconstructed admission watched-map hashes from these existing fingerprint metadata and known source/smoke witnesses. This is metadata authentication, not a fresh full cache-byte scan.
- For every newly reviewed checkpoint: exact epoch×fold-specific batches, zero epoch-end cursor/loss accumulators, deterministic next-epoch whole-clip order, identical provenance, ledger history prefix and summary hash. Inner metrics identify only the correct214/211/197 validation clips, full-clip context and closed outer gate. Numeric values are finite, ranges/nonnegative losses valid, per-event PCE represents integer inner hits, percentages and mean aggregation agree. Predictions/PCE were not recomputed.
- Every checkpoint's `best_model` is bitwise equal to the `model` saved at its earliest maximum inner-PCE epoch, including final4-temporal selection19. Model/selected tensors and Adam moments are finite; Adam parameters retain original hyperparameters, correctly shaped moment tensors and exact per-parameter optimizer step counts. Saved Python/NumPy/CPU-Torch RNG state is structurally restorable using standalone RNG objects, with no model construction or update. Final model tensor hashes agree with each final ledger.
- All completed no-op witnesses cover exactly21 files per head (ledger+20 checkpoints), with identical before/after maps matching current bytes and unchanged attempt/update counts. The last witness is artifacts/additional-folds-018/fold4-temporal-noop-witness.json; earlier five are under017. No actual completed training was rerun for review.

## Recovery continuity and limits

Authenticated artifacts/additional-folds-018/interrupted-run-snapshot.json SHA `9dcbaceef6b4fac7a646e4d8af2c8725c8e1598dcb2bf787335b620fef9b1d87` against recovery-request.json. The final4-temporal job preserves the same job ID, immutable source/config/runtime/admission/initialization identity, all11 original checkpoint rows and their file bytes, the first11 history entries and the original attempt record. The recorded reservation was2354→2568 updates. Recovery continued this existing job to4280 updates; it created no replacement head/job.

The original attempt has no `ended_at`, state or elapsed time because its process disappeared. Those missing values remain missing, with no invented stop time or full-run timing. The recovery attempt completed and records368.37176837499965seconds; this excludes the stopped attempt and does not establish full training wall time or inference latency. Fold2-frame's separate017 first-epoch partial+19-epoch continuation remains the two-attempt witness independently reviewed in017.

Every final ledger retains `outer_test_evaluated=false`, `normal_model_available=false`, `validated_for_serving=false` and empty advice. Frozen cached timestamps retain the earlier reencoded-clip PTS scope; no original-source clock equivalence was newly established. Historical fold1 had already been observed and other folds' outer clips appeared in fold1 train/inner. No wholly prospective four-fold, uploaded-video, unseen-golfer, dependable-event, spatial/outcome or coaching claim follows from training completion.

## Reproduction

Use the isolated interpreter without synchronization or actual training. For each known ledger above, hash its bytes first, require the known identity, then hash/load its20 saved checkpoint paths with `weights_only=True`. Recompute earliest maximum from each history prefix and compare every `best_model` tensor to the corresponding epoch checkpoint's `model`; verify each witness map against the same current files. Core selection reproduction:

```python
import hashlib, json
from pathlib import Path
import torch
root = Path('artifacts/additional-folds-017/fold4-temporal-v1')
raw = (root / 'run.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == '08f0ce094785a80e9e331b3f2a5172c39295b917e5d5c5cd48e003a29ec2a33b'
s = json.loads(raw)
models = {}
for epoch, row in enumerate(s['checkpoints'], 1):
    path = Path(row['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    p = torch.load(path, map_location='cpu', weights_only=True)
    models[epoch] = p['model']
    scores = [h['validation']['pce'] for h in p['progress']['history']]
    chosen = scores.index(max(scores)) + 1
    assert p['progress']['best_epoch'] == chosen
    assert all(torch.equal(v, models[chosen][k]) for k, v in p['best_model'].items())
assert chosen == 19 and s['summary']['optimizer_steps'] == 4280
```

Independent saved-artifact audit passes. Coordinator software checks, final018completion receipt and status/backlog claim review remain separate publication steps; this report does not certify an unobserved receipt or unrun CI.

## Final receipt and publication-claim acceptance — 2026-11-19

Independently authenticated the NEW artifacts/additional-folds-018/completion-receipt.json SHA **7c37a7d1b04e618dda41673f2a0b5e252b23f1822e6d3fc394590cbc0b06b407**. Its artifact type and scope identify completed training, explicitly excluding outer evaluation and serving. Verified all six entries against the known current run.json bytes: job/base/state/config/config-file/manifest identities, train counts, initialization/final tensor-hash metadata, epochs/updates/inner selection, checkpoint count, attempts exactly as recorded, selected/final checkpoint paths and file hashes, and21-file completed no-op witness identities/counts. Reused the completed017/018payload audit; no model load, forward or fit was repeated for this receipt check.

Recomputed120head-epochs and25320durable updates from those six ledgers. Receipt protocol/dataset/known015run+report/known016smoke/source bindings match local authenticated bytes. Its controller snapshot SHA0401ca7d9ddc26c449f3cfd2101706bddcb7197292eae0d77314dfac1b1bd4a8 remains the preserved running/five-entry producer snapshot. Recovery snapshot identity, same-job eleven-checkpoint/history prefix and the original attempt's missing end/elapsed fields match the final ledger. Process-absence declarations remain coordinator observations. No stopped-controller producer success or unknown timing was fabricated.

Reviewed autonomy/reports/additional-folds-training-018.json and validation-018.md, CURRENT_STATUS.md, backlog.json, README.md and the relevant experiments/model-card/remaining-work/architecture/data-contracts documentation. The small aggregate's job fields are exact projections of the authenticated receipt and current ledgers; its runtime, seed, all three dependency-lock hashes, recovery description, counts and closed gates agree. `003h3-training=completed`, `003h3-eval=pending`, and parents003h3/003h remain in progress. Published claims correctly distinguish selected inner epochs from additional-fold held-out accuracy, disclose prior fold1/source-recording/PTS scope, retain unavailable normal models and empty advice, and identify the unfinished latency prototypes/missing admission helper as future work. No material publication-claim findings.

Coordinator reported refreshed2026-11-19software checks:653base tests/18optional skips,929ML tests/1optional skip,Ruff,9frontend tests/build and current npm audit0, with unchanged three uv locks and frontend package pins. These are coordinator-run checks; this reviewer did not rerun them or independently adjudicate the separate upstream advisory. Review acceptance covers the evidence and claims above. Final intended-file staging, exclusion of unfinished prototypes/raw artifacts, commit/push/remote-SHA verification and any later CI status remain coordinator-owned and must be reported only when observed.

**Accepted for coordinator publication within the completed-training scope. No open material findings. No new outer accuracy, hardware latency, product inference or coaching admission is certified.**
