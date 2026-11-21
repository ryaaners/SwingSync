# Additional-fold CPU latency

The research command measures the fixed training clips registered in
`experiments/additional-folds-v1.json`, after the six additional-fold heads have
completed training. It computes no event accuracy and does not enable automatic
models or advice. Synthetic tests establish software behavior only; they are not
hardware latency measurements.

The study gate requires the independently reviewed execution018 completion
receipt SHA256
`7c37a7d1b04e618dda41673f2a0b5e252b23f1822e6d3fc394590cbc0b06b407`.
It authenticates the original configurations, protocol, source/runtime,
execution015 audit, both execution016 smokes, all six completed full-training
ledgers, all 120 epoch checkpoints, inner-only selection and completed-resume
witnesses. The preserved execution017 controller remains stale and cannot serve
as completion evidence. Each head loader uses the final checkpoint's saved
`best_model`; the final epoch's model is not a fallback.

Use the isolated ML environment and reviewed, published implementation. The
following example creates a new one-shot timing directory; choose a fresh path
explicitly. Inspect existing evidence before invoking it. The shared expensive
work lock prevents concurrent training, extraction, evaluation or timing.

```sh
.tools/ml-runtime/bin/python scripts/benchmark_additional_folds.py \
  --annotations /absolute/path/to/authenticated/golfDB.mat \
  --manifests-dir artifacts/golfdb/preparation-002-all \
  --audit-run artifacts/additional-folds-015/audit-v1/run.json \
  --smokes-dir artifacts/additional-folds-016 \
  --training-runs-dir artifacts/additional-folds-017 \
  --training-receipt artifacts/additional-folds-018/completion-receipt.json \
  --expected-receipt-sha256 7c37a7d1b04e618dda41673f2a0b5e252b23f1822e6d3fc394590cbc0b06b407 \
  --clips-dir /absolute/path/to/authenticated/preprocessed/clips \
  --run-dir artifacts/additional-folds-019/latency-v1
```

Supply the actual local annotation and preprocessed video paths used by the
authenticated audit. No video, cache, checkpoint or reference weight is fetched
by this command. The full official MobileNetV2 weight file must already exist in
`$TORCH_HOME/hub/checkpoints/`, or the project's `data/torch/hub/checkpoints/`
when `TORCH_HOME` is unset. Its complete SHA256 is checked before construction.
A missing or changed file produces an `unavailable` ledger and no completed
report.

The fixed inputs are training clips 0/1 on fold 2, 6/7 on fold 3, and 0/1 on fold 4:
six clip observations across four unique clips. Each fold warms up on exactly
16 frames from its first training clip, with both selected heads and both
registered contexts. One measured pass per clip uses backbone batches of 16 and
retains the final partial batch. CPU execution uses one Torch thread and
deterministic algorithms. Detected accelerators are recorded but are not timed.

The measurement retains actual decoded PTS and requires exact agreement with
the authenticated cached float64 timestamps. It checks source/cache bytes,
frame counts and the frozen preprocessing/backbone identity before timing.
Decode samples time retrieval of decoded frames; preprocessing includes RGB
conversion and the registered transform. Backbone samples include batch
stacking, forward and output transfer. Head timings cover each full clip under
`reference-64-reset` and `full-clip` separately. The composed duration for one
head/context adds its head duration to the shared feature-extraction wall time;
it is not a separately executed pipeline. Whole-clip wall time includes all four
head/context calls. Warmup, admission, integrity verification and metadata
publication are outside the stage timings.

`run.json` records identity and immutable inputs before model construction,
then warming/running/completed or failed/interrupted/unavailable state.
`latency.json` is published only after all six observations succeed; its SHA256
is retained in the completed ledger. Existing run directories are refused;
there is no automatic resume or extra repetition. Source/runtime/input drift,
invalid PTS, changed weights or a replaced predecessor prevent completion.
A separately named conflict witness preserves an externally changed ledger.
On a later failure, cleanup removes the just-published report only if its full
recorded file identity and SHA256 are unchanged. Changed or replaced reports,
including in-place corruption or reused inodes, remain as conflict evidence.
The failed ledger has `report_sha256: null`; a retained report is not authenticated
completed evidence and must be inspected together with that ledger.

One repetition of four unique training clips provides descriptive samples, not
population bounds, a minimum hardware requirement, application latency,
accelerator performance or uploaded-video reliability. Reencoded clip PTS do
not establish original-source or browser-clock equivalence. Accuracy,
coaching, pose reliability, outcome integration and serving gates remain closed.
