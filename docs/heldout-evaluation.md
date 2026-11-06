# Fixed fold-1 held-out evaluation

The prospective contract is `experiments/heldout-v1.json`. It binds the completed training ledgers, final continuation checkpoints and selected epochs before any outer-test feature extraction or predictions. This is one fold of project-trained event heads using a frozen ImageNet-initialized backbone. It is not a replication of pretrained SwingNet or a four-fold result. No test-driven tuning, configuration selection or replacement of failed results is permitted.

## Fixed inputs and selection

Use the previously audited GolfDB MAT and exact source-grouped fold-1 manifest: 840 training, 210 inner-validation and 350 official outer-test clips. Frame epoch 19 and temporal epoch 20 were already selected by the earliest maximum inner-validation PCE within the fixed 20-epoch budget. Load `best_model` inside each final checkpoint. In particular, the frame checkpoint's last-epoch `model` differs from its selected weights. No optimizer restoration, fitting or validation reselection occurs during held-out evaluation.

Both models use the same deterministic MobileNetV2 ImageNet V2 resize232/center224 features and pinned runtime. This differs from the reference's historical image preprocessing and temporal architecture. Test caches must be a separate exact test-only inventory; the training catalog remains immutable and excludes test clips. Verify source membership, annotations, byte hashes, feature signatures and actual re-encoded clip PTS. Original YouTube PTS cannot be recovered from these clips.

## Two separately reported contexts

1. **Reference 64-frame reset:** slice from frame zero into independent consecutive 64-frame chunks, with a final shorter unpadded tail. Reset temporal hidden state on each model call, apply row softmax and concatenate probabilities in original frame order before finding peaks across the full clip. This matches reference recurrence boundaries; it does not imply a complete reference architecture/preprocessing replication.
2. **Full clip:** call the same selected model once for the full sequence, matching its preregistered training/inner-selection context. Report separately. Do not choose a context from held-out scores. Independent-frame outputs may differ by floating-point rounding across batching, so compute and label both explicitly.

Both contexts use the existing audited official independent-event evaluator. Targets are chronologically occupied event frames (`np.where(labels < 8)` in the reference), including the preserved clip677 annotation-order anomaly. Tolerance is `int(max(np.round((events[5]-events[0])/30),1))`; a prediction is correct when its absolute frame error is at most that tolerance. For each class, use NumPy default `argsort` and its final index, recording NumPy2.2.5. Do not replace this with an ordered decoder, semantic-label repair or a different tie policy.

Report overall/per-event PCE, frame MAE and actual PTS-millisecond MAE separately per head/context, with exact coverage and anomaly count. The anomalous chronological slots are not corrected semantic labels. No headline held-out benchmark is published until all 350 clips are verified and processed. Persist per-clip results atomically; explicit resume requires identical contract, data, checkpoints, runtime and code. Interrupted computation may continue without tuning, while changed results or provenance fail closed.

## CPU latency scope

Benchmark only fixed **training clips0 and1**. CPU one Torch thread, batch16, one fixed 16-frame warmup and one measured pass per clip. Record initialization/warmup separately. Time actual sequential decoding, RGB conversion/registered preprocessing, backbone forward and each selected head/context separately. Preserve decoded PTS and report the number of frames, media durations and timing denominators. Label per-batch versus per-frame percentiles explicitly. Time integrity preflight separately or state its exclusion. Do not present cached-head speed as end-to-end video latency or training elapsed time as inference speed. The measured two-clip sample does not establish minimum hardware, battery use or representative user-video latency.

The evaluator and benchmark serialize expensive work using the existing shared lock. They do not enable the app, alter training, generate advice or open supervised shot-outcome gates.

Sources: pinned official [evaluation loop](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/eval.py) and [PCE implementation](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/util.py). Independent source/contract review is recorded in `autonomy/reports/review-006.md`.
