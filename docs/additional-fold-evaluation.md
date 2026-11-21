# Additional-fold outer evaluation

This runner implements the fixed execution015 protocol for official GolfDB folds2–4, after all six execution017/018 training heads completed20epochs. Implementation and synthetic verification are not an outer accuracy result. The coordinator must independently review, push and verify CI before executing on real outer caches. Never repeat completed forwards to tune or improve their appearance.

`backend/swingsync/research/additional_study.py` authenticates the separately completed execution018 receipt, all six completed training ledgers and selected final checkpoint payload paths, original configurations, fixed source-grouped manifests and execution015 cache audit. The stale execution017 controller is not a completion witness. The independently known receipt digest is `7c37a7d1b04e618dda41673f2a0b5e252b23f1822e6d3fc394590cbc0b06b407`; passing a different self-computed digest does not authorize a different study.

## Coordinator invocation

Run from the dedicated repository using its existing isolated ML environment. Replace only the annotations and preprocessed clips directory with their already authenticated local locations; no downloads, fresh decoding or training are needed. The audit argument is its `run.json`, not its directory. Confirm the actual completed training parent directory and annotation path against saved ledgers before invoking.

```sh
.tools/ml-runtime/bin/python scripts/evaluate_additional_folds.py \
  --annotations /absolute/path/to/GolfDB.mat \
  --manifests-dir artifacts/golfdb/preparation-002-all \
  --audit-run artifacts/additional-folds-015/audit-v1/run.json \
  --smokes-dir artifacts/additional-folds-016 \
  --training-runs-dir artifacts/additional-folds-017 \
  --training-receipt artifacts/additional-folds-018/completion-receipt.json \
  --expected-receipt-sha256 7c37a7d1b04e618dda41673f2a0b5e252b23f1822e6d3fc394590cbc0b06b407 \
  --clips-dir /absolute/path/to/authenticated/preprocessed/clips \
  --run-dir artifacts/additional-folds-019/outer-v1 \
  --max-new-clips 1
```

Inspect the one-clip durable ledger and full probability artifact, then invoke the identical command with `--resume --max-new-clips 1049` in place of the first bound. Use `--resume` again after completion to authenticate saved arrays, native timestamps, metrics and aggregate report without loading a model, forwarding, appending an attempt or changing artifact bytes. Calling without `--resume` against an existing directory refuses to overwrite it. Source changes after the first real forward invalidate resume; finish software changes and review before the real invocation.

The CLI is standalone and its `--help` and malformed-bound handling do not require importing the optional ML runtime. Python API: `run_additional_evaluation(*, annotations, manifests_dir, audit_run, smokes_dir, training_runs_dir, training_receipt, expected_receipt_sha256, clips_dir, run_dir, protocol_path=DEFAULT_PROTOCOL, lock_path=DEFAULT_LOCK_PATH, resume=False, max_new_clips=None)`. Production uses the shared expensive-work lock. Isolated synthetic tests alone provide their own lock and mocked study admission; production receipt/protocol hashes remain fixed.

## Protocol and artifacts

Each of350 outer clips per fold receives both heads in both registered contexts: primary `reference-64-reset`, and separately reported `full-clip`. The frozen generic `heldout.load_selected_head` loads selected `best_model` only, with hidden size64. `heldout.predict_probabilities` accepts complete float32 `[T,1280]` features and resets temporal state per64-frame chunk in the primary context. There is no high-level fold1 heldout admission, model fitting, checkpoint substitution or event order repair.

Before any model construction, the job persists its PID/start identity, job ID, seed, CPU runtime, input/source/config/checkpoint/receipt fingerprint, Git base identity and attempt. Every outer cache and source video is bound to the known audit: cache SHA/key, full annotation identity, source ID, source video bytes, labels, feature shape, pipeline and float64 native PTS hash. Rechecking cached PTS against the authenticated historical chain is not fresh decode or original-source PTS equivalence.

`run.json` records an exact completed clip prefix, an in-flight clip identity and pending publication identities. The in-flight marker is saved before any selected-head forward. If execution stops after a forward but before exact probability bytes become durable, resume fails closed instead of repeating that clip. `foldF-clipID.npz` contains four full float32 `[T,9]` probability arrays, unchanged float64 `pts_seconds`, and scalar JSON metadata with the evaluation fingerprint and immutable input identity. Loading uses `allow_pickle=False`. Each artifact has a saved SHA256 and independently replayed per-clip metrics. Before publishing an NPZ, its expected digest and metrics are durably recorded. On explicit resume, an exact known NPZ orphan is authenticated and adopted without repeating its forwards. Unknown files, unknown or changed orphans, and a pending artifact that never reached disk fail closed for inspection. Output replacement compares its predecessor digest and refuses outside edits. The aggregate report also has a durable `pending_report` identity before publication: resume independently replays all saved probabilities and either adopts the exact known report without changing its bytes/stat identity or recreates an absent report without model construction or forwarding. Unknown or changed reports remain refused.

All immutable paths are authenticated by SHA256 between matching filesystem snapshots at invocation start. During the loop, every forward and publication checks pinned device/inode/size/mtime_ns/ctime_ns/mode identities, selected in-memory model tensor hashes and the CPU runtime. A write followed by restoration of the original bytes and mtime still changes ctime and is refused. Newly published artifacts are pinned after the temporary hard-link alias is removed. Full SHA verification runs before and after bounded completion, and on completed resume. This avoids rehashing the entire study around every forward while retaining byte authentication at invocation boundaries; it assumes the local operating system/filesystem is trusted. If detected drift follows final ledger publication, the runner retracts only its own unchanged final successor to the preceding running ledger. It preserves any outside edit to the ledger.

The official unchanged `events_from_labels`, `evaluate_clip` and `summarize_evaluations` define the benchmark. Per-event peaks use NumPy default `argsort` and its last index, with no stable-sort substitution. Targets are chronological occupied label frames, preserving and flagging the raw class-order anomaly (including clip677). PCE tolerance is `max(np.round((target[5]-target[0])/30),1)` and equality is correct. Frame errors and signed/absolute milliseconds derive from actual cached float64 PTS, never FPS. Raw class targets and predicted order diagnostics remain visible; no peaks are reordered or repaired.

No aggregate report is published until all1050 clips complete. `report.json` contains overall and per-event PCE, frame/native-PTS-ms MAE and order diagnostics by fold/head/context, a separate additional-three-fold aggregate and descriptive fold variation (individual values, minimum, maximum and population SD). The primary context and full-clip context remain separate. Latency has its own registered training-only runner and timing boundary; evaluation authentication/I/O is not model latency.

This is additional official-fold research on source-grouped recordings. Fold1 had already been observed and these caches participated in historical fold1 training/inner validation. It is not a wholly prospective four-fold result, verified unseen-golfer accuracy, upload-domain validation, minimum hardware claim or coaching validation. Normal models, pose/outcomes, serving and advice remain closed. Raw media, caches, checkpoints, probability arrays and large job ledgers stay ignored and out of Git.

## Verification

`tests/test_additional_evaluation.py` uses synthetic features, probabilities and checkpoints only. It covers bounded resume, all six selected heads, complete1050-item outer admission, exact saved PTS, official tie/tolerance/anomaly arithmetic, selected-versus-last checkpoint behavior, shared-lock exclusion, pre-forward crash refusal, probability/report orphan recovery, restored-byte/mtime and inode/symlink drift refusal, stat-bracketed initial hashes, final-publication rollback with outside-edit preservation, no-forward completed replay, and standalone CLI help/invalid bounds. Synthetic checks make no accuracy or trained-model claim. Independent review can read saved real probability arrays and replay official metrics without actual model forwards.
