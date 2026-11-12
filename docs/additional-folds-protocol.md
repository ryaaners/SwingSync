# Additional official GolfDB folds — fixed registration

`experiments/additional-folds-v1.json` registers folds2,3and4 as a separate extension of the completed fold1 comparison. This file contains no training or evaluation result and enables no app model, pose measurement, outcome prediction or coaching. The previously evaluated fold1 remains historical evidence; its outer results were already observed. Clips held out by the new folds previously occurred in historical fold1 training or inner validation. New heads must start from scratch and cannot reuse any fold1 head or smoke state. A future four-fold summary must identify that mixed registration history rather than claim a wholly prospective or unseen-golfer study.

## Fixed data and partition contract

The exact original MAT SHA, upstream revision, source-grouped split algorithm, split seed20261104 and validation-source fraction0.2 are preserved. Saved manifest bytes and regenerated complete clip/source assignments must both agree. Counts are measured from the actual manifests, not a frame-random split:

| New official outer fold | Train clips / sources | Inner validation clips / sources | Test clips / sources | Fixed training-only latency IDs |
| --- | --- | --- | --- | --- |
| 2 | 836 / 350 | 214 / 88 | 350 / 142 | 0,1 |
| 3 | 839 / 340 | 211 / 86 | 350 / 154 | 6,7 |
| 4 | 853 / 352 | 197 / 88 | 350 / 140 | 0,1 |

The outer folds and related source recordings stay intact. These are source-disjoint folds, not verified unseen-golfer splits. Internal source ranking does not inspect event timing to choose validation groups. Existing cached labels preserve class-specific annotation indices and the677 nonchronological anomaly. Re-encoded clip PTS are not original-source timestamps.

## Unchanged training and fixed evaluation

Both original JSON head configs remain byte-identical: seed20261105,20epochs,batch4,Adam learning rate0.001,CPU one numerical thread/deterministic algorithms,hidden64for the temporal head,full-clip training/inner context,no augmentation. Class weights use only the relevant fold's training labels. Each fold/model initializes its head anew with the same frozen pretrained ImageNet V2 MobileNetV2 backbone. No event-trained backbone, learned scaler or other fitted cross-fold transform can be reused. A train-only five-update smoke precedes full training and does not seed that full run.

Selection remains highest standard inner-validation PCE with earliest exact ties and the selected `best_model` state. The held-out comparison fixes both heads and both contexts before any new fitting: reference64-frame recurrent reset is primary; full-clip context is separately reported. Independent class peaks, NumPy default-argsort last-index ties and the existing official evaluator are preserved. No chronology repair, confidence calibration, test-driven reselection or repeated tuning is allowed.

The official PCE tolerance remains `max(np.round((chronological_target[5]-chronological_target[0])/30),1)` with inclusive absolute frame error. The target slots are chronological occupied-label frames, including the677 anomaly, rather than silently relabeled semantic events. See the pinned [official eval](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/eval.py) and [util](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/util.py); the frozen local evaluator is unchanged. Report overall/per-event PCE, frame and actual native-PTS millisecond errors per fold/model/context, plus explicitly scoped inference latency on recorded hardware. Never reconstruct milliseconds from FPS. The latency inputs above are training only, with16-frame warmup,batch16,one measured repetition and both contexts separately disclosed; these narrow samples cannot establish a hardware minimum.

New-fold aggregate results require all three complete folds with immutable inputs. Partial jobs are not final evidence. A four-fold summary is separate and must retain the previously completed fold1 results, matching contexts and registration-history disclosure. Weak phases, order failures and training-domain limitations must remain visible. No new accuracy exists merely because this contract or a cache audit passes.

## Read-only audit and future launch gates

The existing1400cached feature sequences can be considered only because their backbone and image preprocessing were frozen and not fitted to GolfDB events. Existing full catalog bytes, per-cache identity/arrays, exact source recording bytes, annotation labels, PTS provenance and pipeline must pass the separate read-only audit. Reading stored arrays for compatibility is not a new model forward or evaluation. Never warm-start the new heads from historical trained heads. Audit output is research compatibility evidence, not automatic training admission.

Before fitting, a future reviewed launcher must independently verify that the exact registration/code was committed and pushed, admit an independently reviewed completed audit using its known SHA, run the train-only smoke and preserve one shared expensive-work lock. Record dataset/catalog/config/environment/seed/process/job identity before launch; save resumable model/optimizer/RNG state and only resume identical work. Never restart duplicate expensive work or treat a launched process as completed training. Historical source files, root lock, registrations, caches and checkpoints must not be changed to make new work pass.

`backend.swingsync.research.fold_protocol.validate_protocol` is a pure-base validator for exact registered bytes, strict typed operational fields, original config/lock/frozen-source bindings and optionally actual dataset/manifests. It reads no checkpoint, probability, cache array or video. It does not inspect remote Git state, establish cache equivalence or launch/admit training. Registration alone therefore cannot open a serving or training gate. Cache auditing, runtime admission and the future launch driver remain separate reviewed work.

No new fitting/evaluation is performed by this registration increment. Additional raw-upload domain/crop/reliability evidence, independent pose references, original-source timing and coaching validity remain separate gates. CaddieSet equivalence/generalization admission remains closed.
