# Current status — 2026-11-08

Execution 008 completed the registered fold-1 event research comparison (003f/parent003) and an offline reproducible results packager. Actual test results were independently replayed exactly. The app remains manual-review only for swing events; automatic proposals, pose extraction, outcomes and advice remain disabled. Full product/release goals are not complete.

## Jobs — no expensive worker active

Feature job `833e9ffe5592457f9257a46474c72153` genuinely completed350/350,144 sources,98,956 frames. OldPID40807 ended. Both20epoch/4200update head training runs are completed. First held-out evaluation PID49176 also ended; shared lock available. Inspect actual processes/ledgers and `artifacts/training/active-training.lock` on every execution; do not restart or duplicate completed work.

Frozen test catalog `artifacts/features/fold1-test-job-v1/completed-catalog-v1.json`, SHA256 `159ec421d82ca82d66cfe070d4272e013c65604365cf2c31fc11a57af0d66cf4`, is byte-identical to completedjob.json. Original train catalog `artifacts/features/fold1-job-v1/completed-catalog-v1.json` SHA256 `daf5f108512fc8468476722e7a751e1d95fe64e51e045a9bc95ab65df9918b9e` remains840train/210inner/295,221frames withzeroouterrows. Separate directories remain ignored. Full350cache/media/actualPTS/labels/source/registeredpipeline audit and4rejectioncontrols passed; reports test-cache-audit-008.json/.md, ignoredfullwitnessSHA09ca6f779dbb45a96d9e49d2b90147aab677a56088d4cbffa51bab6bdb9569e9.

## Fixed actual held-out result — preserve, do not tune

Prospective contract `experiments/heldout-v1.json` was committed641fe51 beforetestprocessing; SHA256 `4b4339a469dd5fe09f97b57ae4b650ce378ce1c3a3863f86b46249ad8d9d0008` unchanged. Source-grouped fold 1:train840/348sources,inner210/88,test350/144. Seed20261105,split seed 20261104; ImageNet V2 backbone frozen, event heads genuinely trained 20 epochs, CPU 1 thread, noaugmentation/fullclip training+innerselection. Frame selected epoch 19 finalcheckpointSHA8203c5f5d2d84d2fb8519dffa5d053932cfe353069f145d54b17ff28ce5a2431; temporal selected epoch 20 finalSHA910f6d06a253e8117e825f5d625a1dcc11fe6db61344d169daac3fd973ceb5c3. Use **best_model**, not final-epoch **model**. Do not alter checkpoints, registrations, root lock or old research source without an explicit new prospective experiment.

Primary reference 64-frame-reset **PCE 45.11% frame / 58.79% temporal**; MAE22.224frames/741.489recording-ms versus11.700frames/390.368ms. Separate predeclared full-clip temporal **64.43%**,6.212frames/207.253ms. Fullclip is not a replacement for reference 64-frame; no order-constrained decoder, context/epoch reselection, fitting or test-driven tuning occurred. No test annotation anomaly exists;677 is TRAIN/fold3, original inversion preserved.

Raw `artifacts/evaluation/fold1-v1/run.json` SHA256 `93d9aba2332fd90fdb7cf77891a580550e42a38c68c2b10c0e774fa754c6bdb0` contains one actually completed350-clip attempt. Its19.520488 s is post-admission evaluation-loop elapsed, not video inference latency. Explicit completed --resume returned unchanged bytes/oneattempt without repeating durable clip inference. Independent new model objects/direct selected weights/separate batching+officialutil transcription reproduced all 1,400result dictionaries and every aggregate/per-event field exactly; replay witnesses under ignored artifacts/evaluation/fold1-v1, hashes in review-008.md.

Machine aggregate `autonomy/reports/heldout-results-008.json` SHA256 `32330db0fbac8b687f7264019ee322f7a58ba67f8864263b4e40fe9eb9b1ffc3` exactly reproduces with the new stdlib-only report CLI from a known-SHA completed run. It checks coverage/rows/arithmetic/aggregate/provenance without model/cache/media imports, refuses partial/tampered inputs and overwrite. Post-hoc saved-peak order diagnostics do not alter metrics:strict order111/350frame,279/350temporal64,321/350 full-clip. Label them descriptive, not calibrated reliability or decoder improvement.

## Limits and existing useful app

Reference64 temporal Address/Finish PCE 13.43% / 17.14% is weak. One official source-disjoint fold is not four-fold/unseen-golfer performance, calibrated event confidence, uploaded-video generalization or coaching efficacy. Golfer identities can repeat across sources. Do not use these test scores to tune another variant or declare normal app model availability. Research-only portfolio statements now have evidence; scoped examples in docs/portfolio-evidence.md. No end-to-end coaching or user-testing claim.

The manual event app remains working:canonical decoded SHA/PTS PNG evidence, exact frame-index marks, revision CAS/dirty-draft conflicts, recomputed manual recording-time tempo and self-contained reports with correction history. Synthetic browser QA in007 proves software only. Browser/audio/time-origin/display-rotation correspondence remains unverified; framepanel is evidence. Pure pose-measurement arithmetic exists but no actualpose adapter/inference, empirical accuracy or advice.

## Next highest-priority increment

Proceed with004d:isolated non-workspace Python3.11pose subproject/full committeduvlock, candidateMediaPipe 1.2.2/NumPy 2.2.5/OpenCV 4.12/PyAV 16, effective Mac ARMfloor 14; import/native smoke. RootMLlock stays immutable. Acquire/version/hash/audit exact official model bundle/member licenses into ignoredstorage, implement actual PTS versus integer-ms API clock provenance/collision rejection and failclosed missing/multiple/identity/view/mirror handling. Run bounded actual CPU extraction only with the expensive-work lock and record identity/artifacts. Distinguish actual inference from empirical landmark/measurement validation and keep advice suppressed. Source/package/model evidence is already in pose-runtime-audit.md/.json; no install/model acquired yet.

Uploaded event-serving input/reliability admission003g remains pending separately. Original-source image/PTS equivalence needs authorized originals; browser synchronization, empirical pose/event accuracy, permitted real-golf video-to-report demo, CaddieSet grouped research/pairedfeatureequivalence and cross-platform artifact reproduction remain gates. Do not repeatedly poll known blockers. More outer folds require new prospective registration; they are not required to relabel this completed first comparison as four-fold.

## Checks, Git and scheduling

Locked base191tests/9explicit optionalMLskips, fullML286tests, Ruffbackend/tests/scripts, frontendcleaninstall4tests/typecheck/build all pass. One existing AnyIOdeprecationwarning;0known npm vulnerabilities. Newpackager63tests pass and imports noTorch/NumPy/AV. Fresh independent cache/protocol/artifact/packager/document review has no unresolved materialfinding. No rootdependency/oldmodelsource mutation, no usermedia/checkpoint/data committed.

Dedicated `https://github.com/ryaaners/SwingSync`, `agent/swingsync`; only coordinator usesGit, noforce/automaticmerge. Prior db330bf CI42358270651 passed. Current checkpoint/remoteSHA/CI reported in chat; inspect actualHEAD/remote/CI next execution. Daily user-authorized1:30a.m.Toronto continuation stays active; no persistent agent or new schedule was created. All old workers are ended, not waiting for an automatic chained evaluation.
