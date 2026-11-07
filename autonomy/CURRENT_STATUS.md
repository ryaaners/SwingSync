# Current status — 2026-11-07

Execution 007 completed manual event review (004b) and the upstream local-pose runtime audit (004c) while the existing test-feature job continued. Automatic inference, pose extraction, outcomes and coaching remain unavailable. Parent003/004 and real-video coaching demonstration005 remain incomplete. Synthetic UI/report evidence is software validation only.

## ACTIVE JOB — inspect before expensive work

Same test-only feature job **833e9ffe5592457f9257a46474c72153**, actual worker **PID 40807**, start identity `Fri Nov  6 12:38:35 2026` (Toronto). Ledger `artifacts/features/fold1-test-job-v1/job.json`: observed **running 310/350** at `2026-11-07T09:02:56.408305+00:00`. Read actual state, command/start and shared `artifacts/training/active-training.lock` on resume. Do not duplicate extraction or start training/evaluation/pose/timing while that lock is held. This execution did not launch/resume/stop that worker or change its isolated `.tools/ml-runtime`.

Launch/log: `artifacts/features/fold1-test-job-v1/launch-full-006.json` and `full-preparation-006.log`. Test cache directory `artifacts/features/fold1-test-v1` stays separate from training. A partial/running inventory is not completed features or model evaluation. No outer-test prediction/run exists. The computer must be awake/available for this real OS process; a scheduled chat is not a persistent agent.

Prospective held-out contract committed641fe51 before test access; `experiments/heldout-v1.json` SHA256 `4b4339a469dd5fe09f97b57ae4b650ce378ce1c3a3863f86b46249ad8d9d0008` unchanged. Root `uv.lock`, all research/training/feature implementation and prior checkpoints/configs stayed unchanged.

## Newly working manual review

New uploads fingerprint video bytes. UI/API let users mark any subset of eight events using canonical decoded frames, save/correct/reload markers and recompute three recording-time tempo quantities. Source/native PTS and relative PTS remain distinct; no FPS/browser-time guessing. PNGs verify media hash and full decoded timestamps in bounded subprocesses, preserve coded orientation and aspect within640px, and use at most two decoder slots. Older uploads without hashes need reupload for annotation/frame evidence; ordinary history remains.

SQLite revision CAS prevents stale overwrite and deleted-row resurrection. Dirty drafts survive polling and receive explicit conflicts; reload fetches current state. Exports contain current manual event images, evidence/limitations, all current/superseded corrections including clear-all, analysisID and mediaSHA. Manual reliability is unvalidated. No pose observations, model events, outcomes or advice are fabricated. Browser preview seeking/orientation synchronization remains unverified and explicitly labeled; use the canonical frame panel.

Browser QA used synthetic9frame VFR H.264 with native start5s: Address0/Top3/Impact8 yielded0.180/0.920s; corrected Top5 yielded0.410/0.690s, ratio0.594203. Reload/restart persisted revision4/history; concurrent API edit preserved the unsaved browser draft and disabled save until explicit reload. Independent reviewer matched all3 downloaded PNGs exactly to sourceframes0/5/8. Original browser download preserved as `artifacts/qa-007/browser-manual-report.html`; final API-exported `manual-report.html` adds recordingidentity. Screenshots and media remain ignored. Summary `autonomy/reports/manual-review-007.json`; guides `docs/manual-event-review.md` and `review-007.md`.

## Pose runtime audit — no installed adapter

Exact official wheel metadata/source/model-card/HEAD evidence and licenses are audited. Proposed isolated candidate MediaPipe1.2.2 /NumPy2.2.5 /OpenCV4.12 /PyAV16 is **metadata-compatible only**, full binary MacARM minimum14. Legacy0.10.21 conflicts with root NumPy2.2.5. No package environment/model bundle was acquired or inferred. Next pose work needs separate non-workspace pyproject/committedlock, import/native smoke, pinned model-bundle bytes/member/license manifest, integer-ms API clock preserving actualPTS (reject collisions), then actual permitted-media validation. See pose-runtime-audit.md /pose-audit-007.json. Advice and CaddieSet equivalence stay closed.

## Checks and review

Locked base128passed/9explicit optional ML skips; locked ML **223passed**; Ruff backend/tests/scripts clean. Frontend clean npmci,4Node tests,TypeScript/Vite build pass;0known vulnerabilities. Three P2 findings resolved and independently retested:clear-all correction history omitted from export, nonfinite request error serialization, and pose candidate Macminimum mismatch. Fresh independent review found no unresolved material finding and checked screenshot/report/frame evidence. QA servers stopped; user data untouched. See validation-007.md.

## Completed training/selection — preserve, do not retune

Frame/temporal heads already completed20epochs/4,200updates. Final frame checkpoint SHA256 `8203c5f5d2d84d2fb8519dffa5d053932cfe353069f145d54b17ff28ce5a2431`, selectedepoch19; temporal finalSHA `910f6d06a253e8117e825f5d625a1dcc11fe6db61344d169daac3fd973ceb5c3`, selectedepoch20. Use **best_model**, not final-epoch **model**. Ledgers `artifacts/training/{frame,temporal}-v1/run.json`; old training workers ended. INNER selection PCE41.96%/64.58% is not held-out performance.

Immutable training catalog `artifacts/features/fold1-job-v1/completed-catalog-v1.json` SHA256 `daf5f108512fc8468476722e7a751e1d95fe64e51e045a9bc95ab65df9918b9e`:840train/210inner,295,221frames, independently audited. CPU timing final artifact `artifacts/latency/fold1-train-v1-final.json` SHA256 `47cd4c4851f6cf6ddd8c464b37bd485102988bed656392a951d5a6c0cc24d72c`:twoTRAINclips/462frames,58.154366ms/frame shared decode/transform/backbone onM1/8GiB/CPU1thread, startup/warmup/head time separately scoped. No user-latency/minimum-hardware/fourfold/accuracy/coaching claim.

## Next increment and gates

When the active inventory truly finishes350clips, independently audit/freeze it, then run the unchanged fixed evaluator from README for both selected heads and both predeclared contexts. No context/model/test-driven tuning. Explicit interrupted-run resume revalidates durable rows and never publishes partial aggregates. Independently verify complete overall/per-event official PCE, frame/actualPTS-ms errors, annotation677 handling and provenance before reporting results. Never equate feature completion with evaluation completion. Additional outer folds need separate preregistration.

While that job is active, a separately locked pose environment/adapter software can be implemented without inference or root changes. Remaining gates: empirical pose/event reliability, original-source image/PTS correspondence (authorized originals), browser synchronization, permitted real-golf video-to-report demonstration, CaddieSet grouped tabular research and paired feature equivalence/generalization, cross-platform artifact reproduction. No ML accuracy/coaching résumé claim is supported. Do not repeatedly poll documented blockers.

## Git and scheduling

Only coordinator uses Git. Dedicated `https://github.com/ryaaners/SwingSync`, `agent/swingsync`; no force or automatic merge. Prior verified checkpointed36bfc CI42264778315 passed. Current checkpoint/remote SHA/newCI outcome reported in chat; inspect actual HEAD/remote/CI next run. Data/media/caches/checkpoints/jobs remain ignored. Daily user-authorized1:30a.m.Toronto schedule remains active; one-time task removed earlier. No new schedule or continuous agent was created.
