# Completed cache independent audit — 005

**Passed.** Actual job `184c02b4709746b79f15a8976399101f` completed all **1,050 selected clips** on 2026-11-06 at 03:57:10 UTC. Its final extraction attempt completed 1,039 new clips after the 11 previously persisted clips. The recorded PID 21311 was absent and the shared work lock was available at the audit start. This is completed frozen-feature preparation, not head training or model evaluation.

| Partition | Clips | Source recordings | Cached frames |
| --- | ---: | ---: | ---: |
| Train | 840 | 348 | 236,088 |
| Inner validation | 210 | 88 | 59,133 |
| Selected total | 1,050 | 436 | 295,221 |
| Outer test excluded | 350 | 144 | Not decoded or evaluated in this audit |

The MAT file was loaded again and the fold1 source-grouped manifest regenerated exactly with seed 20261104. All partitions have disjoint clip IDs and source recordings. The completed ledger and media inventory cover exactly train plus inner validation. The shared feature directory contains exactly 1,050 NPZ files, all selected; zero outer-test or extra cache files were found.

Every selected NPZ passed `load_feature_cache` pickle-free schema, semantic, finite-value and array-checksum validation. Independently verified each file's SHA256 against the ledger, cache key/path, video-byte SHA256 against the current source clip, immutable job inventory, previous full-video inventory and cache identity, unchanged MAT events and source identity, class labels, historical source indices, actual frame count, and exact equality of **every cached PTS** to a fresh sequential decode. Video and cache hashes were checked again after each clip. Arrays were discarded after each clip; no dataset-wide feature tensor was retained. Total measured audit time was 17.961 seconds in the isolated ML runtime.

All features are finite float32 `[T,1280]`. Training clips contain 70–1,109 frames; validation clips contain 87–1,276 frames. Every clip's first PTS is 0 seconds. Training last PTS spans 2.301150575–36.970303637 seconds; validation spans 2.869536203–42.542542543 seconds. Timestamps are strictly increasing actual **re-encoded clip PTS**, not inferred FPS times or original YouTube PTS. Historical source-frame indices remain a separate checked mapping. Original-source image/PTS equivalence remains unverified.

Train-only class counts are **840 each for classes 0–7** and **229,368 background frames**, summing to 236,088 training frames. Validation labels did not contribute to these counts. Official annotation identities, including any upstream ordering anomalies, were preserved without repair.

The actual local ImageNet V2 checkpoint's frozen feature-layer tensor bytes match the registered backbone hash `46cd17dc0b0bbf24bdf5f26a1f20a82f96cc2f5c863fa2fcd87856b5518971be`. The recorded pipeline matches `experiments/event-v1.json` exactly: deterministic resize232/center224, CPU, ImageNet normalization, Torch2.9.1/torchvision0.24.1, NumPy2.2.5, PyAV16.0.1, Python3.11.14, FFmpeg versions and local source/runtime-code fingerprints. No network download, extraction, backbone forward pass, head training, benchmark or outer-test evaluation was performed.

Job configuration, dependency lock, extractor module and runtime code, job implementation and combined job fingerprints were recomputed and match. Frame and temporal configuration byte hashes match their preregistration. The coordinator confirmed the current pipeline, training code, dependency lock and experiment files equal committed HEAD `45b96c0a81290f56ca242adf7196a7559fe28dd9`; this auditor independently checked the current bytes and performed no Git operations. The job's older `git_base_commit` remains its original creation-time provenance; exact implementation hashes bind the extraction code. Source/config/dependency fingerprints remained unchanged at audit completion.

The frozen admission catalog `artifacts/features/fold1-job-v1/completed-catalog-v1.json` is byte-identical to the audited completed `job.json`. Both have SHA256 `daf5f108512fc8468476722e7a751e1d95fe64e51e045a9bc95ab65df9918b9e`. Registered head training may proceed against this frozen catalog, subject to the existing shared lock and complete-cache admission checks.

Machine-readable evidence: [cache-completion-005.json](cache-completion-005.json) contains fingerprints, exact per-partition timestamps/frame totals and train-only counts. The full 1,050-row witness is kept outside Git at artifacts/features/fold1-job-v1/cache-completion-005-full-audit.json, fingerprinted in the summary. The sorted compact-JSON row transcript digest is `8d4a34f3cbcdc4cbdf26bfa766eb99751007a0ca1f5404a8e0c6eb809b2918d5`. **No findings requiring correction.** This audit establishes preparation readiness only; no project-trained checkpoint, generalization, serving, browser synchronization or coaching reliability is established.
