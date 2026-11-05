# Independent review — increment 002

Reviewed 2026-11-04. Reviewer ownership: this report only; no implementation edits or Git operations. Scope: GolfDB loader/manifests, official evaluation, research timeline mapping, preparation CLI, tests, and evidence boundaries. Read AGENTS.md, CURRENT_STATUS.md, backlog.json, reviewer role, and docs/data-contracts.md.

## Resolved finding; final disposition

No unresolved implementation findings in reviewed scope. P2 resolved — reject complex timestamps before conversion in `backend/swingsync/research/evaluation.py` (`evaluate_clip`, timestamp conversion block). The initial implementation executed `times[:count].astype(np.float64)` before `np.isrealobj(times)`, discarded imaginary components, and made the subsequent real-valued check ineffective. Reproduction: `evaluate_clip(np.full((8, 9), 1/9), np.arange(8), timestamps_seconds=np.arange(8) + 100j)` initially emitted ComplexWarning and returned millisecond errors instead of rejecting invalid presentation timestamps. The coordinator moved real-valued validation before conversion and added three rejection regression cases. Reviewer reran the original reproduction: it now raises ValueError before conversion. This concerns malformed research input; PyAV's decoded PTS path supplies real numbers.

## Verified independently

- Initial targeted new tests: 43 passed. Final full suite after correction and CLI regressions: 55 passed, one existing Starlette/AnyIO deprecation warning. Ruff: all checks passed. Executed with project `.venv/bin/python`; plain `uv` is absent from reviewer shell PATH, so this is not independent locked-install evidence.
- Compared the pinned upstream `util.py` and `data/preprocess_videos.py` directly with implementation. Official `correct_preds` uses chronologically occupied label frames, default NumPy argsort ties, inclusive tolerance, and NumPy rounding. Historical source preprocessing increments before checking inclusive trim boundaries. Implementation preserves these conventions.
- Loaded actual ignored MAT artifact: SHA256 `36523fcef016d93334691ea058d643bcf0d3a4a8c70b733f3b4cc0b8e5e0cf0a`, 1,400 records, 580 source IDs. All four manifests cover all rows exactly once and have disjoint source groups across train/internal-validation/test. Official test partitions each contain 350 rows. No frame-based split exists in this increment.
- Real re-encoded clips 0 and 677 decode to 138 and 278 frames, matching annotation trim counts. Their historical source-index ranges are 407–544 and 793–1070. Annotation 677 retains semantic Impact/MFT indices 168/167; evaluation separately records chronological benchmark targets and an anomaly flag. Official tolerance is two frames for this annotation, rather than the three-frame semantic Impact gap.
- New CLI tests verify all four manifests, deterministic summaries and hashes, exact JSON return roundtrip, overwrite refusal, archive inventory rejection and no output directory for invalid inputs.
- Preparation CLI completed on actual annotations plus clips 0 and 677, writing a fresh reviewer artifact directory under `/private/tmp/swingsync-review-002-prep`. Default seed 20261104 yields train/validation/test clip counts 840/210/350, 836/214/350, 839/211/350, 853/197/350 for folds 1–4. Output records hashes, environment versions, group manifests and honest evidence scope.
- Source-group ranking uses source IDs plus seed, not event positions or test labels. Outer test data are not used for inner validation selection. Tuple-backed frozen records avoid annotation mutation. Output files refuse overwrite.

## Evidence limits

These checks establish annotation partitions, benchmark software behavior, exact frame-count checks and decoded re-encoded-clip PTS mechanics. They do not establish original-source image alignment, variable timestamp preservation through upstream re-encoding, browser/container synchronization, trained model accuracy, unseen-golfer generalization, or advice reliability. Both source-image and browser-alignment fields remain false in preparation output. No checkpoint, event inference, training result or outcome integration was introduced. Synthetic score/video fixtures are software tests only.

Process listing was denied by the execution environment (`ps: operation not permitted`); reviewer did not independently certify running-job state. Coordinator owns full required checks, artifact provenance, final status/backlog update, complete diff review, and Git checkpoint verification.

Upstream sources: [official correct_preds](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/util.py), [official preprocessing](https://github.com/wmcnally/golfdb/blob/a63b4ce8c0900d09abbe92519223efe16810f5f4/data/preprocess_videos.py).

## Delta review — full inventory option

Reviewed `--all-clips` addition, its exclusive-selection/missing-directory regression, and the corrected official-evaluation contract. The flag processes all loaded annotation IDs through the existing exact-count/PTS path, validates media before reserving output, and retains explicit re-encoded provenance. No new implementation findings. Final delta checks: 49 research/preparation tests passed; Ruff including `scripts/golfdb_prepare.py` passed.

Independently inspected coordinator-produced `artifacts/golfdb/inventory-002.json`: 1,400 rows, 1,400 passed, zero failures, matching annotation fingerprint, original-source alignment false. Also checked every row of `artifacts/golfdb/preparation-002-all/summary.json`: exactly 1,400 unique annotation IDs, exact expected frame counts, recorded PTS finite and strictly increasing, correct historical source indices, unchanged class-specific event indices, and source/browser alignment flags false. This is an independent audit of recorded all-clip evidence, not a second independent full-video decode. Reviewer directly decoded clips 0 and 677 earlier.

Documentation follow-up sent to coordinator: the historical source-index formula needs the implemented zero-boundary special case `max(1, trim_start) - 1`, and data contracts can distinguish all-clip count/PTS integrity from the two clips' visual sanity scope. Neither observation changes the preserved source-equivalence or browser gates.
