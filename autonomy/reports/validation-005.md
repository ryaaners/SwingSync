# Validation 005 — completed preparation, real head training, descriptive arithmetic

Date: 2026-11-06. This increment completes backlog 003c/003d/004a; parent event evaluation and pose/coaching integration remain open.

## Software checks

- `uv sync --locked --offline` in base environment passed; `uv run pytest` passed 107 tests with 7 explicit optional-ML-module skips.
- `uv sync --locked --extra ml --offline` passed; `uv run --extra ml pytest`: 161 passed (one existing Starlette/AnyIO deprecation warning).
- Ruff `backend tests scripts`: all checks passed.
- Frontend `npm ci --ignore-scripts` and `npm run build`: passed clean install, TypeScript and Vite bundle; npm audit reported 0 known vulnerabilities. No frontend code changed.
- 51 new synthetic pose-contract/measurement tests cover variable actual PTS, manual correction provenance, aspect scaling, known angles, body normalization, handedness/mirror conventions and unavailable-input suppression. Synthetic geometry proves software mechanics only.

The isolated training environment was not mutated by base/ML application checks. Root dependency lock and registered ML source/config bytes remained unchanged.

## Independent evidence and review

Data specialist audited all 1,050 cache/video pairs against actual MAT/source manifest and fresh sequential PTS decode. 295,221 frames; 840 train / 210 inner; source groups disjoint; zero outer cache. See `cache-completion-005.json/.md`. Detailed witness stays ignored and hash-bound.

Both actual registered training runs completed 20 epochs / 4,200 updates sequentially, with 168 checkpoint hashes/head independently verified. Reviewer validated immutable provenance, train-only counts, finite model/Adam/RNG state and earliest-maximum selection; read-only replay of selected weights reproduced every selected inner-summary field exactly (temporal weighted loss too). No outer clip was loaded in training/selection/replay. See `review-005.md`, `frame-training-005.json` and `temporal-training-005.json`. No unresolved material finding.

Selected INNER PCE 41.96% frame (epoch 19), 64.58% temporal (epoch 20). These are model-selection scores, not held-out accuracy. Full-clip context differs from reference 64-frame resets. Backbone is pretrained ImageNet initialization; event heads are genuinely project-trained. Training attempt elapsed times are not inference latency. Weights/caches/data remain ignored.

Movement implementation independently reviewed against official MediaPipe indices/coordinate definitions. Every result retains unvalidated reliability and no advice. No local pose inference, empirical geometry/coaching validation or outcome integration is asserted.

## Remaining gates and synchronization

Next: predeclare/implement separate finalized held-out evaluation with standard reference recurrence and disclosed full-clip comparison; measure actual CPU inference latency. App remains model unavailable. Source-original/browser synchronization, actual pose validation, permitted real video coaching demonstration and supervised feature equivalence/generalization stay open.

Coordinator inspects intended diff, fetches/checks divergence, commits/pushes to dedicated `agent/swingsync` without force and verifies remote SHA. Exact checkpoint and current CI outcome are reported in chat and must be inspected on next execution; no datasets/media/checkpoints enter Git.
