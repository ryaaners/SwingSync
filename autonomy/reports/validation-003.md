# Increment 003a validation — 2026-11-05

## Working increment

Implemented shared frozen torchvision MobileNetV2 with explicit pretrained/random provenance, frame and packed bidirectional LSTM heads, device detection and padding-safe weighted loss. Added bounded full-clip frozen feature extraction, provenance-keyed pickle-free checksummed atomic caches, CPU checkpoint/model/Adam/RNG restore, and tracked duplicate-protected smoke CLI. No user-facing inference or coaching was enabled.

## Measured smoke (not benchmark)

Run smoke-003-20261105 completed on actual fold 1 training clips 0/1:462 frames,1280 features/frame. Five logical optimizer updates per head, seed 20261105, CPU, one thread, Adam0.001. Additional three-step replay per head verified checkpoint continuation. No test/validation split evaluation. Frame training loss2.218995→1.949782; temporal2.212210→1.774242. These losses do not establish accuracy or model superiority.

Apple M1/8GiB; CUDA unavailable, MPS detected available but not used. Total37.516 seconds, extraction/cache preparation36.014 seconds. This is observed smoke runtime, not inference latency or a minimum hardware specification. Official ImageNet V2 initialization downloaded locally; no weights redistributed or represented as project event training. Record contains run process identity/config/data/split/code/lock hashes before work. Four checkpoint hashes and exact CPU resume verification are in smoke-003.json. Large artifacts are ignored under artifacts/training and artifacts/features; reference image weights under data/torch.

## Checks

- `uv sync --locked --offline` and base `uv run --locked --offline pytest`:56 passed,4 explicit optional-ML module skips.
- `uv sync --locked --extra ml --offline` and `uv run --locked --extra ml --offline pytest`:82 passed; one pre-existing Starlette/AnyIO deprecation warning.
- `uv run --extra ml ruff check backend tests scripts`:passed.
- Frontend npm ci --ignore-scripts and production/typecheck build:passed; npm audit zero known vulnerabilities.
- Independent fresh reviewer verified all82 tests, cache stale-implementation regression, actual run completion, train-only clip/source membership, cache arrays/hashes, all4checkpoint hashes and model/optimizer/RNG state without launching duplicate work. No unresolved material finding. See review-003.md.
- Previous 6df193c GitHub workflow success observed. New workflow adds a CPU ML-contract job with synthetic offline fixtures only; no dataset, pretrained download or full retraining required. Current checkpoint remote CI outcome is reported in chat after push.

## Scope and remaining work

Parent 003 remains in progress. Full feature preparation, sequence sampling/augmentation if enabled, full real training/model selection, standard outer-test PCE/errors and actual inference latency remain undone. Original-source/browser alignment and pose/coaching/outcome gates remain intact. Training loss reduction and software tests are not generalization evidence. A completed five-step smoke checkpoint is not a validated model for user videos.

## Documentation consulted

[Official paired Torch2.9.1/torchvision0.24.1 releases](https://pytorch.org/get-started/previous-versions/), [MobileNetV2 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html), [packed sequences](https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.rnn.pack_padded_sequence.html), [checkpoint serialization](https://docs.pytorch.org/docs/2.9/notes/serialization.html), [uv PyTorch indexes](https://docs.astral.sh/uv/guides/integration/pytorch/), [torchvision BSD3 code license](https://github.com/pytorch/vision/blob/v0.24.1/LICENSE). Source/data/weights/original-project licenses remain separate.
