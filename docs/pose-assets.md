# Isolated pose environment and assets

Execution 009 resolved and installed the separate Python 3.11 environment on macOS 26.3 ARM64. Native imports of MediaPipe, NumPy, OpenCV and PyAV and an OpenCV RGB conversion passed. This establishes local binary compatibility, not landmark accuracy, measurement reliability, inference speed or a minimum hardware specification. Actual extraction evidence belongs in the separate execution report.

`pose_runtime/` is an independent uv project, with `package = false`, its own complete committed lock and no root workspace membership. It neither installs SwingSync as a package nor changes root `uv.lock`. The root lock still hashes to `8ce1d6674a07bf122541f555b90ad81532029b7ce343af54a78cd057fc2b26fc`; frozen event-research artifacts remain bound to that original environment.

From the repository root, install explicitly:

```sh
UV_CACHE_DIR="$PWD/.tools/uv-cache" \
UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" \
UV_PROJECT_ENVIRONMENT="$PWD/.tools/pose-runtime" \
.tools/uv-env/bin/uv sync --project pose_runtime --locked --python 3.11
```

The direct pins are MediaPipe 1.2.2, NumPy 2.2.5, OpenCV contrib 4.12.0.88 and PyAV 16.0.1. The full lock includes all transitives; do not replace it with an unconstrained `pip install`. Only `opencv-contrib-python` owns `cv2` in this environment. The effective macOS ARM wheel floor is macOS 14 because of PyAV's CPython 3.11 wheel. Linux x86_64 is lock-resolved but requires its own native smoke; aarch64 and Windows are not claimed as tested platforms. Linux OpenCV may require system `libGL`/GLib libraries even for CPU imports.

Set `MPLCONFIGDIR` to a writable ignored directory and `MPLBACKEND=Agg` for offline work. MediaPipe imports Matplotlib; its first import built a font cache on this machine. That initialization is not a measured pose latency result. The 20 installed package versions, full lock hash, FFmpeg library versions, native binary hash and 14 upstream license/notice file identities are recorded in [pose-environment-009.json](../autonomy/reports/pose-environment-009.json).

## Explicit model acquisition

Run this separately while network access is available, before offline inference:

```sh
python3 scripts/acquire_pose_asset.py --directory artifacts/pose
```

The helper uses only Python's standard library. It downloads one fixed official URL, verifies exact size/SHA256 and both member hashes, and refuses to replace changed existing assets or manifests. Downloads and complete manifests are published without overwriting existing files; interrupted temporary downloads are removed. Inference must not download or silently substitute a model. The task bundle and raw audit evidence stay in ignored `artifacts/pose/`, never ordinary Git.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| `pose_landmarker_full-v1.task` | 9,398,198 | `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1` |
| `pose_detector.tflite` | 2,959,046 | `04f5a96483f8ce5913f8730ad2aa0c4c999c80a4b75653bc0e4712de18e104b2` |
| `pose_landmarks_detector.tflite` | 6,438,874 | `82be6d591b9dad7d29fe21dc9fd892bf8b9602c458fb05209283de8282a0c488` |

Source: [official full / float16 / version 1 download](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task), linked from the [official task model table](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker?tab=Python). These are acquired pretrained upstream weights, not models trained by this project. The ignored manifest binds acquisition time, fixed URL, bundle/member hashes and separate license evidence to extraction artifacts. No moving `latest` identity is admitted.

## Licenses and privacy

Code, packaged third-party dependencies and model rights have separate evidence:

- [Pinned MediaPipe code LICENSE](https://github.com/google-ai-edge/mediapipe/blob/2c2d4a7454d6358e0d6f0b54e254d231fe4cc78f/LICENSE) is Apache 2.0 with additional per-file terms. The installed wheel retains `licenses/LICENSE` (SHA256 `8707eef0533987efc5b155d64761eeb6e20793f50b9bd1a68dad1cf4719d0ed8`) and `licenses/NOTICE` (`78eed891b4676ead09d8bf72bbab6b1fc8404475ed68da2cdd9f7d5e8dad22c5`). The latter includes third-party notices. Do not strip these from a packaged runtime or apply their license automatically to original SwingSync code.
- The separately fetched [BlazePose GHUM 3D model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf), SHA256 `e982460cedece5ad9a46aa8f13523328e9ab95d7d5b2b8c99b33f13f6a79ca86`, identifies Apache License Version 2.0 on page 2. Its described landmark family is not an exact byte-level license declaration for every detector/landmarker member.
- Both exact TFLite members contain metadata with author `MediaPipe` and version `1`; their metadata license fields are absent. Their associated-file ZIP inventories are empty. No separate embedded license/notices were found. **Model redistribution remains disabled** until exact-member attribution is established; acquiring from the official distribution for local research does not establish redistribution rights or transfer rights in the separate GHUM research assets.

The 1.2.2 wheel's NOTICE dated June 5, 2026 states input image/video/text processing stays on device, and also states that performance/utilization metrics are sent to Google. The [release page reproduces that notice](https://pypi.org/project/mediapipe/1.2.2/). No supported Python opt-out was found in the pinned `BaseOptions` fields. Its CA-bundle option is for usage logging; omitting or corrupting that path is not a verified privacy control.

The public source logging factory returns a dummy logger, but the installed native wheel exports a Google-internal statistics logger and contains a logging endpoint. Public-source behavior therefore does not prove that the release binary has disabled telemetry. No packet-capture, payload characterization, or claim of image transmission is made. Native extraction must run under independently verified **OS-level network denial** before import and task initialization. An environment variable or Python socket monkeypatch is insufficient for native code. Product pose serving remains disabled; the extraction documentation describes the current supported offline invocation.

Original golf media has separate source/usage restrictions. The research archive's reencoded PTS are not original recording PTS. Keep source rights and timestamp basis explicit in each extraction; pose model licensing does not authorize video redistribution. No upstream model card score or generic detection threshold validates golf coaching.
