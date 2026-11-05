"""Report actual runtime capabilities; no benchmark results are inferred."""
import json
import platform
import shutil

import av

result = {
    "python": platform.python_version(), "platform": platform.platform(),
    "machine": platform.machine(), "ffmpeg_cli": shutil.which("ffmpeg"),
    "pyav": av.__version__,
    "ffmpeg_libraries": {name: list(version) for name, version in av.library_versions.items()},
    "training_job": None,
}
try:
    import torch
    result["torch"] = torch.__version__
    result["cuda"] = torch.cuda.is_available()
    result["mps"] = torch.backends.mps.is_available()
    result["selected_device"] = "cuda" if result["cuda"] else "mps" if result["mps"] else "cpu"
except ImportError:
    result["torch"] = "not installed; ML increment pending"
    result["selected_device"] = "cpu (video inspection only; accelerators not probed)"
print(json.dumps(result, indent=2))
