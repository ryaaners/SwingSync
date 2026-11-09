#!/usr/bin/env python3
"""Acquire one explicitly pinned upstream model; never download during inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile


MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
    'pose_landmarker_full/float16/1/pose_landmarker_full.task'
)
MODEL_SHA256 = '5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1'
MODEL_SIZE = 9398198
MEMBERS = [
    {'name': 'pose_detector.tflite', 'size_bytes': 2959046,
     'sha256': '04f5a96483f8ce5913f8730ad2aa0c4c999c80a4b75653bc0e4712de18e104b2'},
    {'name': 'pose_landmarks_detector.tflite', 'size_bytes': 6438874,
     'sha256': '82be6d591b9dad7d29fe21dc9fd892bf8b9602c458fb05209283de8282a0c488'},
]
LICENSE_EVIDENCE = {
    'code': {
        'license': 'Apache-2.0 with third-party notices; not a license for original SwingSync code',
        'source_revision': '2c2d4a7454d6358e0d6f0b54e254d231fe4cc78f',
        'source_url': 'https://github.com/google-ai-edge/mediapipe/blob/'
                      '2c2d4a7454d6358e0d6f0b54e254d231fe4cc78f/LICENSE',
        'installed_wheel_license_sha256': '8707eef0533987efc5b155d64761eeb6e20793f50b9bd1a68dad1cf4719d0ed8',
        'installed_wheel_notice_sha256': '78eed891b4676ead09d8bf72bbab6b1fc8404475ed68da2cdd9f7d5e8dad22c5',
        'notice_scope': 'MediaPipe 1.2.2 macOS ARM wheel; other-platform notice bytes require inspection',
    },
    'model_card': {
        'url': 'https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf',
        'sha256': 'e982460cedece5ad9a46aa8f13523328e9ab95d7d5b2b8c99b33f13f6a79ca86',
        'license_statement': 'Apache License Version 2.0, page 2',
        'scope': 'BlazePose GHUM landmark model family; card does not bind exact bundle/member hashes',
    },
    'members': [
        {'name': m['name'], 'metadata_author': 'MediaPipe', 'metadata_version': '1',
         'metadata_license': None, 'associated_files': [],
         'redistribution': 'Not enabled: exact-member license attribution remains unestablished'}
        for m in MEMBERS
    ],
    'usage': 'Explicit local research acquisition from official task download; no model redistribution',
    'privacy': {
        'notice_date': '2026-06-05',
        'input_processing': 'Upstream NOTICE states image/video/text inputs stay on device',
        'metrics': 'Upstream NOTICE states performance/utilization metrics are sent to Google',
        'policy': 'Run native extraction only within verified OS-level network denial; no Python opt-out verified',
    },
}


def inspect_model(path: Path) -> None:
    data = path.read_bytes()
    if len(data) != MODEL_SIZE or hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError('Model size/SHA256 differs from the pinned reviewed asset')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != [m['name'] for m in MEMBERS]:
            raise ValueError('Model member inventory differs from the pinned asset')
        for expected in MEMBERS:
            member = archive.read(expected['name'])
            if (len(member) != expected['size_bytes']
                    or hashlib.sha256(member).hexdigest() != expected['sha256']):
                raise ValueError('Model member size/SHA256 differs from the pinned asset')


def acquire(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / 'pose_landmarker_full-v1.task'
    manifest_path = directory / 'pose_landmarker_full-v1.manifest.json'
    if not model.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.partial', delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                    # A strict cap also detects a changed URL/body before publication.
                    data = response.read(MODEL_SIZE + 1)
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            inspect_model(temporary)
            os.link(temporary, model)  # no overwrite if another acquisition won
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    inspect_model(model)
    manifest = {
        'schema_version': 1, 'url': MODEL_URL, 'sha256': MODEL_SHA256,
        'size_bytes': MODEL_SIZE, 'members': MEMBERS, 'license_evidence': LICENSE_EVIDENCE,
        'acquired_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ('schema_version', 'url', 'sha256', 'size_bytes', 'members', 'license_evidence'):
            if existing.get(key) != manifest[key]:
                raise ValueError('Existing asset manifest differs; refusing to overwrite evidence')
    else:
        encoded = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.partial', delete=False) as output:
            temporary = Path(output.name)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('artifacts/pose'))
    args = parser.parse_args()
    print(acquire(args.directory))


if __name__ == '__main__':
    main()
