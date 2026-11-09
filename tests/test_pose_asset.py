"""Asset download publication/admission checks with explicitly synthetic bytes."""
from contextlib import contextmanager
import hashlib
import io
import json
import zipfile

import pytest

from scripts import acquire_pose_asset as asset


@pytest.fixture
def synthetic_asset(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('test-only.tflite', b'synthetic fixture; never a trained model')
    data = buffer.getvalue()
    member = b'synthetic fixture; never a trained model'
    monkeypatch.setattr(asset, 'MODEL_SHA256', hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(asset, 'MODEL_SIZE', len(data))
    monkeypatch.setattr(asset, 'MEMBERS', [{'name': 'test-only.tflite', 'size_bytes': len(member),
                                         'sha256': hashlib.sha256(member).hexdigest()}])
    calls = []

    @contextmanager
    def download(url, timeout):
        calls.append(url)
        yield io.BytesIO(data)

    monkeypatch.setattr(asset.urllib.request, 'urlopen', download)
    return data, calls


def test_asset_acquisition_reuses_verified_bytes_and_preserves_evidence(tmp_path, synthetic_asset):
    _, calls = synthetic_asset
    manifest = asset.acquire(tmp_path)
    before = manifest.read_bytes()
    assert asset.acquire(tmp_path).read_bytes() == before
    assert len(calls) == 1
    assert not list(tmp_path.glob('*.partial'))


def test_changed_local_model_refused_without_redownload(tmp_path, synthetic_asset):
    _, calls = synthetic_asset
    asset.acquire(tmp_path)
    model = tmp_path / 'pose_landmarker_full-v1.task'
    model.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='pinned reviewed asset'):
        asset.acquire(tmp_path)
    assert model.read_bytes() == b'corrupt' and len(calls) == 1


def test_changed_license_manifest_refused_without_overwrite(tmp_path, synthetic_asset):
    asset.acquire(tmp_path)
    manifest = tmp_path / 'pose_landmarker_full-v1.manifest.json'
    value = json.loads(manifest.read_text())
    value['license_evidence'] = {'invented': 'unrestricted redistribution'}
    manifest.write_text(json.dumps(value))
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match='refusing to overwrite evidence'):
        asset.acquire(tmp_path)
    assert manifest.read_bytes() == before


def test_changed_download_never_publishes_model_or_manifest(tmp_path, monkeypatch):
    @contextmanager
    def bad_download(url, timeout):
        yield io.BytesIO(b'not official model bytes')
    monkeypatch.setattr(asset.urllib.request, 'urlopen', bad_download)
    with pytest.raises(ValueError, match='pinned reviewed asset'):
        asset.acquire(tmp_path)
    assert list(tmp_path.iterdir()) == []
