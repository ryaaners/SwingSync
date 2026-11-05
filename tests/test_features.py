"""Offline cache tests validate software mechanics, never pretrained accuracy."""
from dataclasses import replace
from fractions import Fraction
import json

import av
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from backend.swingsync.research.features import (  # noqa: E402
    FeatureClip, backbone_state_sha256, build_cache_identity, cache_key,
    extract_clip_features, load_feature_cache, save_feature_cache,
)
from backend.swingsync.research.golfdb import GolfDBRecord  # noqa: E402


class TinyFrozen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(1.0))
        self.batch_lengths = []

    def forward(self, images):
        self.batch_lengths.append(len(images))
        return images.float().mean(dim=(-2, -1)) * self.scale


@pytest.fixture
def fixture_video(tmp_path):
    path = tmp_path / 'software-fixture.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=25)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        stream.codec_context.time_base = Fraction(1, 1000)
        for i in range(9):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 20, dtype=np.uint8), format='rgb24')
            frame.pts, frame.time_base = i * 40 + i * i * 2, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    record = GolfDBRecord(0, 'software-source', 'fixture', 'unknown', 'iron', 'face-on', False,
                          (1, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0., 0., 1., 1.), 1)
    return path, record


def make_clip(fixture_video):
    path, record = fixture_video
    backbone = TinyFrozen()
    clip = extract_clip_features(path, record, annotation_sha256='a' * 64,
                                 backbone=backbone, batch_size=4)
    return clip, backbone


def test_streaming_default_transform_preserves_all_timestamps_and_labels(fixture_video):
    clip, backbone = make_clip(fixture_video)
    assert backbone.batch_lengths == [4, 4, 1]
    assert clip.features.shape == (9, 3)
    assert clip.features.dtype == np.float32
    assert clip.labels.tolist() == list(range(8)) + [8]
    assert clip.pts_seconds == pytest.approx([(i * 40 + i * i * 2) / 1000 for i in range(9)])
    assert clip.historical_source_indices.tolist() == list(range(9))
    assert clip.metadata['identity']['timestamp_provenance'] == 'reencoded-clip-pts'
    assert clip.metadata['identity']['transform']['resize_short_edge'] == 232
    assert clip.metadata['identity']['transform']['stochastic'] is False
    assert not backbone.training


def test_exact_cache_roundtrip_reuse_and_atomic_no_overwrite(tmp_path, fixture_video):
    clip, _ = make_clip(fixture_video)
    path = tmp_path / 'cache.npz'
    save_feature_cache(clip, path)
    loaded = load_feature_cache(path, expected_identity=clip.metadata['identity'])
    for name in ('features', 'labels', 'pts_seconds', 'historical_source_indices'):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(clip, name))
    assert loaded.metadata == clip.metadata
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_feature_cache(clip, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.part'))
    with np.load(path, allow_pickle=False) as cache:
        assert cache['metadata'].dtype.kind == 'U'


def test_identity_changes_with_backbone_bytes_annotation_video_or_transform(fixture_video):
    path, record = fixture_video
    backbone = TinyFrozen()
    initial = build_cache_identity(path, record, 'a' * 64, backbone)
    assert initial == build_cache_identity(path, record, 'a' * 64, backbone)
    before = backbone_state_sha256(backbone)
    backbone.scale += 1
    assert backbone_state_sha256(backbone) != before
    assert cache_key(build_cache_identity(path, record, 'a' * 64, backbone)) != cache_key(initial)
    assert cache_key(build_cache_identity(path, record, 'b' * 64, TinyFrozen())) != cache_key(initial)
    changed = replace(record, source_id='other-source')
    assert cache_key(build_cache_identity(path, changed, 'a' * 64, TinyFrozen())) != cache_key(initial)
    transformed = build_cache_identity(path, record, 'a' * 64, TinyFrozen(), {'name': 'different-explicit-transform'})
    assert cache_key(transformed) != cache_key(initial)
    path.write_bytes(path.read_bytes() + b'changed')
    assert cache_key(build_cache_identity(path, record, 'a' * 64, TinyFrozen())) != cache_key(initial)


def test_cache_rejects_wrong_provenance_checksum_and_semantic_labels(tmp_path, fixture_video):
    clip, _ = make_clip(fixture_video)
    path = tmp_path / 'cache.npz'
    save_feature_cache(clip, path)
    wrong = dict(clip.metadata['identity'], video_sha256='b' * 64)
    with pytest.raises(ValueError, match='provenance'):
        load_feature_cache(path, expected_identity=wrong)
    altered = clip.features.copy()
    altered[0, 0] += 1
    bad = FeatureClip(altered, clip.labels, clip.pts_seconds, clip.historical_source_indices, clip.metadata)
    with pytest.raises(ValueError, match='checksum'):
        save_feature_cache(bad, tmp_path / 'modified.npz')
    labels = clip.labels.copy()
    labels[0] = 8
    bad = replace(clip, labels=labels)
    with pytest.raises(ValueError, match='Labels differ'):
        save_feature_cache(bad, tmp_path / 'labels.npz')
    with np.load(path, allow_pickle=False) as cached:
        arrays = {k: cached[k] for k in cached.files}
    arrays['features'] = altered
    np.savez(tmp_path / 'tampered.npz', **arrays)
    with pytest.raises(ValueError, match='checksum'):
        load_feature_cache(tmp_path / 'tampered.npz')


def test_invalid_cache_types_and_inputs_are_rejected(tmp_path, fixture_video):
    path, record = fixture_video
    with pytest.raises(ValueError, match='frozen'):
        extract_clip_features(path, record, annotation_sha256='a' * 64, backbone=torch.nn.Linear(3, 3))
    with pytest.raises(ValueError, match='transform_metadata'):
        extract_clip_features(path, record, annotation_sha256='a' * 64,
                              backbone=TinyFrozen(), transform=lambda t: t.float())
    for size in (0, 257, True):
        with pytest.raises(ValueError, match='batch_size'):
            extract_clip_features(path, record, annotation_sha256='a' * 64,
                                  backbone=TinyFrozen(), batch_size=size)
    with pytest.raises(ValueError):
        load_feature_cache(tmp_path / 'missing.npz')
    np.savez(tmp_path / 'object.npz', features=np.array([{}], dtype=object), metadata=np.array('{}'))
    with pytest.raises(ValueError):
        load_feature_cache(tmp_path / 'object.npz')
    np.savez(tmp_path / 'badmeta.npz', **{
        'features': np.zeros((9, 3), np.float32), 'labels': np.arange(9, dtype=np.int64),
        'pts_seconds': np.arange(9, dtype=np.float64), 'historical_source_indices': np.arange(9, dtype=np.int64),
        'metadata': np.array(json.dumps({'identity': {}, 'cache_key': 'wrong'})),
    })
    with pytest.raises(ValueError, match='identity'):
        load_feature_cache(tmp_path / 'badmeta.npz')


def test_identity_includes_pipeline_and_actual_forward_implementation(fixture_video, monkeypatch):
    path, record = fixture_video
    initial = build_cache_identity(path, record, 'a' * 64, TinyFrozen())
    hashes = initial['implementation_sha256']
    assert {'features.py', 'models.py', 'timeline.py', 'golfdb.py',
            'backbone_module', 'backbone_forward_code'} <= set(hashes)

    def changed_forward(self, images):
        return images.float().sum(dim=(-2, -1)) * self.scale

    monkeypatch.setattr(TinyFrozen, 'forward', changed_forward)
    changed = build_cache_identity(path, record, 'a' * 64, TinyFrozen())
    assert changed['backbone_state_sha256'] == initial['backbone_state_sha256']
    assert changed['implementation_sha256']['backbone_forward_code'] != hashes['backbone_forward_code']
    assert cache_key(changed) != cache_key(initial)


def test_corrupt_archive_is_rejected_as_validation_error(tmp_path):
    path = tmp_path / 'broken.npz'
    path.write_bytes(b'not a ZIP archive')
    with pytest.raises(ValueError, match='corrupt'):
        load_feature_cache(path)
