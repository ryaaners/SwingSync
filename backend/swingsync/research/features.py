"""Frozen full-clip feature caches, with explicit re-encoded timestamp provenance.

The default ImageNet V2 resize232/center224 tensor transform deliberately differs
from the historical SwingNet160 pipeline. It is deterministic and applies the
same spatial rule to every frame. No augmentations or fitted preprocessing here.
Only trusted local research media are decoded; this is not the upload service.
"""
from dataclasses import dataclass
import hashlib
import inspect
import marshal
import platform
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile, ZipFile

import av
import numpy as np
import torch
import torchvision
from torchvision.models import MobileNet_V2_Weights

from .golfdb import GolfDBRecord
from .timeline import inspect_clip_timeline

SCHEMA_VERSION = 1
ARRAY_NAMES = ('features', 'labels', 'pts_seconds', 'historical_source_indices')
DEFAULT_TRANSFORM = {
    'name': 'torchvision-MobileNet_V2_Weights.IMAGENET1K_V2.transforms-tensor',
    'resize_short_edge': 232, 'center_crop': 224, 'interpolation': 'bilinear',
    'antialias': True, 'input': 'RGB-uint8-CHW', 'output': 'float32-normalized',
    'mean': [.485, .456, .406], 'std': [.229, .224, .225], 'stochastic': False,
}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def backbone_state_sha256(backbone) -> str:
    """Hash names, dtypes, dimensions and exact state bytes, including BN buffers."""
    digest = hashlib.sha256()
    for name, tensor in sorted(backbone.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(_json_bytes([name, str(tensor.dtype), list(tensor.shape)]))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _implementation_hashes(backbone):
    """Hash local pipeline modules plus the actual runtime backbone forward code."""
    root = Path(__file__).parent
    module_path = Path(inspect.getfile(type(backbone)))
    forward = type(backbone).forward
    if not module_path.is_file() or not hasattr(forward, '__code__'):
        raise ValueError('Backbone requires inspectable implementation provenance')
    paths = {name: root / name for name in ('features.py', 'models.py', 'timeline.py', 'golfdb.py')}
    hashes = {name: _file_hash(path) for name, path in paths.items()}
    hashes['backbone_module'] = _file_hash(module_path)
    hashes['backbone_forward_code'] = hashlib.sha256(marshal.dumps(forward.__code__)).hexdigest()
    return hashes


def build_cache_identity(path, record: GolfDBRecord, annotation_sha256: str, backbone,
                         transform_metadata=None, *, device='cpu') -> dict:
    if not isinstance(annotation_sha256, str) or len(annotation_sha256) != 64:
        raise ValueError('annotation_sha256 must be a SHA256 hex digest')
    try:
        int(annotation_sha256, 16)
    except ValueError as exc:
        raise ValueError('annotation_sha256 must be a SHA256 hex digest') from exc
    transform_metadata = DEFAULT_TRANSFORM if transform_metadata is None else transform_metadata
    if not isinstance(transform_metadata, dict) or not transform_metadata.get('name'):
        raise ValueError('Transform metadata must identify the exact preprocessing rule')
    return json.loads(_json_bytes({
        'schema_version': SCHEMA_VERSION, 'video_sha256': _file_hash(path),
        'annotation_sha256': annotation_sha256, 'clip_id': record.clip_id,
        'source_id': record.source_id, 'outer_split': record.outer_split,
        'original_events': list(record.original_events),
        'expected_frame_count': record.expected_frame_count,
        'historical_source_indices': [record.source_frame_index(i) for i in range(record.expected_frame_count)],
        'annotation_event_indices': list(record.event_indices),
        'timestamp_provenance': 'reencoded-clip-pts',
        'backbone_class': f'{type(backbone).__module__}.{type(backbone).__qualname__}',
        'backbone_state_sha256': backbone_state_sha256(backbone),
        'backbone_provenance': getattr(backbone, 'provenance', {'weights': 'unspecified-test-backbone'}),
        'transform': transform_metadata, 'numpy': np.__version__, 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'pyav': av.__version__,
        'ffmpeg_libraries': av.library_versions, 'extraction_device': str(device),
        'implementation_sha256': _implementation_hashes(backbone), 'python': platform.python_version(),
    }))


def cache_key(identity: dict) -> str:
    return hashlib.sha256(_json_bytes(identity)).hexdigest()


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(_json_bytes([str(array.dtype), list(array.shape)]))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureClip:
    features: np.ndarray
    labels: np.ndarray
    pts_seconds: np.ndarray
    historical_source_indices: np.ndarray
    metadata: dict


def _validate(clip: FeatureClip, *, check_hashes=True):
    identity = clip.metadata.get('identity', {})
    if not isinstance(identity, dict):
        raise ValueError('Cache identity must be an object')
    if identity.get('schema_version') != SCHEMA_VERSION or clip.metadata.get('cache_key') != cache_key(identity):
        raise ValueError('Cache identity or schema is invalid')
    n = identity.get('expected_frame_count')
    f = clip.features
    if f.dtype != np.float32 or f.ndim != 2 or f.shape[0] != n or f.shape[1] < 1 or not np.isfinite(f).all():
        raise ValueError('Features must be finite float32 [expected_frames, feature_dim]')
    if clip.labels.dtype != np.int64 or clip.labels.shape != (n,):
        raise ValueError('Labels must be int64 matching frame count')
    expected_labels = np.full(n, 8, dtype=np.int64)
    events = identity.get('annotation_event_indices', [])
    if len(events) != 8 or len(set(events)) != 8 or any(not isinstance(i, int) or not 0 <= i < n for i in events):
        raise ValueError('Invalid event indices in cache identity')
    expected_labels[events] = np.arange(8)
    if not np.array_equal(clip.labels, expected_labels):
        raise ValueError('Labels differ from annotation identity')
    if (clip.pts_seconds.dtype != np.float64 or clip.pts_seconds.shape != (n,)
            or not np.isfinite(clip.pts_seconds).all() or np.any(np.diff(clip.pts_seconds) <= 0)):
        raise ValueError('PTS must be finite strictly increasing float64')
    if (clip.historical_source_indices.dtype != np.int64 or clip.historical_source_indices.shape != (n,)
            or not np.array_equal(clip.historical_source_indices, identity.get('historical_source_indices'))):
        raise ValueError('Historical source indices differ from annotation identity')
    if check_hashes:
        hashes = clip.metadata.get('array_sha256', {})
        if not isinstance(hashes, dict):
            raise ValueError('Cache array checksums must be an object')
        if any(hashes.get(name) != _array_hash(getattr(clip, name)) for name in ARRAY_NAMES):
            raise ValueError('Cache array checksum mismatch')


def extract_clip_features(path, record: GolfDBRecord, *, annotation_sha256: str, backbone,
                          transform=None, transform_metadata=None, batch_size=16, device='cpu') -> FeatureClip:
    """Sequential preprocessed-clip decode with only one bounded RGB batch in memory.

    Features themselves are retained per clip. Exact source PTS cannot be inferred
    from re-encoding; historical source indices are explicitly a different field.
    Custom transforms require explicit metadata and are caller-owned research
    configuration; default transform uses no random operation.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 256:
        raise ValueError('batch_size must be between1 and256')
    if transform is not None and transform_metadata is None:
        raise ValueError('Custom transforms require explicit transform_metadata')
    if transform is None and transform_metadata is not None and transform_metadata != DEFAULT_TRANSFORM:
        raise ValueError('Default transform must use its actual transform metadata')
    if transform_metadata and transform_metadata.get('stochastic'):
        raise ValueError('This cache stage requires deterministic temporal preprocessing')
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('Feature extraction requires a fully frozen backbone')
    transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms() if transform is None else transform
    timeline = inspect_clip_timeline(path, record, origin='preprocessed')
    identity = build_cache_identity(path, record, annotation_sha256, backbone, transform_metadata, device=device)
    backbone.to(device)
    backbone.eval()
    batches, pending, pts = [], [], []

    def flush():
        if not pending:
            return
        with torch.inference_mode():
            outputs = backbone(torch.stack(pending).to(device))
        if outputs.is_complex() or outputs.ndim != 2 or outputs.shape[0] != len(pending) or not torch.isfinite(outputs).all():
            raise ValueError('Backbone must return finite [batch, feature_dim] features')
        batches.append(outputs.detach().cpu().to(torch.float32).numpy())
        pending.clear()

    path = Path(path)
    formats = {'.mp4': 'mov', '.mov': 'mov', '.webm': 'matroska', '.avi': 'avi'}
    with av.open(str(path), format=formats[path.suffix.lower()], options={'protocol_whitelist': 'file'}) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= record.expected_frame_count:
                raise ValueError('Video changed or extra frame found during feature extraction')
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Missing frame PTS during feature extraction')
            pts.append(float(frame.pts * frame.time_base))
            rgb = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1)
            image = transform(rgb)
            if image.dtype != torch.float32 or image.ndim != 3 or image.shape[0] != 3 or not torch.isfinite(image).all():
                raise ValueError('Transform must return finite RGB CHW tensors')
            pending.append(image)
            if len(pending) == batch_size:
                flush()
        flush()
    if tuple(pts) != timeline.pts_seconds or _file_hash(path) != identity['video_sha256']:
        raise ValueError('Video changed between timeline inspection and feature extraction')
    if backbone_state_sha256(backbone) != identity['backbone_state_sha256']:
        raise ValueError('Frozen backbone state changed during extraction')
    arrays = {
        'features': np.concatenate(batches).astype(np.float32, copy=False),
        'labels': np.asarray(timeline.labels, dtype=np.int64),
        'pts_seconds': np.asarray(pts, dtype=np.float64),
        'historical_source_indices': np.asarray(timeline.historical_source_indices, dtype=np.int64),
    }
    metadata = {'identity': identity, 'cache_key': cache_key(identity),
                'array_sha256': {name: _array_hash(array) for name, array in arrays.items()},
                'evidence_scope': 'Frozen features only; no event model, benchmark or source-image equivalence'}
    clip = FeatureClip(**arrays, metadata=metadata)
    _validate(clip)
    return clip


def save_feature_cache(clip: FeatureClip, path) -> None:
    """Atomic publish via hard link, never replace an existing completed cache."""
    _validate(clip)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            np.savez_compressed(output, **{name: getattr(clip, name) for name in ARRAY_NAMES},
                                metadata=np.array(_json_bytes(clip.metadata).decode()))
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_feature_cache(path, *, expected_identity=None) -> FeatureClip:
    """Pickle-free load with semantic checks and exact array/identity hashes.

    Cache files are local research artifacts, bounded here to256MiB uncompressed.
    Checksums detect corruption, not malicious parties able to rewrite metadata.
    """
    try:
        with ZipFile(path) as zipped:
            if sum(i.file_size for i in zipped.infolist()) > 256 * 1024 * 1024:
                raise ValueError('Cache exceeds uncompressed size budget')
        with np.load(path, allow_pickle=False) as cached:
            if set(cached.files) != set(ARRAY_NAMES) | {'metadata'}:
                raise ValueError('Cache schema contains unexpected arrays')
            metadata_raw = cached['metadata']
            if metadata_raw.ndim != 0 or metadata_raw.dtype.kind != 'U':
                raise ValueError('Cache metadata must be a JSON string')
            metadata = json.loads(str(metadata_raw.item()))
            if not isinstance(metadata, dict):
                raise ValueError('Cache metadata must be an object')
            arrays = {name: cached[name].copy() for name in ARRAY_NAMES}
        clip = FeatureClip(**arrays, metadata=metadata)
        _validate(clip)
        if expected_identity is not None and metadata['identity'] != expected_identity:
            raise ValueError('Cache provenance does not match expected identity')
        return clip
    except (OSError, BadZipFile, EOFError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Missing or corrupt feature cache') from exc
