"""Research-only selected event proposals from generic local video pixels.

No labels or GolfDBRecord enter extraction. Compatibility means frozen tensor
preprocessing and frame correspondence, never uploaded-video accuracy/reliability.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np

from .evaluation import EVENT_NAMES

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / 'artifacts/training/active-training.lock'
DEFAULT_PROTOCOL = ROOT / 'experiments/event-input-v1.json'
CONTEXT = 'reference-64-reset'
BATCH_SIZE = 16
ARRAY_NAMES = ('features', 'frame_probabilities', 'temporal_probabilities', 'pts_seconds')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _publish_json(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.part')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_arrays(path, arrays):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=Path(path).parent, suffix='.part', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_identity(protocol):
    files = [Path(__file__), ROOT/'backend/swingsync/research/event_input_contract.py',
             ROOT/'scripts/propose_events.py', Path(protocol)]
    return {str(p.resolve().relative_to(ROOT)): sha256(p) for p in files}


def inspect_bounded(video):
    """Native decoder preflight isolated before constructing models."""
    try:
        child = subprocess.run([sys.executable, '-m', 'backend.swingsync.research.event_input',
                                '--inspect', str(video)], cwd=ROOT,
                               capture_output=True, text=True, check=True, timeout=45)
        return json.loads(child.stdout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError('Input decode preflight failed or exceeded45seconds') from exc


def proposal_rows(probabilities, frames):
    """Independent class peaks; no chronology repair or calibrated confidence."""
    array = np.asarray(probabilities)
    if (array.dtype != np.float32 or array.shape != (len(frames), 9) or len(frames) < 2
            or not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1)
            or not np.allclose(array.sum(axis=1), 1., rtol=0, atol=1e-6)):
        raise ValueError('Finite normalized float32[T,9] probabilities required')
    peaks = np.argsort(array[:, :8], axis=0)[-1]
    events = []
    for label, peak in enumerate(peaks):
        frame = frames[int(peak)]
        events.append({'name': EVENT_NAMES[label], 'class_index': label,
                       'frame_index': int(peak), 'pts_seconds': frame['pts_seconds'],
                       'pts': frame['pts'], 'time_base_numerator': frame['time_base_numerator'],
                       'time_base_denominator': frame['time_base_denominator'],
                       'score': float(array[peak, label]),
                       'background_probability': float(array[peak, 8])})
    indices = [e['frame_index'] for e in events]
    return {'events': events, 'strict_order': all(a < b for a, b in zip(indices, indices[1:])),
            'duplicate_peak_count': len(indices) - len(set(indices))}


def _pipeline_identity(backbone):
    import av
    import torch
    import torchvision
    from .features import DEFAULT_TRANSFORM, _implementation_hashes, backbone_state_sha256
    return json.loads(json.dumps({
        'backbone_class': f'{type(backbone).__module__}.{type(backbone).__qualname__}',
        'backbone_state_sha256': backbone_state_sha256(backbone),
        'backbone_provenance': backbone.provenance, 'transform': DEFAULT_TRANSFORM,
        'numpy': np.__version__, 'torch': str(torch.__version__),
        'torchvision': torchvision.__version__, 'pyav': av.__version__,
        'ffmpeg_libraries': av.library_versions, 'extraction_device': 'cpu',
        'implementation_sha256': _implementation_hashes(backbone), 'python': platform.python_version(),
    }, allow_nan=False))


def _prepare_models(contract, annotations, manifest):
    import torch
    from .heldout import validate_contract
    from .latency import _load_local_backbone
    if sys.version_info[:2] != (3, 11) or sys.prefix == sys.base_prefix:
        raise ValueError('Use a Python3.11 environment matching the frozen ML lock')
    admission = validate_contract(contract, annotations, manifest)
    os.environ.setdefault('TORCH_HOME', str(ROOT/'data/torch'))
    backbone, weight = _load_local_backbone()
    pipeline = _pipeline_identity(backbone)
    if pipeline != admission['contract']['feature_pipeline']:
        raise ValueError('Generic input backbone/transform/runtime differs from frozen pipeline')
    torch.set_num_threads(1)
    return admission, backbone, weight, pipeline


def extract_generic_features(video, inspection, backbone):
    """All decoded frames, actual PTS, bounded RGB batches; no annotation input."""
    import av
    import torch
    from torchvision.models import MobileNet_V2_Weights
    from .event_input_contract import _check_geometry
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('A fully frozen backbone is required')
    transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
    pending, outputs = [], []
    frames = inspection['frames']
    formats = {'.mp4': 'mov', '.mov': 'mov', '.webm': 'matroska', '.avi': 'avi'}
    path = Path(video)
    if path.suffix.lower() not in formats:
        raise ValueError('Unsupported local video format')
    backbone.to(device='cpu', dtype=torch.float32).eval()

    def flush():
        if not pending:
            return
        with torch.inference_mode():
            value = backbone(torch.stack(pending))
        if (value.shape != (len(pending), 1280) or value.dtype != torch.float32
                or not torch.isfinite(value).all()):
            raise ValueError('Frozen backbone returned malformed features')
        outputs.append(value.detach().cpu().numpy().copy())
        pending.clear()

    count = 0
    with av.open(str(path), format=formats[path.suffix.lower()], options={'protocol_whitelist': 'file'}) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Exactly one video stream required')
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(video=0)):
            if index >= len(frames):
                raise ValueError('Decoded frame count changed')
            _check_geometry(stream, frame)
            saved = frames[index]
            if (frame.pts is None or frame.time_base is None
                    or (frame.pts, frame.time_base.numerator, frame.time_base.denominator)
                    != (saved['pts'], saved['time_base_numerator'], saved['time_base_denominator'])
                    or float(frame.pts * frame.time_base) != saved['pts_seconds']
                    or (frame.width, frame.height) != (inspection['width'], inspection['height'])):
                raise ValueError('Native PTS/raster correspondence changed during extraction')
            rgb = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1)
            transformed = transform(rgb)
            if transformed.shape != (3, 224, 224) or transformed.dtype != torch.float32:
                raise ValueError('Frozen input transform returned a different tensor geometry')
            pending.append(transformed)
            if len(pending) == BATCH_SIZE:
                flush()
            count += 1
        flush()
    if count != len(frames):
        raise ValueError('Decoded frame coverage changed')
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _predict_heads(admission, features):
    from .heldout import predict_probabilities
    return {name: predict_probabilities(head, features, CONTEXT)
            for name, head in admission['heads'].items()}


def _assert_integrity(identity, admission, backbone, weight, *, video, contract, protocol, annotations, manifest):
    from .heldout import _assert_run_integrity
    if (sha256(video) != identity['video_sha256'] or _source_identity(protocol) != identity['source_sha256']
            or sha256(annotations) != identity['annotations_sha256']
            or sha256(manifest) != identity['manifest_sha256']
            or sha256(weight) != identity['backbone_weight_sha256']
            or _pipeline_identity(backbone) != identity['feature_pipeline']):
        raise ValueError('Live source/media/model/transform/runtime drift; publication refused')
    for head in admission['contract']['heads'].values():
        if sha256(head['checkpoint_path']) != head['checkpoint_sha256']:
            raise ValueError('Selected event checkpoint changed during inference')
    _assert_run_integrity(admission['immutable'], contract)


def run_event_input(video, output, *, metadata, contract, annotations, manifest, protocol=DEFAULT_PROTOCOL):
    """One-shot offline observation job. The normal app remains model-unavailable."""
    from .event_input_contract import validate_metadata, validate_protocol
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run this research command from the repository root')
    context = validate_metadata(**metadata)
    video, output, contract, annotations, manifest, protocol = map(
        lambda p: Path(p).resolve(), (video, output, contract, annotations, manifest, protocol))
    protocol_value = validate_protocol(protocol)
    if (sha256(contract) != protocol_value['model_contract']['sha256']
            or sha256(ROOT/'uv.lock') != protocol_value['dependency_lock_sha256']):
        raise ValueError('Supplied model contract/root lock differs from registered input protocol')
    if output.exists():
        raise ValueError('Output exists; inspect the saved identity rather than duplicate work')
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another expensive job owns the shared lock') from exc
        sources = _source_identity(protocol)
        source_hash = sha256(video)
        output.mkdir(parents=True, exist_ok=False)
        ledger_path = output/'run.json'
        ledger = {'schema_version': 1, 'state': 'initializing', 'job_id': uuid.uuid4().hex,
                  'created_at': _now(), 'pid': os.getpid(), 'context': context,
                  'video_path': str(video), 'video_sha256': source_hash,
                  'source_sha256': sources, 'input_protocol': protocol_value, 'resume_supported': False,
                  'normal_model_available': False, 'validated_for_serving': False,
                  'accuracy_metrics_computed': False, 'advice': []}
        _publish_json(ledger_path, ledger)
        started = time.perf_counter()
        try:
            inspection = inspect_bounded(video)
            if sha256(video) != source_hash or sources != _source_identity(protocol):
                raise ValueError('Source changed during preflight')
            admission, backbone, weight, pipeline = _prepare_models(contract, annotations, manifest)
            identity = {'video_sha256': source_hash, 'source_sha256': sources,
                        'annotations_sha256': sha256(annotations), 'manifest_sha256': sha256(manifest),
                        'backbone_weight_sha256': sha256(weight), 'feature_pipeline': pipeline,
                        'selected_head_admission': admission['immutable'],
                        'heldout_contract_sha256': sha256(contract),
                        'selected_heads': admission['contract']['heads'],
                        'context': CONTEXT, 'batch_size': BATCH_SIZE, 'protocol_sha256': sha256(protocol),
                        'python_executable_sha256': sha256(sys.executable)}
            ledger.update(state='running', inference_started_at=_now(), identity=identity,
                          video={k: v for k, v in inspection.items() if k != 'frames'})
            _publish_json(ledger_path, ledger, replace=True)
            features = extract_generic_features(video, inspection, backbone)
            probabilities = _predict_heads(admission, features)
            if set(probabilities) != {'frame', 'temporal'}:
                raise ValueError('Both selected event heads are required')
            heads = {name: proposal_rows(value, inspection['frames']) for name, value in probabilities.items()}
            arrays = {'features': features, 'frame_probabilities': probabilities['frame'],
                      'temporal_probabilities': probabilities['temporal'],
                      'pts_seconds': np.asarray([f['pts_seconds'] for f in inspection['frames']], dtype=np.float64)}
            _assert_integrity(identity, admission, backbone, weight, video=video, contract=contract,
                              protocol=protocol, annotations=annotations, manifest=manifest)
            _publish_arrays(output/'arrays.npz', arrays)
            result = {**ledger, 'state': 'completed', 'completed_at': _now(),
                      'frames': inspection['frames'], 'heads': heads,
                      'arrays_file': 'arrays.npz', 'arrays_sha256': sha256(output/'arrays.npz'),
                      'primary_head': 'temporal', 'temporal_context': CONTEXT,
                      'selection': 'numpy-default-argsort-last-index-per-class',
                      'order_constrained_decoding': False, 'scores_are_calibrated': False,
                      'domain_compatibility_validated': False, 'physical_timing_validated': False,
                      'outcome_integration_permitted': False,
                      'elapsed_seconds': time.perf_counter()-started,
                      'limitations': ['Research diagnostic only; not uploaded-video accuracy evidence',
                                      'Given-identical-RGB feature parity does not establish raw-video domain compatibility',
                                      'Independent peaks may duplicate or be nonchronological; no decoder repair',
                                      'Scores are uncalibrated and background may exceed a selected event probability',
                                      'No timing measurements, pose, shot outcomes or coaching advice computed']}
            _publish_json(output/'result.json', result)
            _assert_integrity(identity, admission, backbone, weight, video=video, contract=contract,
                              protocol=protocol, annotations=annotations, manifest=manifest)
            if sha256(output/'arrays.npz') != result['arrays_sha256']:
                raise ValueError('Output array bytes changed during publication')
            ledger.update(state='completed', completed_at=result['completed_at'],
                          result_sha256=sha256(output/'result.json'), arrays_sha256=result['arrays_sha256'],
                          elapsed_seconds=result['elapsed_seconds'])
            _publish_json(ledger_path, ledger, replace=True)
            _assert_integrity(identity, admission, backbone, weight, video=video, contract=contract,
                              protocol=protocol, annotations=annotations, manifest=manifest)
            return result
        except BaseException as exc:
            for name in ('result.json', 'arrays.npz'):
                (output/name).unlink(missing_ok=True)
            ledger.update(state='failed', error_type=type(exc).__name__, error=str(exc)[:2000], failed_at=_now())
            _publish_json(ledger_path, ledger, replace=True)
            raise


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--inspect':
        raise SystemExit('Use scripts/propose_events.py for research inference')
    from .event_input_contract import inspect_input
    print(json.dumps(inspect_input(Path(sys.argv[2])), allow_nan=False))
