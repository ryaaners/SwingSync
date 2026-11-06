"""Local CPU latency research; timing evidence never enables web inference.

Measurements separate decoder iteration from RGB conversion/preprocessing and
backbone forward batches. Shared features feed each fixed selected head/context.
A composed single-model duration sums its observed shared feature-extraction wall
time and head time; the observed whole-clip wall time instead includes ALL heads.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import av
import numpy as np
import torch
from torchvision.models import MobileNet_V2_Weights

from .cache_job import DEFAULT_LOCK_PATH, _git_base_commit, _process_start, _publish_json
from .features import backbone_state_sha256, build_cache_identity
from .golfdb import GolfDBRecord
from .models import FrozenMobileNetV2, detect_devices
from .timeline import inspect_clip_timeline


CONTEXTS = ('reference-64-reset', 'full-clip')
LATENCY_CONFIG = {'clip_ids': [0, 1], 'batch_size': 16, 'warmup_frames': 16,
                  'repetitions': 1, 'contexts': list(CONTEXTS)}


def _hash(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _exclusive_json(path, value):
    """No-overwrite publication, including against another concurrent writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.latency-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as output:
            json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def timing_summary(seconds, *, sample_unit):
    values = np.asarray(seconds, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Timing samples must be a nonempty finite nonnegative vector')
    return {'sample_unit': sample_unit, 'sample_count': len(values),
            'total_seconds': float(values.sum()), 'mean_seconds': float(values.mean()),
            'p50_seconds': float(np.quantile(values, .5)), 'p95_seconds': float(np.quantile(values, .95)),
            'quantile_method': 'numpy-linear; descriptive sample quantiles, not confidence bounds'}


def _extract_timed(path, record: GolfDBRecord, backbone, *, batch_size=16, limit=None,
                   clock=time.perf_counter):
    """Bound RGB memory to one batch, retain only one clip's features and actual PTS.

    CPU timings are synchronous. Backbone samples include stacking/copying inputs,
    forward, output checks and conversion to float32 NumPy. Decode samples time
    next(iterator) only; RGB conversion belongs to preprocessing. Container open,
    EOF drain and the complete extraction wall time are separately recorded.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 256:
        raise ValueError('batch_size must be an integer within [1,256]')
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError('limit must be a positive integer')
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('Backbone must be frozen')
    path = Path(path)
    formats = {'.mp4': 'mov', '.mov': 'mov', '.webm': 'matroska', '.avi': 'avi'}
    if path.suffix.lower() not in formats:
        raise ValueError('Unsupported local video extension')
    transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
    outputs, pending, pts, decode, preprocessing, forward, batches = [], [], [], [], [], [], []
    backbone.to(device='cpu', dtype=torch.float32).eval()
    started = clock()
    opened = clock()
    container = av.open(str(path), format=formats[path.suffix.lower()], options={'protocol_whitelist': 'file'})
    open_seconds = clock() - opened
    eof_seconds = 0.

    def flush():
        if not pending:
            return
        begin = clock()
        with torch.inference_mode():
            value = backbone(torch.stack(pending))
        if (not isinstance(value, torch.Tensor) or value.is_complex() or value.ndim != 2
                or value.shape[0] != len(pending) or not torch.isfinite(value).all()):
            raise ValueError('Backbone must return finite [batch, feature_dim] values')
        outputs.append(value.detach().cpu().to(torch.float32).numpy())
        forward.append(clock() - begin)
        batches.append(len(pending))
        pending.clear()

    try:
        if len(container.streams.video) != 1:
            raise ValueError('Exactly one video stream required')
        frames = iter(container.decode(video=0))
        while limit is None or len(pts) < limit:
            begin = clock()
            try:
                frame = next(frames)
            except StopIteration:
                eof_seconds = clock() - begin
                break
            elapsed = clock() - begin
            if len(pts) >= record.expected_frame_count:
                raise ValueError('Decoded frame count exceeds the audited trim')
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Missing actual decoded PTS')
            timestamp = float(frame.pts * frame.time_base)
            if not np.isfinite(timestamp) or (pts and timestamp <= pts[-1]):
                raise ValueError('Decoded PTS must be finite and strictly increasing')
            pts.append(timestamp)
            decode.append(elapsed)
            begin = clock()
            rgb = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1)
            image = transform(rgb)
            if image.dtype != torch.float32 or image.shape != (3, 224, 224) or not torch.isfinite(image).all():
                raise ValueError('Registered image transform produced invalid output')
            pending.append(image)
            preprocessing.append(clock() - begin)
            if len(pending) == batch_size:
                flush()
        flush()
    finally:
        container.close()
    if not pts or (limit is None and len(pts) != record.expected_frame_count):
        raise ValueError('Decoded frame count differs from the audited trim')
    features = np.concatenate(outputs).astype(np.float32, copy=False)
    extraction_wall = clock() - started
    return features, np.asarray(pts, dtype=np.float64), {
        'frame_count': len(pts), 'batch_frame_counts': batches,
        'decode': timing_summary(decode, sample_unit='successful decoded frame retrieval'),
        'preprocessing': timing_summary(preprocessing, sample_unit='frame RGB conversion plus transform'),
        'backbone': timing_summary(forward, sample_unit='forward batch including stack/output transfer'),
        'container_open_seconds': open_seconds, 'decoder_eof_drain_seconds': eof_seconds,
        'feature_extraction_wall_seconds': extraction_wall,
    }


def measure_clip(path, record, backbone, heads, *, expected_pts, batch_size=16,
                 contexts=CONTEXTS, clock=time.perf_counter):
    """Measure one clip and every requested fixed head; compute no event accuracy."""
    from .heldout import predict_probabilities
    if tuple(contexts) != CONTEXTS or set(heads) != {'frame', 'temporal'}:
        raise ValueError('Benchmark must retain both heads and both fixed contexts')
    started = clock()
    features, pts, report = _extract_timed(path, record, backbone, batch_size=batch_size, clock=clock)
    if not np.array_equal(pts, np.asarray(expected_pts, dtype=np.float64)):
        raise ValueError('Timed decode PTS differ from preflight inspection')
    head_times = {}
    for name in ('frame', 'temporal'):
        head_times[name] = {}
        for context in contexts:
            begin = clock()
            probabilities = predict_probabilities(heads[name], features, context)
            elapsed = clock() - begin
            if probabilities.shape != (len(pts), 9) or not np.isfinite(probabilities).all():
                raise ValueError('Selected head produced invalid probabilities')
            head_times[name][context] = {
                'head_seconds': elapsed,
                'composed_video_pipeline_seconds': report['feature_extraction_wall_seconds'] + elapsed,
                'composition': 'observed shared feature-extraction wall plus this head; not a separate replay',
            }
    report.update(clip_id=record.clip_id, source_id=record.source_id,
                  first_pts_seconds=float(pts[0]), last_pts_seconds=float(pts[-1]),
                  pts_sha256=hashlib.sha256(pts.tobytes()).hexdigest(),
                  timestamp_provenance='reencoded-clip-pts', heads=head_times,
                  measured_clip_wall_seconds=clock() - started,
                  measured_clip_wall_scope='shared extraction plus all four head/context calls and checks')
    return report


def _load_local_backbone():
    """Use an already present official file; this benchmark never downloads weights."""
    os.environ.setdefault('TORCH_HOME', str(Path('data/torch').resolve()))
    weight = Path(torch.hub.get_dir()) / 'checkpoints' / 'mobilenet_v2-7ebf99e0.pth'
    if not weight.is_file():
        raise ValueError(f'Required local ImageNet weight file is unavailable: {weight}')
    # Verify the full known file hash, not just torchvision's short filename prefix.
    raw = weight.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f':
        raise ValueError('Local ImageNet weight file differs from the audited reference')
    # None never downloads; load the exact bytes already checked into the frozen feature layers.
    backbone = FrozenMobileNetV2(weights=None)
    state = torch.load(BytesIO(raw), map_location='cpu', weights_only=True)
    backbone.features.load_state_dict({k[len('features.'):]: v for k, v in state.items()
                                       if k.startswith('features.')}, strict=True)
    backbone.provenance.update(weights='IMAGENET1K_V2',
                               weights_url=MobileNet_V2_Weights.IMAGENET1K_V2.url)
    return backbone, weight


def _hardware():
    hardware = {'machine': platform.machine(), 'processor': platform.processor(),
                'logical_cpu_count': os.cpu_count(), 'physical_memory_bytes': None}
    if platform.system() == 'Darwin':
        for key, field in (('machdep.cpu.brand_string', 'processor'), ('hw.memsize', 'physical_memory_bytes')):
            try:
                value = subprocess.check_output(['sysctl', '-n', key], text=True, timeout=5).strip()
                hardware[field] = int(value) if field == 'physical_memory_bytes' else value
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
    else:
        try:
            hardware['physical_memory_bytes'] = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except (OSError, ValueError):
            pass
    return hardware


def _assert_benchmark_integrity(state, contract_path):
    """Refuse completed timing evidence after live helper/config/runtime/code drift."""
    from .heldout import _assert_run_integrity
    _assert_run_integrity(state['immutable'], contract_path)
    expected = state['timing_implementation_sha256']
    paths = {'latency.py': Path(__file__),
             'benchmark_event_pipeline.py': Path(__file__).resolve().parents[3]
             / 'scripts/benchmark_event_pipeline.py'}
    if any(_hash(path) != expected[name] for name, path in paths.items()):
        raise ValueError('Timing source bytes changed during invocation; publication refused')


def run_latency_benchmark(contract_path, annotations, manifest, clips_dir, output):
    """Exclusive, one-shot timing run on two registered TRAIN clips; no test data."""
    from .heldout import predict_probabilities, validate_contract
    output = Path(output).resolve()
    state_path = output.with_name(output.name + '.run.json')
    if output.exists() or state_path.exists():
        raise ValueError('Benchmark output/state already exists; inspect it and use a new explicit output')
    DEFAULT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_LOCK_PATH.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another feature/training/evaluation job owns the shared lock') from exc
        state = {'schema_version': 1, 'state': 'initializing', 'started_at': _now(),
                 'pid': os.getpid(), 'process_start_identity': _process_start(os.getpid()),
                 'git_base_commit': _git_base_commit(), 'evidence_type': 'local-CPU-latency-not-accuracy',
                 'output': str(output), 'validated_for_serving': False,
                 'timing_implementation_sha256': {
                     'latency.py': _hash(__file__),
                     'benchmark_event_pipeline.py': _hash(Path(__file__).resolve().parents[3]
                                                         / 'scripts/benchmark_event_pipeline.py')}}
        _exclusive_json(state_path, state)
        try:
            initialized = time.perf_counter()
            admission = validate_contract(contract_path, annotations, manifest)
            contract, dataset = admission['contract'], admission['dataset']
            split, heads = admission['manifest'], admission['heads']
            latency_config = contract.get('latency', {})
            if (not isinstance(latency_config, dict)
                    or {k: latency_config.get(k) for k in LATENCY_CONFIG} != LATENCY_CONFIG
                    or set(latency_config) - set(LATENCY_CONFIG) - {'scope'}):
                raise ValueError('Benchmark differs from the fixed train-only latency contract')
            ids = contract['latency']['clip_ids']
            if ids != sorted(split['partitions']['train']['clip_ids'])[:2]:
                raise ValueError('Latency inputs must be the first two training clips only')
            records = {r.clip_id: r for r in dataset.records}
            backbone, weight_path = _load_local_backbone()
            clips_dir = Path(clips_dir).resolve()
            preflight = []
            for clip_id in ids:
                record, path = records[clip_id], clips_dir / f'{clip_id}.mp4'
                identity = build_cache_identity(path, record, dataset.sha256, backbone, device='cpu')
                pipeline = {key: identity[key] for key in contract['feature_pipeline']}
                if pipeline != contract['feature_pipeline']:
                    raise ValueError('Actual timing backbone/transform/runtime differs from registered pipeline')
                timeline = inspect_clip_timeline(path, record, origin='preprocessed')
                preflight.append((path, record, identity, timeline))
            state.update(state='warming', immutable=admission['immutable'], config=LATENCY_CONFIG,
                         runtime={'platform': platform.platform(), 'hardware': _hardware(), 'devices': detect_devices(),
                                  'torch': str(torch.__version__), 'numpy': np.__version__,
                                  'torch_num_threads': torch.get_num_threads(),
                                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
                         reference_weight_path=str(weight_path), reference_weight_sha256=_hash(weight_path),
                         initialization_seconds=time.perf_counter() - initialized)
            _publish_json(state_path, state)
            _assert_benchmark_integrity(state, contract_path)
            warm = time.perf_counter()
            features, _, warm_details = _extract_timed(preflight[0][0], preflight[0][1], backbone,
                                                      batch_size=16, limit=16)
            if len(features) != 16:
                raise ValueError('The fixed warmup requires exactly sixteen training frames')
            for model in heads.values():
                for context in CONTEXTS:
                    predict_probabilities(model, features, context)
            state.update(state='running', warmup_seconds=time.perf_counter() - warm,
                         warmup_frame_count=warm_details['frame_count'])
            _publish_json(state_path, state)
            clip_reports = []
            for path, record, identity, timeline in preflight:
                _assert_benchmark_integrity(state, contract_path)
                report = measure_clip(path, record, backbone, heads, expected_pts=timeline.pts_seconds)
                if (_hash(path) != identity['video_sha256']
                        or backbone_state_sha256(backbone) != identity['backbone_state_sha256']):
                    raise ValueError('Media or frozen backbone changed during timing')
                _assert_benchmark_integrity(state, contract_path)
                report['video_sha256'] = identity['video_sha256']
                clip_reports.append(report)
            total_frames = sum(r['frame_count'] for r in clip_reports)
            result = {**state, 'state': 'completed', 'ended_at': _now(), 'partition': 'train-only-timing',
                      'clips': clip_reports, 'frame_count': total_frames,
                      'outer_test_evaluated': False, 'accuracy_metrics_computed': False,
                      'initialization_scope': 'contract/checkpoint admission, local backbone load, media/PTS preflight',
                      'warmup_scope': 'sixteen frames of first training clip plus both heads/contexts; excluded',
                      'stage_scope': 'CPU synchronized perf_counter; decode excludes RGB conversion; no disk cold-cache claim',
                      'limitations': ['Two training clips, one repetition; not a population latency guarantee',
                                      'Source media are re-encoded; not original-source or browser timestamps',
                                      'No minimum hardware, accelerator latency, event accuracy or coaching claim',
                                      'Composed variant times share one feature pass; whole-clip wall includes all variants']}
            result['head_sequence_timings'] = {
                name: {context: timing_summary([r['heads'][name][context]['head_seconds'] for r in clip_reports],
                                               sample_unit='one complete clip head call')
                       for context in CONTEXTS} for name in ('frame', 'temporal')}
            result['feature_wall_seconds_per_frame'] = sum(
                r['feature_extraction_wall_seconds'] for r in clip_reports) / total_frames
            _assert_benchmark_integrity(state, contract_path)
            _exclusive_json(output, result)
            published = output.stat()
            try:
                _assert_benchmark_integrity(state, contract_path)
            except BaseException:
                # Remove only this invocation's just-published inode, never another writer's file.
                if output.exists() and output.stat().st_ino == published.st_ino:
                    output.unlink()
                raise
            state.update(state='completed', ended_at=result['ended_at'], output_sha256=_hash(output))
            _publish_json(state_path, state)
            return result
        except BaseException as exc:
            state.update(state='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                         error=str(exc), ended_at=_now())
            _publish_json(state_path, state)
            raise
