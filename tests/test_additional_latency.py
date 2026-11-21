"""Synthetic byte/clock fixtures verify timing software, never hardware latency.

No real video, ImageNet weight, selected checkpoint, training or model forward is
used. Admission and numerical calls explicitly return declared synthetic values.
"""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from backend.swingsync.research import additional_latency as timing  # noqa: E402


class Clock:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return float(self.count)


def read(path):
    return json.loads(Path(path).read_text())


def measured(item):
    pts, record = item['pts'], item['record']
    n = len(pts)
    batches = [16] * (n // 16) + ([n % 16] if n % 16 else [])
    return {'frame_count': n, 'clip_id': record.clip_id, 'source_id': record.source_id,
            'timestamp_provenance': 'reencoded-clip-pts',
            'pts_sha256': hashlib.sha256(pts.tobytes()).hexdigest(),
            'first_pts_seconds': float(pts[0]), 'last_pts_seconds': float(pts[-1]),
            'feature_extraction_wall_seconds': 5., 'measured_clip_wall_seconds': 13.,
            'batch_frame_counts': batches,
            **{stage: timing.latency.timing_summary([1.] * count, sample_unit='SYNTHETIC clock unit')
               for stage, count in [('decode', n), ('preprocessing', n), ('backbone', len(batches))]},
            'heads': {name: {context: {'head_seconds': 2., 'composed_video_pipeline_seconds': 7.}
                             for context in timing.CONTEXTS} for name in ('frame', 'temporal')}}


@pytest.fixture
def ready(tmp_path, monkeypatch):
    runtime_state = {'synthetic': True, 'threads': 1, 'deterministic': True}
    dataset = SimpleNamespace(sha256='d' * 64, records=[])
    protocol = {'evaluation': {'latency_batch_size': 16, 'latency_warmup_frames': 16,
                              'latency_repetitions': 1, 'latency_contexts': list(timing.CONTEXTS),
                              'heads': ['frame', 'temporal'],
                              'latency_partition': 'two-fixed-training-clips-per-fold'}, 'folds': {}}
    clips = tmp_path / 'SYNTHETIC-clips'
    clips.mkdir()
    by_id, rows, cache_by_path = {}, {}, {}
    for clip_id in (0, 1, 6, 7):
        record = SimpleNamespace(clip_id=clip_id, source_id=f'SYNTHETIC-source-{clip_id}', expected_frame_count=17)
        dataset.records.append(record)
        pts = np.cumsum(np.array([.021 + i * .0003 for i in range(17)], dtype=np.float64))
        source = clips / f'{clip_id}.mp4'
        source.write_bytes(f'SYNTHETIC video-byte witness {clip_id}; no decoder'.encode())
        cache = tmp_path / f'SYNTHETIC-cache-{clip_id}.npz'
        cache.write_bytes(f'SYNTHETIC cache-byte witness {clip_id}; no arrays'.encode())
        identity = {'clip_id': clip_id, 'source_id': record.source_id, 'annotation_sha256': dataset.sha256,
                    'expected_frame_count': 17, 'timestamp_provenance': 'reencoded-clip-pts',
                    'backbone_state_sha256': 'b' * 64, 'video_sha256': timing.file_sha256(source)}
        arrays = SimpleNamespace(features=np.zeros((17, 1280), np.float32), pts_seconds=pts,
                                 metadata={'identity': identity})
        cache_by_path[str(cache)] = arrays
        rows[str(clip_id)] = {'source_video_sha256': identity['video_sha256'],
                             'cache_sha256': timing.file_sha256(cache),
                             'native_pts_sha256': hashlib.sha256(pts.tobytes()).hexdigest()}
        by_id[clip_id] = {'path': source, 'record': record, 'identity': identity,
                          'pts': pts, 'cache_path': cache}
    folds, training, checkpoint_to_head = {}, {}, {}
    for fold, ids in timing.LATENCY_IDS.items():
        protocol['folds'][str(fold)] = {'latency_training_clip_ids': ids.copy()}
        folds[fold] = SimpleNamespace(manifest={'partitions': {'train': {'clip_ids': ids + [20, 21]},
                                                              'test': {'clip_ids': [90, 91]}}},
                                      cache_paths={i: by_id[i]['cache_path'] for i in ids},
                                      feature_pipeline={'backbone_state_sha256': 'b' * 64})
        training[fold] = {}
        for name in ('frame', 'temporal'):
            path = tmp_path / f'SYNTHETIC-selected-{fold}-{name}.pt'
            path.write_bytes(f'SYNTHETIC checkpoint-byte witness {fold} {name}'.encode())
            head = SimpleNamespace(name=name, fold=fold, selected='best_model', tensor_sha=f'{fold}-{name}')
            checkpoint_to_head[str(path)] = head
            training[fold][name] = {'checkpoint_path': str(path), 'checkpoint_sha256': timing.file_sha256(path),
                                    'config': SimpleNamespace(hidden_size=64), 'ledger': {'selected': 'best_model'}}
    study = SimpleNamespace(dataset=dataset, protocol=protocol, folds=folds, training=training,
                            audit_report={'clips': rows}, immutable={'synthetic': True, 'validated_for_serving': False})
    source = tmp_path / 'SYNTHETIC-source.py'
    source.write_text('SYNTHETIC source witness')
    weight = tmp_path / 'SYNTHETIC-ImageNet-weight.pth'
    weight.write_bytes(b'SYNTHETIC non-weight bytes')
    calls = {'admission': [], 'selected': [], 'warmup': [], 'predictions': [], 'measured': [], 'backbone': []}
    def admit(**kwargs):
        calls['admission'].append(kwargs)
        return study
    def load(path, digest, model, hidden):
        assert timing.file_sha256(path) == digest
        assert hidden == 64
        head = checkpoint_to_head[str(path)]
        assert head.selected == 'best_model' and head.name == model
        assert read(tmp_path / 'run/run.json')['immutable']['study'] == study.immutable
        calls['selected'].append((head.fold, model, head.selected))
        return head
    def warmup(path, record, backbone, **kwargs):
        assert kwargs['batch_size'] == kwargs['limit'] == 16
        calls['warmup'].append(record.clip_id)
        return np.zeros((16, 1280), np.float32), by_id[record.clip_id]['pts'][:16].copy(), {'frame_count': 16}
    def predict(head, features, context):
        calls['predictions'].append((head.fold, head.name, len(features), context))
        return np.ones((len(features), 9), np.float64) / 9
    def measure(path, record, backbone, heads, **kwargs):
        assert kwargs['batch_size'] == 16 and kwargs['contexts'] == timing.CONTEXTS
        np.testing.assert_array_equal(kwargs['expected_pts'], by_id[record.clip_id]['pts'])
        calls['measured'].append((heads['frame'].fold, record.clip_id))
        return measured(by_id[record.clip_id])
    def backbone():
        calls['backbone'].append(True)
        assert read(tmp_path / 'run/run.json')['immutable']['runtime'] == runtime_state
        return object(), weight
    monkeypatch.setattr(timing, 'admit_completed_training', admit)
    monkeypatch.setattr(timing, 'recheck_study', lambda _: None)
    monkeypatch.setattr(timing, '_runtime', lambda: deepcopy(runtime_state))
    monkeypatch.setattr(timing, '_sources', lambda: {str(source): timing.file_sha256(source)})
    monkeypatch.setattr(timing, '_weight_path', lambda: weight)
    monkeypatch.setattr(timing, 'WEIGHT_SHA256', timing.file_sha256(weight))
    monkeypatch.setattr(timing, '_git_base_commit', lambda: 'SYNTHETIC-no-Git-operation')
    monkeypatch.setattr(timing, '_process_start', lambda _: 'SYNTHETIC-process')
    monkeypatch.setattr(timing.latency, '_hardware', lambda: {'synthetic': True})
    monkeypatch.setattr(timing.latency, 'detect_devices', lambda: ['SYNTHETIC-CPU'])
    monkeypatch.setattr(timing.latency, '_load_local_backbone', backbone)
    monkeypatch.setattr(timing, 'load_feature_cache', lambda p: cache_by_path[str(p)])
    monkeypatch.setattr(timing, 'inspect_clip_timeline', lambda p, r, **k: SimpleNamespace(
        timestamp_provenance='reencoded-clip-pts', pts_seconds=by_id[r.clip_id]['pts'].copy()))
    monkeypatch.setattr(timing, 'build_cache_identity', lambda p, r, *a, **k: deepcopy(by_id[r.clip_id]['identity']))
    monkeypatch.setattr(timing, 'load_selected_head', load)
    monkeypatch.setattr(timing, '_tensor_hash', lambda m: m.tensor_sha)
    monkeypatch.setattr(timing, 'backbone_state_sha256', lambda _: 'b' * 64)
    monkeypatch.setattr(timing.latency, '_extract_timed', warmup)
    monkeypatch.setattr(timing, 'predict_probabilities', predict)
    monkeypatch.setattr(timing.latency, 'measure_clip', measure)
    options = {name: tmp_path / f'SYNTHETIC-{name}' for name in (
        'annotations', 'manifests_dir', 'audit_run', 'smokes_dir', 'training_runs_dir', 'training_receipt')}
    options.update(expected_receipt_sha256='a' * 64, clips_dir=clips, run_dir=tmp_path / 'run',
                   lock_path=tmp_path / 'shared.lock', clock=Clock())
    return SimpleNamespace(options=options, study=study, calls=calls, by_id=by_id,
                           cache=cache_by_path, source=source, weight=weight, runtime=runtime_state)


def test_complete_one_shot_retains_registered_scope_pts_selection_and_closed_gates(ready):
    result = timing.run_additional_latency(**ready.options)
    state = read(ready.options['run_dir'] / 'run.json')
    path = ready.options['run_dir'] / 'latency.json'
    assert state['state'] == result['state'] == 'completed'
    assert state['report_sha256'] == timing.file_sha256(path)
    assert result['completed_clip_observations'] == 6 and result['partition'] == 'train-only-timing'
    assert ready.calls['measured'] == [(2, 0), (2, 1), (3, 6), (3, 7), (4, 0), (4, 1)]
    assert ready.calls['warmup'] == [0, 6, 0]
    assert ready.calls['selected'] == [(f, n, 'best_model') for f in (2, 3, 4) for n in ('frame', 'temporal')]
    assert len(ready.calls['predictions']) == 12 and ready.calls['backbone'] == [True]
    for key, expected in timing.CLOSED_GATES.items():
        assert state[key] == result[key] == expected
    for fold in result['folds'].values():
        assert fold['warmup_frames'] == 16
        assert fold['feature_wall_seconds_per_frame'] == pytest.approx(5 / 17)
        for row in fold['clips']:
            assert row['batch_frame_counts'] == [16, 1]
            assert row['first_pts_seconds'] != 0
            for head in row['heads'].values():
                for context in head.values():
                    assert context['composed_video_pipeline_seconds'] == 7.
        assert fold['head_sequence_timings']['frame']['full-clip']['sample_count'] == 2
    assert ready.calls['admission'][0]['expected_receipt_sha256'] == 'a' * 64
    before = {p.name: p.read_bytes() for p in ready.options['run_dir'].iterdir()}
    with pytest.raises(ValueError, match='one-shot'):
        timing.run_additional_latency(**ready.options)
    assert before == {p.name: p.read_bytes() for p in ready.options['run_dir'].iterdir()}


@pytest.mark.parametrize('edit', [
    lambda s: s.protocol['evaluation'].update(latency_repetitions=2),
    lambda s: s.protocol['evaluation'].update(latency_batch_size=True),
    lambda s: s.protocol['evaluation'].update(latency_warmup_frames=15),
    lambda s: s.protocol['evaluation'].update(latency_contexts=['full-clip']),
    lambda s: s.protocol['folds']['3'].update(latency_training_clip_ids=[0, 1]),
    lambda s: s.folds[2].manifest['partitions']['train'].update(clip_ids=[1, 2, 20]),
    lambda s: s.training[4].pop('temporal'),
    lambda s: s.folds.pop(3),
])
def test_unregistered_scope_refused_before_any_model(ready, edit):
    edit(ready.study)
    with pytest.raises(ValueError, match='registered|TRAIN'):
        timing.run_additional_latency(**ready.options)
    assert ready.calls['backbone'] == ready.calls['selected'] == []
    state = read(ready.options['run_dir'] / 'run.json')
    assert state['state'] == 'failed' and state['completed_clip_observations'] == 0
    assert not (ready.options['run_dir'] / 'latency.json').exists()


@pytest.mark.parametrize('kind', ['missing', 'wrong'])
def test_weight_absence_or_wrong_bytes_never_construct_or_download(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path))
    monkeypatch.setattr(torch.hub, 'download_url_to_file', lambda *a, **k: pytest.fail('Network download'))
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', lambda *a, **k: pytest.fail('Network weight load'))
    if kind == 'wrong':
        directory = tmp_path / 'checkpoints'
        directory.mkdir()
        (directory / 'mobilenet_v2-7ebf99e0.pth').write_bytes(b'SYNTHETIC wrong weight')
    with pytest.raises(timing.LocalWeightsUnavailable, match='no .*download'):
        timing._weight_path()


def test_unavailable_weight_has_durable_unavailable_ledger_without_model(ready, monkeypatch):
    def unavailable():
        raise timing.LocalWeightsUnavailable('SYNTHETIC absent, no download')
    monkeypatch.setattr(timing, '_weight_path', unavailable)
    with pytest.raises(timing.LocalWeightsUnavailable):
        timing.run_additional_latency(**ready.options)
    assert read(ready.options['run_dir'] / 'run.json')['state'] == 'unavailable'
    assert ready.calls['backbone'] == []


@pytest.mark.parametrize('kind', ['video-bytes', 'cache-bytes', 'pts', 'pts-dtype', 'features', 'identity', 'timeline'])
def test_source_cache_native_pts_and_pipeline_preflight_fail_closed(ready, monkeypatch, kind):
    item = ready.by_id[0]
    arrays = ready.cache[str(item['cache_path'])]
    if kind == 'video-bytes':
        item['path'].write_bytes(b'changed')
    elif kind == 'cache-bytes':
        item['cache_path'].write_bytes(b'changed')
    elif kind == 'pts':
        arrays.pts_seconds = np.arange(17, dtype=np.float64) / 30
    elif kind == 'pts-dtype':
        arrays.pts_seconds = arrays.pts_seconds.astype(np.float32)
    elif kind == 'features':
        arrays.features = arrays.features[:16]
    elif kind == 'identity':
        arrays.metadata['identity']['timestamp_provenance'] = 'fps-reconstruction'
    else:
        monkeypatch.setattr(timing, 'inspect_clip_timeline', lambda *a, **k: SimpleNamespace(
            timestamp_provenance='reencoded-clip-pts', pts_seconds=np.arange(17, dtype=np.float64)))
    with pytest.raises(ValueError, match='source|cache|timestamps'):
        timing.run_additional_latency(**ready.options)
    assert ready.calls['backbone'] == []


@pytest.mark.parametrize('edit', [
    lambda r: r.update(frame_count=True), lambda r: r.update(pts_sha256='0' * 64),
    lambda r: r.update(first_pts_seconds=0), lambda r: r.update(batch_frame_counts=[17]),
    lambda r: r['decode'].update(sample_count=16), lambda r: r['backbone'].update(total_seconds=-1),
    lambda r: r.update(feature_extraction_wall_seconds=float('nan')),
    lambda r: r['heads']['frame'].pop('reference-64-reset'),
    lambda r: r['heads']['temporal']['full-clip'].update(composed_video_pipeline_seconds=9),
])
def test_invalid_timing_measurement_refuses_completion(ready, monkeypatch, edit):
    def measure(path, record, *a, **k):
        report = measured(ready.by_id[record.clip_id])
        edit(report)
        return report
    monkeypatch.setattr(timing.latency, 'measure_clip', measure)
    with pytest.raises(ValueError, match='Timed'):
        timing.run_additional_latency(**ready.options)
    assert read(ready.options['run_dir'] / 'run.json')['state'] == 'failed'
    assert not (ready.options['run_dir'] / 'latency.json').exists()


@pytest.mark.parametrize('kind', ['warmup-pts', 'warmup-length', 'warmup-output', 'backbone', 'head'])
def test_warmup_and_frozen_selection_drift_refuse_completion(ready, monkeypatch, kind):
    if kind.startswith('warmup-'):
        if kind == 'warmup-output':
            monkeypatch.setattr(timing, 'predict_probabilities', lambda *a: np.zeros((15, 9)))
        else:
            def warmup(path, record, *a, **k):
                count = 15 if kind == 'warmup-length' else 16
                pts = ready.by_id[record.clip_id]['pts'][:count].copy()
                if kind == 'warmup-pts':
                    pts += .01
                return np.zeros((count, 1280), np.float32), pts, {'frame_count': count}
            monkeypatch.setattr(timing.latency, '_extract_timed', warmup)
    elif kind == 'backbone':
        monkeypatch.setattr(timing, 'backbone_state_sha256', lambda _: 'changed')
    else:
        original = timing.latency.measure_clip
        def mutate(path, record, backbone, heads, **kwargs):
            report = original(path, record, backbone, heads, **kwargs)
            heads['frame'].tensor_sha = 'changed'
            return report
        monkeypatch.setattr(timing.latency, 'measure_clip', mutate)
    with pytest.raises(ValueError, match='Warmup|changed during timing'):
        timing.run_additional_latency(**ready.options)
    assert not (ready.options['run_dir'] / 'latency.json').exists()


@pytest.mark.parametrize('kind', ['runtime', 'source', 'video', 'checkpoint', 'interrupted'])
def test_midrun_drift_or_interruption_preserves_closed_failed_evidence(ready, monkeypatch, kind):
    original = timing.latency.measure_clip
    def altered(path, record, backbone, heads, **kwargs):
        report = original(path, record, backbone, heads, **kwargs)
        if kind == 'runtime':
            ready.runtime['threads'] = 2
        elif kind == 'source':
            ready.source.write_text('changed')
        elif kind == 'video':
            ready.by_id[record.clip_id]['path'].write_bytes(b'changed')
        elif kind == 'checkpoint':
            monkeypatch.setattr(timing, 'recheck_study', lambda _: (_ for _ in ()).throw(ValueError('SYNTHETIC checkpoint changed')))
        else:
            raise KeyboardInterrupt('SYNTHETIC interruption')
        return report
    monkeypatch.setattr(timing.latency, 'measure_clip', altered)
    with pytest.raises((ValueError, KeyboardInterrupt)):
        timing.run_additional_latency(**ready.options)
    state = read(ready.options['run_dir'] / 'run.json')
    assert state['state'] == ('interrupted' if kind == 'interrupted' else 'failed')
    assert state['report_sha256'] is None and state['validated_for_serving'] is False
    assert not (ready.options['run_dir'] / 'latency.json').exists()


def test_external_predecessor_is_preserved_with_separate_conflict_witness(ready, monkeypatch):
    original = timing.latency.measure_clip
    external = b'{"external": "SYNTHETIC changed predecessor"}\n'
    def changed(path, record, backbone, heads, **kwargs):
        report = original(path, record, backbone, heads, **kwargs)
        (ready.options['run_dir'] / 'run.json').write_bytes(external)
        return report
    monkeypatch.setattr(timing.latency, 'measure_clip', changed)
    with pytest.raises(ValueError, match='predecessor'):
        timing.run_additional_latency(**ready.options)
    assert (ready.options['run_dir'] / 'run.json').read_bytes() == external
    conflicts = list(ready.options['run_dir'].glob('run-conflict-*.json'))
    assert len(conflicts) == 1 and read(conflicts[0])['state'] == 'failed'


def test_busy_shared_lock_refuses_before_output_creation(ready):
    with ready.options['lock_path'].open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='shared expensive-work lock'):
            timing.run_additional_latency(**ready.options)
    assert not ready.options['run_dir'].exists()


def test_source_bindings_include_direct_study_admission():
    names = {Path(p).name for p in timing._sources()}
    assert 'additional_study.py' in names and 'additional_latency.py' in names
    assert 'benchmark_additional_folds.py' in names


def test_cli_help_runs_from_outside_repo_without_pythonpath(tmp_path):
    script = Path(timing.__file__).resolve().parents[3] / 'scripts/benchmark_additional_folds.py'
    environment = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and '--expected-receipt-sha256' in result.stdout
    assert '--training-receipt' in result.stdout


@pytest.mark.parametrize('kind', ['corrupt-own', 'external-replacement'])
def test_failed_report_publication_removes_only_owned_inode(ready, monkeypatch, kind):
    original = timing._write
    external = b'{"external": "SYNTHETIC replacement report"}\n'
    def altered(path, value, **kwargs):
        result = original(path, value, **kwargs)
        if Path(path).name == 'latency.json':
            if kind == 'external-replacement':
                Path(path).unlink()
                replacement = Path(path).with_name('external-report.tmp')
                replacement.write_bytes(external)
                replacement.replace(path)
            else:
                Path(path).write_bytes(b'SYNTHETIC corrupted own report')
        return result
    monkeypatch.setattr(timing, '_write', altered)
    with pytest.raises(ValueError, match='report changed'):
        timing.run_additional_latency(**ready.options)
    path = ready.options['run_dir'] / 'latency.json'
    if kind == 'external-replacement':
        assert path.read_bytes() == external
    else:
        assert not path.exists()
    assert read(ready.options['run_dir'] / 'run.json')['state'] == 'failed'


def test_training_admission_failure_is_recorded_before_model_construction(ready, monkeypatch):
    def refuse(**kwargs):
        raise ValueError('SYNTHETIC untrusted 017 controller; not 018 completion')
    monkeypatch.setattr(timing, 'admit_completed_training', refuse)
    with pytest.raises(ValueError, match='not 018 completion'):
        timing.run_additional_latency(**ready.options)
    state = read(ready.options['run_dir'] / 'run.json')
    assert state['state'] == 'failed' and state['completed_clip_observations'] == 0
    assert ready.calls['backbone'] == ready.calls['selected'] == []


def test_deleted_predecessor_is_not_recreated_and_conflict_witness_is_saved(ready, monkeypatch):
    original = timing.latency.measure_clip
    def removed(path, record, backbone, heads, **kwargs):
        report = original(path, record, backbone, heads, **kwargs)
        (ready.options['run_dir'] / 'run.json').unlink()
        return report
    monkeypatch.setattr(timing.latency, 'measure_clip', removed)
    with pytest.raises(ValueError, match='predecessor'):
        timing.run_additional_latency(**ready.options)
    assert not (ready.options['run_dir'] / 'run.json').exists()
    conflicts = list(ready.options['run_dir'].glob('run-conflict-*.json'))
    assert len(conflicts) == 1 and read(conflicts[0])['state'] == 'failed'
