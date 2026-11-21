"""Synthetic software checks only; no real checkpoint loads or model forwards."""
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

pytest.importorskip('torch')
from backend.swingsync.research import additional_evaluation as ev

REAL_LOAD_CLIP = ev._load_clip


@pytest.fixture
def harness(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.write_bytes(b'fixed synthetic source')
    labels = np.full(12, 8, dtype=np.int64)
    labels[[0, 1, 2, 3, 4, 5, 8, 11]] = np.arange(8)
    clip = SimpleNamespace(features=np.zeros((12, 1280), np.float32), labels=labels,
                           pts_seconds=np.array([0, .02, .05, .11, .19, .28, .41, .55,
                                                 .72, .90, 1.11, 1.35], np.float64))
    study = SimpleNamespace(watched={}, immutable={'six_selected_heads': 'synthetic-only'},
        training={f: {h: {'checkpoint_path': f'selected-{f}-{h}', 'checkpoint_sha256': 'a'*64}
                      for h in ev.HEADS} for f in (2, 3, 4)})
    items = [{'fold': f, 'clip_id': f, 'source_id': f'video-{f}'} for f in (2, 3, 4)]
    loads, calls = [], []
    monkeypatch.setattr(ev, 'admit_completed_training', lambda **kw: study)
    monkeypatch.setattr(ev, 'recheck_study', lambda study: None)
    monkeypatch.setattr(ev, '_preflight', lambda study, clips: (items, {str(source): ev.file_sha256(source)}))
    monkeypatch.setattr(ev, '_load_clip', lambda study, item: clip)
    monkeypatch.setattr(ev, '_sources', lambda: {str(source): ev.file_sha256(source)})
    monkeypatch.setattr(ev, '_runtime', lambda: {'synthetic': True})
    monkeypatch.setattr(ev, '_tensor_hash', lambda model: model)
    monkeypatch.setattr(ev, '_git_base_commit', lambda: 'synthetic-no-git')
    monkeypatch.setattr(ev, '_process_start', lambda pid: 'synthetic-start')
    def load(path, sha, head, hidden_size):
        assert hidden_size == 64 and path.startswith('selected-') and sha == 'a'*64
        loads.append((path, head))
        return head
    def predict(model, features, context):
        calls.append((model, context))
        assert features.dtype == np.float32 and features.shape == (12, 1280)
        return np.full((12, 9), np.float32(1/9), np.float32)
    monkeypatch.setattr(ev, 'load_selected_head', load)
    monkeypatch.setattr(ev, 'predict_probabilities', predict)
    args = dict(annotations='unused', manifests_dir='unused', audit_run='unused', smokes_dir='unused',
        training_runs_dir='unused', training_receipt='unused', expected_receipt_sha256='unused',
        clips_dir=tmp_path, run_dir=tmp_path / 'job', lock_path=tmp_path / 'expensive.lock')
    return SimpleNamespace(args=args, clip=clip, items=items, study=study, loads=loads,
                           calls=calls, source=source)


def test_bounded_resume_complete_and_noop(harness):
    h = harness
    partial = ev.run_additional_evaluation(**h.args, max_new_clips=1)
    assert partial['state'] == 'partial' and len(partial['clips']) == 1
    assert not (h.args['run_dir'] / 'report.json').exists()
    assert len(h.loads) == 2 and len(h.calls) == 4
    first = Path(partial['clips'][0]['path']).read_bytes()
    completed = ev.run_additional_evaluation(**h.args, resume=True, max_new_clips=2)
    assert completed['state'] == 'completed' and len(h.loads) == 6 and len(h.calls) == 12
    assert Path(partial['clips'][0]['path']).read_bytes() == first
    before = {p.name: p.read_bytes() for p in h.args['run_dir'].iterdir()}
    assert ev.run_additional_evaluation(**h.args, resume=True) == completed
    assert {p.name: p.read_bytes() for p in h.args['run_dir'].iterdir()} == before
    assert len(h.loads) == 6 and len(h.calls) == 12
    report = json.loads(before['report.json'])
    assert report['groups']['additional-three-folds']['frame__full_clip']['clip_count'] == 3
    assert report['normal_model_available'] is False and report['advice'] == []


def test_full_arrays_native_pts_and_official_ties(harness):
    state = ev.run_additional_evaluation(**harness.args, max_new_clips=1)
    row = state['clips'][0]
    with np.load(row['path'], allow_pickle=False) as saved:
        assert len(saved.files) == 6
        assert saved['pts_seconds'].dtype == np.float64
        assert np.array_equal(saved['pts_seconds'], harness.clip.pts_seconds)
        expected = int(np.argsort(np.full(12, np.float32(1/9)))[-1])
        for key in row['metrics']:
            assert saved[key].dtype == np.float32 and saved[key].shape == (12, 9)
            assert row['metrics'][key]['predicted_frames'] == [expected]*8
            assert row['metrics'][key]['signed_error_ms'][0] == pytest.approx(
                (harness.clip.pts_seconds[expected] - harness.clip.pts_seconds[0])*1000)


def test_raw_677_style_annotation_anomaly_no_repair():
    labels = np.full(80, 8, np.int64)
    raw = (2, 20, 10, 30, 40, 50, 60, 70)
    labels[list(raw)] = np.arange(8)
    scores = np.zeros((80, 9), np.float32)
    scores[:, 8] = 1
    targets = sorted(raw)
    # Target5-target0 = 48; np.round(48/30)=2; inclusive error exactly two.
    for event, frame in enumerate(targets):
        scores[frame+2] = 0
        scores[frame+2, event] = 1
    arrays = {ev._array_key(h, c): scores for h in ev.HEADS for c in ev.CONTEXTS}
    metrics = ev._metrics(arrays, ev.events_from_labels(labels), np.arange(80, dtype=np.float64)**1.2)
    for result in metrics.values():
        assert result['annotation_order_anomaly']
        assert result['annotation_event_frames'] == list(raw)
        assert result['event_frames'] == targets
        assert result['tolerance_frames'] == 2 and all(result['correct'])


def test_lock_prevents_admission(harness, monkeypatch):
    def forbidden(**kw):
        pytest.fail('Admission under occupied shared lock')
    monkeypatch.setattr(ev, 'admit_completed_training', forbidden)
    with harness.args['lock_path'].open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='shared expensive-work lock'):
            ev.run_additional_evaluation(**harness.args)
    assert not harness.args['run_dir'].exists()


@pytest.mark.parametrize('which', ['npz', 'ledger', 'report', 'unknown'])
def test_completed_drift_fails_without_forward_or_overwrite(harness, which):
    state = ev.run_additional_evaluation(**harness.args)
    root = harness.args['run_dir']
    if which == 'npz':
        Path(state['clips'][0]['path']).write_bytes(b'outside edit')
    elif which == 'report':
        (root / 'report.json').write_bytes(b'outside edit')
    elif which == 'ledger':
        state['clips'][0]['metrics']['frame__full_clip']['pce'] = .123
        (root / 'run.json').write_text(json.dumps(state))
    else:
        (root / 'unknown.npz').write_bytes(b'unknown orphan')
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(ValueError):
        ev.run_additional_evaluation(**harness.args, resume=True)
    assert len(harness.calls) == 12
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_known_orphan_recovered_without_repeating_forward(harness, monkeypatch):
    original = ev._publish
    def crash(path, raw, **kw):
        result = original(path, raw, **kw)
        if str(path).endswith('.npz'):
            raise RuntimeError('synthetic death after durable array')
        return result
    monkeypatch.setattr(ev, '_publish', crash)
    with pytest.raises(RuntimeError, match='synthetic death'):
        ev.run_additional_evaluation(**harness.args, max_new_clips=1)
    assert len(harness.calls) == 4
    monkeypatch.setattr(ev, '_publish', original)
    monkeypatch.setattr(ev, '_check_previous_process', lambda state: None)  # Simulate ended process.
    state = ev.run_additional_evaluation(**harness.args, resume=True, max_new_clips=2)
    assert state['state'] == 'completed' and len(harness.calls) == 12


def test_publication_input_drift_preserves_predecessor(harness, monkeypatch):
    predict = ev.predict_probabilities
    def drift(*args):
        result = predict(*args)
        harness.source.write_bytes(b'outside edit during forward')
        return result
    monkeypatch.setattr(ev, 'predict_probabilities', drift)
    with pytest.raises(ValueError, match='immutable input'):
        ev.run_additional_evaluation(**harness.args)
    ledger = json.loads((harness.args['run_dir'] / 'run.json').read_bytes())
    assert ledger['state'] == 'running' and ledger['clips'] == []
    assert not list(harness.args['run_dir'].glob('*.npz'))


def test_publication_compare_predecessor(tmp_path):
    path = tmp_path / 'output.json'
    path.write_bytes(b'outside edit')
    with pytest.raises(ValueError, match='predecessor'):
        ev._publish(path, b'new', expected='a'*64)
    assert path.read_bytes() == b'outside edit'
    with pytest.raises(ValueError, match='predecessor'):
        ev._publish(path, b'new')


def test_missing_any_of_six_heads_rejected_before_cache(tmp_path):
    for fold in (2, 3, 4):
        for head in ev.HEADS:
            study = SimpleNamespace(dataset=SimpleNamespace(records=[]),
                folds={f: SimpleNamespace(manifest={'partitions': {'test': {'clip_ids': list(range(350))}}})
                       for f in (2, 3, 4)},
                training={f: {h: {} for h in ev.HEADS} for f in (2, 3, 4)})
            del study.training[fold][head]
            # Each missing head is rejected before opening that fold's cache.
            if fold != 2:
                study.training[2] = {}  # Earlier malformed fold also fails closed.
            with pytest.raises(ValueError, match='six heads'):
                ev._preflight(study, tmp_path)


def test_json_large_ledger_and_duplicate_keys():
    value = {'large': 'x'*(3*1024*1024)}
    assert ev._read_state(json.dumps(value).encode()) == value
    for raw in (b'{"x": 1, "x": 2}', b'{"x":NaN}'):
        with pytest.raises(ValueError):
            ev._read_state(raw)


def test_cli_help_and_invalid_bounds_without_pythonpath(tmp_path):
    script = Path(ev.__file__).resolve().parents[3] / 'scripts/evaluate_additional_folds.py'
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0 and '--max-new-clips' in result.stdout
    args = [sys.executable, str(script)]
    for flag in ('annotations', 'manifests-dir', 'audit-run', 'smokes-dir', 'training-runs-dir',
                 'training-receipt', 'clips-dir', 'run-dir', 'expected-receipt-sha256'):
        args += ['--' + flag, 'missing']
    result = subprocess.run(args + ['--max-new-clips', '0'], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2 and 'must be positive' in result.stderr


def test_frozen_loader_uses_best_model_and_missing_checkpoint_is_unavailable(tmp_path):
    import torch
    from backend.swingsync.research import heldout
    from backend.swingsync.research.models import FrameClassifier
    config = json.loads((ev.ROOT / 'experiments/frame-v1.json').read_text())
    template = FrameClassifier().state_dict()
    selected = {k: torch.ones_like(v) for k, v in template.items()}
    last = {k: torch.zeros_like(v) for k, v in template.items()}
    history = [{'epoch': i, 'validation': {'pce': .8 if i == 3 else .1,
                'partition': 'inner-validation-only', 'outer_test_evaluated': False}}
               for i in range(1, 21)]
    saved = {'schema_version': 1, 'provenance': {'config': config},
             'progress': {'epoch': 20, 'cursor': 0, 'history': history,
                          'best_epoch': 3, 'best_validation_pce': .8},
             'model': last, 'best_model': selected}
    path = tmp_path / 'synthetic-checkpoint.pt'
    torch.save(saved, path)
    rng = torch.get_rng_state().clone()
    model = heldout.load_selected_head(path, ev.file_sha256(path), 'frame', hidden_size=64)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(v, selected[k]) for k, v in model.state_dict().items())
    assert any(not torch.equal(v, last[k]) for k, v in model.state_dict().items())
    path.unlink()
    with pytest.raises(FileNotFoundError):
        heldout.load_selected_head(path, 'a'*64, 'frame', hidden_size=64)


def test_outer_preflight_authenticates_all1050_and_rejects_each_missing_head(tmp_path, monkeypatch):
    records = [SimpleNamespace(clip_id=i, source_id=f'source{i}', outer_split=i//350+2)
               for i in range(1050)]
    report = {'clips': {str(r.clip_id): {'official_outer_split': r.outer_split,
               'cache_path': str(tmp_path / f'{r.clip_id}.npz'), 'cache_sha256': 'a'*64,
               'source_video_sha256': 'b'*64, 'cache_key': 'c'*64, 'native_pts_sha256': 'd'*64}
               for r in records}}
    study = SimpleNamespace(dataset=SimpleNamespace(records=records), audit_report=report,
             folds={f: SimpleNamespace(manifest={'partitions': {'test': {
                 'clip_ids': list(range((f-2)*350, (f-1)*350))}}}) for f in (2, 3, 4)},
             training={f: {h: {} for h in ev.HEADS} for f in (2, 3, 4)})
    seen = []
    monkeypatch.setattr(ev, '_load_clip', lambda study, item: seen.append(item))
    items, watched = ev._preflight(study, tmp_path)
    assert len(items) == 1050 and len(seen) == 1050 and len(watched) == 2100
    for fold in (2, 3, 4):
        for head in ev.HEADS:
            removed = study.training[fold].pop(head)
            with pytest.raises(ValueError, match='six heads'):
                ev._preflight(study, tmp_path)
            study.training[fold][head] = removed


def test_native_pts_hash_and_dtype_fail_closed(harness, monkeypatch):
    h = harness
    record = SimpleNamespace(clip_id=2)
    study = SimpleNamespace(dataset=SimpleNamespace(records=[record], sha256='a'*64),
                            folds={2: SimpleNamespace(feature_pipeline={})})
    item = {'clip_id': 2, 'fold': 2, 'cache_path': str(h.source), 'source_path': str(h.source),
            'source_sha256': 'b'*64, 'cache_sha256': 'c'*64, 'cache_key': 'd'*64,
            'native_pts_sha256': hashlib.sha256(h.clip.pts_seconds.tobytes()).hexdigest()}
    monkeypatch.setattr(ev, '_check_clip', lambda *args: h.clip)
    assert REAL_LOAD_CLIP(study, item) is h.clip
    item['native_pts_sha256'] = 'e'*64
    with pytest.raises(ValueError, match='PTS'):
        REAL_LOAD_CLIP(study, item)


def test_in_memory_selected_model_drift_refuses_artifact(harness, monkeypatch):
    count = 0
    def tensor_hash(model):
        nonlocal count
        count += 1
        return 'original' if count < 4 else 'changed'
    monkeypatch.setattr(ev, '_tensor_hash', tensor_hash)
    with pytest.raises(ValueError, match='in-memory model changed'):
        ev.run_additional_evaluation(**harness.args)
    assert not list(harness.args['run_dir'].glob('*.npz'))


@pytest.mark.parametrize('change', ['restored-bytes', 'replacement', 'symlink'])
def test_stat_pins_reject_same_bytes_after_any_write(tmp_path, change):
    path = tmp_path / 'immutable'
    path.write_bytes(b'original')
    before = path.stat()
    pins = ev._PinnedFiles({str(path): ev.file_sha256(path)})
    if change == 'restored-bytes':
        path.write_bytes(b'outside edit')
        path.write_bytes(b'original')
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert path.stat().st_mtime_ns == before.st_mtime_ns
    elif change == 'replacement':
        other = tmp_path / 'replacement'
        other.write_bytes(b'original')
        os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
        other.replace(path)
    else:
        target = tmp_path / 'target'
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValueError, match='stat identity|regular file'):
        pins.check()


def test_initial_hash_is_bracketed_by_stat_identities(tmp_path, monkeypatch):
    path = tmp_path / 'immutable'
    path.write_bytes(b'original')
    digest = ev.file_sha256(path)
    sha = ev.file_sha256
    def rewrite_during_hash(p):
        result = sha(p)
        Path(p).write_bytes(b'original')
        return result
    monkeypatch.setattr(ev, 'file_sha256', rewrite_during_hash)
    with pytest.raises(ValueError, match='during initial byte authentication'):
        ev._PinnedFiles({str(path): digest})


def test_forwards_use_stat_guards_and_full_sha_at_boundaries(harness, monkeypatch):
    checks = []
    monkeypatch.setattr(ev, 'recheck_study', lambda study: checks.append('full'))
    ev.run_additional_evaluation(**harness.args, max_new_clips=1)
    assert len(harness.calls) == 4 and checks == ['full', 'full']
    ev.run_additional_evaluation(**harness.args, resume=True, max_new_clips=2)
    assert len(harness.calls) == 12 and len(checks) == 4
    ev.run_additional_evaluation(**harness.args, resume=True)
    assert len(harness.calls) == 12 and len(checks) == 5


def test_write_then_restore_inside_forward_fails_before_publication(harness, monkeypatch):
    predict = ev.predict_probabilities
    def restored(*args):
        result = predict(*args)
        before = harness.source.stat()
        raw = harness.source.read_bytes()
        harness.source.write_bytes(b'changed')
        harness.source.write_bytes(raw)
        os.utime(harness.source, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result
    monkeypatch.setattr(ev, 'predict_probabilities', restored)
    with pytest.raises(ValueError, match='stat identity changed'):
        ev.run_additional_evaluation(**harness.args)
    state = json.loads((harness.args['run_dir'] / 'run.json').read_bytes())
    assert state['in_flight'] and not state['clips']
    assert not list(harness.args['run_dir'].glob('*.npz'))


def test_crash_after_first_forward_never_repeats_unknown_predictions(harness, monkeypatch):
    predict = ev.predict_probabilities
    def crash(*args):
        predict(*args)
        raise RuntimeError('synthetic death during first forward')
    monkeypatch.setattr(ev, 'predict_probabilities', crash)
    with pytest.raises(RuntimeError, match='first forward'):
        ev.run_additional_evaluation(**harness.args)
    state_path = harness.args['run_dir'] / 'run.json'
    before = state_path.read_bytes()
    assert json.loads(before)['in_flight'] is not None
    monkeypatch.setattr(ev, 'predict_probabilities', predict)
    monkeypatch.setattr(ev, '_check_previous_process', lambda state: None)
    with pytest.raises(ValueError, match='started without known durable probabilities'):
        ev.run_additional_evaluation(**harness.args, resume=True)
    assert len(harness.calls) == 1 and state_path.read_bytes() == before


@pytest.mark.parametrize('when', ['before', 'after'])
def test_deterministic_report_orphan_recovers_without_forward(harness, monkeypatch, when):
    publish = ev._publish
    def crash(path, raw, **kw):
        is_report = Path(path).name == 'report.json'
        if is_report and when == 'before':
            raise RuntimeError('synthetic death before report publication')
        result = publish(path, raw, **kw)
        if is_report:
            raise RuntimeError('synthetic death after report publication')
        return result
    monkeypatch.setattr(ev, '_publish', crash)
    with pytest.raises(RuntimeError, match='report publication'):
        ev.run_additional_evaluation(**harness.args)
    assert len(harness.calls) == 12
    report_path = harness.args['run_dir'] / 'report.json'
    before = (report_path.read_bytes(), ev._stat_identity(report_path)) if report_path.exists() else None
    monkeypatch.setattr(ev, '_publish', publish)
    monkeypatch.setattr(ev, '_check_previous_process', lambda state: None)
    state = ev.run_additional_evaluation(**harness.args, resume=True)
    assert state['state'] == 'completed' and state['pending_report'] is None
    assert len(harness.calls) == 12 and len(harness.loads) == 6
    if before is not None:
        assert (report_path.read_bytes(), ev._stat_identity(report_path)) == before


@pytest.mark.parametrize('outside_ledger_edit', [False, True])
def test_drift_after_final_publication_retracts_only_owned_successor(harness, monkeypatch, outside_ledger_edit):
    publish = ev._publish
    changed = False
    def drift_after_final(path, raw, **kw):
        nonlocal changed
        result = publish(path, raw, **kw)
        if Path(path).name == 'run.json' and json.loads(raw)['state'] == 'completed' and not changed:
            changed = True
            harness.source.write_bytes(b'outside input edit after final publication')
            if outside_ledger_edit:
                Path(path).write_bytes(b'outside ledger edit must survive')
        return result
    monkeypatch.setattr(ev, '_publish', drift_after_final)
    with pytest.raises(ValueError, match='stat identity'):
        ev.run_additional_evaluation(**harness.args)
    raw = (harness.args['run_dir'] / 'run.json').read_bytes()
    if outside_ledger_edit:
        assert raw == b'outside ledger edit must survive'
    else:
        saved = json.loads(raw)
        assert saved['state'] == 'running' and saved['pending_report'] is not None
