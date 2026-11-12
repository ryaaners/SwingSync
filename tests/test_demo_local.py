"""Original synthetic software reproduction, never golf/model evidence."""
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest

from scripts.demo_local import (OwnedHTTP, SCOPE, TICKS, create_fixture, frontend_fingerprints,
                                run_demo, sha)


@pytest.fixture
def frontend(tmp_path):
    root = tmp_path/'built'
    (root/'assets').mkdir(parents=True)
    (root/'index.html').write_text('<!doctype html><html><body>Software fixture<script type="module" src="/assets/demo.js"></script></body></html>')
    (root/'assets/demo.js').write_text('console.log("synthetic built frontend fixture");')
    return root


def test_original_h264_fixture_retains_nonzero_variable_native_pts(tmp_path):
    path = tmp_path/'synthetic.mp4'
    rows, images = create_fixture(path)
    assert len(rows) == len(images) == 9
    assert [r['seconds'] for r in rows] == [t/1000 for t in TICKS]
    assert rows[0]['seconds'] == 5 and len(set(np.diff([r['seconds'] for r in rows]))) > 1
    assert all(image.shape == (200, 320, 3) for image in images)
    assert all(np.count_nonzero(image[16:44, 16:232] > 200) > 100 for image in images)
    assert not np.array_equal(images[0], images[-1])
    assert 'no golf' in SCOPE


def test_real_owned_http_demo_corrections_restart_report_and_deletion(tmp_path, frontend):
    output = tmp_path/'reproduction'
    result = run_demo(output, frontend)
    assert result['state'] == 'completed'
    assert result['training_performed'] is result['inference_performed'] is result['accuracy_measured'] is False
    assert result['normal_model_available'] is False and result['advice'] == []
    assert result['checks']['manual_revisions'] == 2
    assert result['checks']['report_pngs_exact_rgb'] == 3
    assert result['checks']['stale_save_status'] == 409
    assert result['checks']['restart_preserved'] is result['checks']['second_upload_deleted'] is True
    assert len(result['server_attempts']) == 2
    for attempt in result['server_attempts']:
        assert attempt['state'] == 'stopped' and type(attempt['returncode']) is int
        with pytest.raises(ProcessLookupError):
            os.kill(attempt['pid'], 0)
    for name, digest in result['outputs_sha256'].items():
        assert sha(output/name) == digest
    assert json.loads((output/'run.json').read_text()) == result
    assert len(list((output/'storage').glob('*.mp4'))) == 1
    with pytest.raises(FileExistsError):
        run_demo(output, frontend)
    assert json.loads((output/'run.json').read_text()) == result


def test_failure_stops_only_owned_child_and_retains_failed_ledger(tmp_path, frontend, monkeypatch):
    from scripts import demo_local
    def interrupted(*args):
        raise ValueError('deliberate software test failure after startup')
    monkeypatch.setattr(demo_local, 'upload_and_wait', interrupted)
    output = tmp_path/'failed'
    with pytest.raises(ValueError, match='deliberate'):
        run_demo(output, frontend)
    result = json.loads((output/'run.json').read_text())
    assert result['state'] == 'failed' and len(result['server_attempts']) == 1
    child = result['server_attempts'][0]
    assert child['state'] == 'stopped' and type(child['returncode']) is int
    with pytest.raises(ProcessLookupError):
        os.kill(child['pid'], 0)
    assert not (output/'report.html').exists()


def test_missing_or_symlink_frontend_is_refused_without_server(tmp_path, frontend):
    output = tmp_path/'failed-build'
    with pytest.raises(ValueError, match='Build the frontend'):
        run_demo(output, tmp_path/'missing')
    assert json.loads((output/'run.json').read_text())['server_attempts'] == []
    outside = tmp_path/'private.txt'
    outside.write_text('not for serving')
    (frontend/'leak').symlink_to(outside)
    with pytest.raises(ValueError, match='symlinks'):
        frontend_fingerprints(frontend)


def test_http_requests_refuse_foreign_paths_dead_child_and_expired_budget():
    class Process:
        def poll(self):
            return 0
    with httpx.Client(base_url='http://127.0.0.1:1', trust_env=False) as client:
        owned = OwnedHTTP(Process(), client, float('inf'))
        for path in ('http://example.com', '//example.com/api'):
            with pytest.raises(ValueError, match='relative'):
                owned.request('GET', path)
        with pytest.raises(ValueError, match='stopped'):
            owned.request('GET', '/api/health')
        Process.poll = lambda self: None
        owned.deadline = 0
        with pytest.raises(ValueError, match='budget'):
            owned.request('GET', '/api/health')


def test_cli_has_no_external_service_or_existing_output_override():
    script = (Path(__file__).resolve().parents[1]/'scripts/demo_local.py').read_text()
    assert "trust_env=False, follow_redirects=False" in script
    assert 'base-url' not in script
    assert 'exist_ok=False' in script


def test_provenance_covers_backend_dependencies_and_detects_drift(frontend, monkeypatch):
    from scripts import demo_local
    sources = demo_local.source_fingerprints()
    for name in ('backend/swingsync/proposal_handoff.py', 'backend/swingsync/movement/contracts.py',
                 'backend/swingsync/movement/measurements.py', 'backend/swingsync/research/models.py'):
        assert name in sources
    original = demo_local.sha
    def changed(path):
        return '0'*64 if str(path).endswith('backend/swingsync/proposal_handoff.py') else original(path)
    monkeypatch.setattr(demo_local, 'sha', changed)
    with pytest.raises(ValueError, match='source changed'):
        demo_local._unchanged({'source_sha256': sources, 'frontend_sha256': frontend_fingerprints(frontend)}, frontend)


def test_child_exit_race_is_reaped_without_unrelated_signals(monkeypatch):
    from scripts import demo_local
    class Process:
        pid = 123456789
        returncode = None
        def poll(self):
            return None
        def wait(self, timeout):
            self.returncode = 0
            return 0
    signals = []
    def already_exited(pid, signum):
        signals.append((pid, signum))
        raise ProcessLookupError('child exited after poll')
    monkeypatch.setattr(demo_local.os, 'killpg', already_exited)
    assert demo_local.stop_owned(Process()) == 0
    assert len(signals) == 1 and signals[0][0] == Process.pid


def test_output_drift_is_refused_after_publication(tmp_path, frontend):
    from scripts import demo_local
    fixture = tmp_path/'synthetic-software-only.mp4'
    fixture.write_bytes(b'known fixture bytes')
    report = tmp_path/'report.html'
    report.write_text('software-only report')
    ledger = {'source_sha256': demo_local.source_fingerprints(), 'frontend_sha256': frontend_fingerprints(frontend),
              'fixture_sha256': sha(fixture), 'outputs_sha256': {'report.html': sha(report)}}
    demo_local._unchanged(ledger, frontend, tmp_path)
    report.write_text('changed after publication')
    with pytest.raises(ValueError, match='output changed'):
        demo_local._unchanged(ledger, frontend, tmp_path)
