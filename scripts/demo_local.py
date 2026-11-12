"""Reproduce local HTTP inspection with original synthetic media, never ML evidence."""
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

import av
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TICKS = (5000, 5040, 5100, 5180, 5300, 5480, 5700, 5830, 5940)
SCOPE = 'Original synthetic software-only fixture; no golf, model inference, accuracy or human labels.'
# Original five-column bitmap lettering, not a downloaded font or third-party media.
GLYPHS = {
    'S': ('11111','10000','10000','11111','00001','00001','11111'),
    'Y': ('10001','10001','01010','00100','00100','00100','00100'),
    'N': ('10001','11001','11001','10101','10011','10011','10001'),
    'T': ('11111','00100','00100','00100','00100','00100','00100'),
    'H': ('10001','10001','10001','11111','10001','10001','10001'),
    'E': ('11111','10000','10000','11110','10000','10000','11111'),
    'I': ('11111','00100','00100','00100','00100','00100','11111'),
    'C': ('11111','10000','10000','10000','10000','10000','11111'),
    'O': ('11111','10001','10001','10001','10001','10001','11111'),
    'F': ('11111','10000','10000','11110','10000','10000','10000'),
    'W': ('10001','10001','10001','10101','10101','10101','01010'),
    'A': ('01110','10001','10001','11111','10001','10001','10001'),
    'R': ('11110','10001','10001','11110','10100','10010','10001'),
    'L': ('10000','10000','10000','10000','10000','10000','11111'),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_fingerprints():
    paths = [ROOT/'scripts/demo_local.py', ROOT/'scripts/serve_local.py', ROOT/'pyproject.toml', ROOT/'uv.lock']
    paths.extend(sorted((ROOT/'backend').rglob('*.py')))
    require(not any(path.is_symlink() for path in paths), 'Demo source tree must not contain symlinks.')
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def _json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _text(rgb, text, y):
    for column, letter in enumerate(text):
        for row, bits in enumerate(GLYPHS[letter]):
            for x, bit in enumerate(bits):
                if bit == '1':
                    rgb[y+row*4:y+(row+1)*4, 16+column*24+x*4:16+column*24+(x+1)*4] = 245


def create_fixture(path):
    """H.264 fixture with visible scope and explicit nonzero, nonuniform timestamps."""
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=25)
        stream.width, stream.height, stream.pix_fmt = 320, 200, 'yuv420p'
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.max_b_frames = 0
        stream.codec_context.thread_count = 1
        stream.options = {'preset': 'ultrafast', 'crf': '18'}
        for index, tick in enumerate(TICKS):
            rgb = np.full((200, 320, 3), (22, 42, 39), dtype=np.uint8)
            for text, y in (('SYNTHETIC', 16), ('SOFTWARE', 56), ('ONLY', 96)):
                _text(rgb, text, y)
            rgb[148:178, 16:16+(index+1)*30] = (45+index*20, 160, 90)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts, frame.time_base = tick, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    rows, images = decode_fixture(path)
    if [row['seconds'] for row in rows] != [float(Fraction(t, 1000)) for t in TICKS]:
        raise ValueError('Fixture encoder did not preserve the declared native timestamps.')
    return rows, images


def decode_fixture(path):
    rows, images = [], []
    with av.open(str(path), options={'protocol_whitelist': 'file'}) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            rows.append({'frame_index': index, 'pts': frame.pts,
                         'time_base_numerator': frame.time_base.numerator,
                         'time_base_denominator': frame.time_base.denominator,
                         'seconds': float(frame.pts*frame.time_base)})
            images.append(frame.to_ndarray(format='rgb24'))
    return rows, images


def png_rgb(raw):
    with av.open(io.BytesIO(raw), format='image2pipe') as container:
        frames = list(container.decode(video=0))
        if len(frames) != 1:
            raise ValueError('Canonical PNG did not decode to one image.')
        return frames[0].to_ndarray(format='rgb24')


def frontend_fingerprints(directory):
    directory = Path(directory).resolve()
    if not (directory/'index.html').is_file():
        raise ValueError('Build the frontend before running this demo.')
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Demo frontend files must not contain symlinks.')
        if path.is_file():
            result[str(path.relative_to(directory))] = sha(path)
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


class OwnedHTTP:
    def __init__(self, process, client, deadline):
        self.process, self.client, self.deadline = process, client, deadline

    def request(self, method, path, expected=200, **kwargs):
        require(path.startswith('/') and not path.startswith('//'), 'Only local relative API paths are allowed.')
        require(self.process.poll() is None, 'The owned demo server stopped; no other service will be contacted.')
        require(time.monotonic() < self.deadline, 'Demo exceeded its total execution budget.')
        kwargs.setdefault('timeout', min(5, max(.001, self.deadline-time.monotonic())))
        response = self.client.request(method, path, **kwargs)
        require(response.status_code == expected, f'{method} {path} returned {response.status_code}; expected {expected}.')
        return response


def stop_owned(process):
    """Only the fresh session created for this child is signalled."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # The child can exit between poll and signal; still reap it.
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    return process.returncode


@contextmanager
def own_server(output, frontend, ledger, deadline):
    """Pass a prebound socket rather than release and race for a chosen port."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    process = None
    client = None
    attempt = {'state': 'starting', 'started_at': datetime.now(timezone.utc).isoformat()}
    ledger['server_attempts'].append(attempt)
    try:
        listener.bind(('127.0.0.1', 0))
        listener.listen(32)
        attempt['port'] = listener.getsockname()[1]
        _json(output/'run.json', ledger)
        with (output/f'server-{len(ledger["server_attempts"])}.log').open('xb') as log:
            process = subprocess.Popen(
                [sys.executable, str(ROOT/'scripts/serve_local.py'), '--socket-fd', str(listener.fileno()),
                 '--frontend-dir', str(frontend), '--data-dir', str(output/'storage')],
                cwd=ROOT, pass_fds=(listener.fileno(),), start_new_session=True,
                stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        attempt.update(pid=process.pid, state='starting')
        _json(output/'run.json', ledger)
        client = httpx.Client(base_url=f'http://127.0.0.1:{attempt["port"]}', timeout=5,
                              trust_env=False, follow_redirects=False)
        http = OwnedHTTP(process, client, deadline)
        startup_deadline = min(deadline, time.monotonic()+30)
        while True:
            require(process.poll() is None, 'Owned server exited during startup; inspect its demo log.')
            require(time.monotonic() < startup_deadline, 'Owned server did not start within30seconds.')
            try:
                health = http.request('GET', '/api/health', timeout=min(.25, max(.001, startup_deadline-time.monotonic()))).json()
                break
            except (httpx.ConnectError, httpx.ReadTimeout):
                time.sleep(.05)
        require(health['model']['status'] == 'unavailable', 'Demo must not enable automatic models.')
        index = http.request('GET', '/').content
        require(hashlib.sha256(index).hexdigest() == ledger['frontend_sha256']['index.html'],
                'Server frontend does not match the recorded built index.')
        attempt['state'] = 'running'
        _json(output/'run.json', ledger)
        yield http
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            cleanup_ok = False
            try:
                if process is not None:
                    attempt['returncode'] = stop_owned(process)
                cleanup_ok = True
            finally:
                listener.close()
                attempt.update(state='stopped' if cleanup_ok else 'cleanup-failed',
                               stopped_at=datetime.now(timezone.utc).isoformat())
                _json(output/'run.json', ledger)


def upload_and_wait(http, fixture, name):
    with fixture.open('rb') as stream:
        item = http.request('POST', '/api/analyses', 202, files={'file': (name, stream, 'video/mp4')},
                            data={'camera_view': 'face-on', 'handedness': 'right', 'club': 'software fixture',
                                  'mirrored': 'false'}).json()
    while item['status'] in ('queued', 'processing'):
        time.sleep(.05)
        item = http.request('GET', '/api/analyses/'+item['id']).json()
    require(item['status'] == 'completed', 'Synthetic fixture inspection failed.')
    return item


def run_demo(output, frontend_dir=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frontend = Path(frontend_dir or ROOT/'frontend/dist').resolve()
    ledger = {'schema_version': 1, 'job_id': uuid.uuid4().hex, 'state': 'initializing',
              'created_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'scope': SCOPE,
              'training_performed': False, 'inference_performed': False, 'accuracy_measured': False,
              'independent_human_labels': False, 'normal_model_available': False, 'advice': [],
              'runtime': {'python': platform.python_version(), 'executable': sys.executable,
                          'platform': platform.platform(), 'machine': platform.machine(),
                          'av': av.__version__, 'numpy': np.__version__, 'httpx': httpx.__version__},
              'source_sha256': {}, 'frontend_sha256': {}, 'server_attempts': []}
    _json(output/'run.json', ledger)
    started = time.monotonic()
    try:
        ledger['source_sha256'] = source_fingerprints()
        ledger['frontend_sha256'] = frontend_fingerprints(frontend)
        ledger.update(state='processing', requested_fixture_ticks=list(TICKS), requested_time_base=[1, 1000])
        _json(output/'run.json', ledger)
        fixture = output/'synthetic-software-only.mp4'
        rows, images = create_fixture(fixture)
        ledger.update(fixture_sha256=sha(fixture), native_frames=rows, frame_count=len(rows), codec='h264')
        _json(output/'run.json', ledger)
        with own_server(output, frontend, ledger, started+120) as http:
            require(http.request('GET', '/api/analyses').json() == [], 'Demo storage must start empty.')
            item = upload_and_wait(http, fixture, fixture.name)
            url = '/api/analyses/'+item['id']
            require(item['video']['sha256'] == ledger['fixture_sha256'], 'Inspected video SHA differs.')
            require(item['video']['source_timestamps_seconds'] == [r['seconds'] for r in rows], 'Native PTS differ.')
            require(not item['events'] and not item['measurements'] and not item['advice'], 'Initial inspection fabricated analysis.')
            canonical = http.request('GET', url+'/frames/5').content
            require(np.array_equal(png_rgb(canonical), images[5]), 'Canonical frame RGB differs from the decoded source.')
            (output/'canonical-frame5.png').write_bytes(canonical)
            initial = {'expected_revision': 0, 'events': [{'name': 'Address', 'frame_index': 0},
                       {'name': 'Top', 'frame_index': 4}, {'name': 'Impact', 'frame_index': 6}]}
            saved = http.request('PUT', url+'/events', json=initial).json()
            corrected = {'expected_revision': 1, 'events': [{'name': 'Address', 'frame_index': 0},
                         {'name': 'Top', 'frame_index': 5}, {'name': 'Impact', 'frame_index': 6}]}
            saved = http.request('PUT', url+'/events', json=corrected).json()
            http.request('PUT', url+'/events', 409, json=initial)
            require(saved['annotation_revision'] == 2 and len(saved['annotation_history']) == 2,
                    'Manual correction history did not persist.')
            require(saved['annotation_history'][0]['events'][1]['frame_index'] == 4,
                    'Original manual Top frame was overwritten.')
            expected = [rows[5]['seconds']-rows[0]['seconds'], rows[6]['seconds']-rows[5]['seconds']]
            require([m['value'] for m in saved['measurements']] == [*expected, expected[0]/expected[1]],
                    'Manual timing did not use actual PTS.')
            require(saved['model']['status'] == 'unavailable' and not saved['observations'] and not saved['advice'],
                    'Demo must keep unsupported models/advice unavailable.')
            report = http.request('GET', url+'/report').content
            encoded_images = re.findall(rb'src="data:image/png;base64,([^"]+)"', report)
            require(len(encoded_images) == 3, 'Report did not contain three canonical manual frames.')
            for raw, index in zip(encoded_images, (0, 5, 6)):
                require(np.array_equal(png_rgb(base64.b64decode(raw, validate=True)), images[index]),
                        'Report canonical image differs from source RGB.')
            require(b'not a trained-model demonstration' in report and b'Manual corrections' in report,
                    'Report did not preserve software scope and correction evidence.')
            (output/'report.html').write_bytes(report)
            _json(output/'analysis.json', saved)
        with own_server(output, frontend, ledger, started+120) as http:
            require(http.request('GET', url).json() == saved, 'Restart changed saved annotations/history.')
            second = upload_and_wait(http, fixture, 'synthetic-deletion-check.mp4')
            http.request('DELETE', '/api/analyses/'+second['id'], 204)
            http.request('GET', '/api/analyses/'+second['id'], 404)
            require([a['id'] for a in http.request('GET', '/api/analyses').json()] == [saved['id']],
                    'Deletion did not retain only the original analysis.')
            media = list((output/'storage').glob('*.mp4'))
            require(len(media) == 1 and sha(media[0]) == ledger['fixture_sha256'], 'Deletion left extra stored media.')
        ledger['checks'] = {'native_pts_exact': True, 'canonical_png_exact_rgb': True,
                            'report_pngs_exact_rgb': 3, 'manual_revisions': 2, 'stale_save_status': 409,
                            'restart_preserved': True, 'second_upload_deleted': True,
                            'retained_analysis_id': saved['id']}
        ledger['outputs_sha256'] = {name: sha(output/name) for name in ('synthetic-software-only.mp4',
                                   'canonical-frame5.png', 'report.html', 'analysis.json')}
        _unchanged(ledger, frontend, output)
        ledger.update(state='completed', completed_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-started,
                      timing_scope='Entire synthetic HTTP workflow including two local server lifetimes; not model latency.')
        _json(output/'run.json', ledger)
        _unchanged(ledger, frontend, output)
        return ledger
    except BaseException as exc:
        ledger.update(state='failed', error=f'{type(exc).__name__}: {exc}',
                      failed_at=datetime.now(timezone.utc).isoformat())
        _json(output/'run.json', ledger)
        raise


def _unchanged(ledger, frontend, output=None):
    require(source_fingerprints() == ledger['source_sha256'], 'Demo source changed during execution.')
    require(frontend_fingerprints(frontend) == ledger['frontend_sha256'], 'Frontend build changed during execution.')
    if output is not None:
        require(sha(output/'synthetic-software-only.mp4') == ledger['fixture_sha256'], 'Synthetic fixture changed during execution.')
        require(all(sha(output/name) == digest for name, digest in ledger['outputs_sha256'].items()),
                'Demo output changed during publication.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory; never overwrites existing demo state.')
    parser.add_argument('--frontend-dir', type=Path, default=ROOT/'frontend/dist')
    args = parser.parse_args()
    try:
        result = run_demo(args.output, args.frontend_dir)
    except (ValueError, OSError, httpx.HTTPError, av.error.FFmpegError) as exc:
        parser.exit(2, f'Local software demo failed: {exc}\n')
    print(json.dumps({'state': result['state'], 'job_id': result['job_id'], 'output': str(args.output),
                      'scope': SCOPE, 'normal_model_available': False}))


if __name__ == '__main__':
    main()
