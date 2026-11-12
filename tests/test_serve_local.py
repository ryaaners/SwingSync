"""Built-interface/local-API serving tests; no model or dataset required."""
import os
import socket

from fastapi.testclient import TestClient
import pytest

from scripts.serve_local import create_local_app, main, validate_listener, is_listening


@pytest.fixture
def build(tmp_path):
    root = tmp_path / 'public'
    (root / 'assets').mkdir(parents=True)
    (root / 'index.html').write_text('<!doctype html><title>Synthetic interface fixture</title>')
    (root / 'assets/app.js').write_text('console.log("software fixture");')
    return root


def test_same_origin_index_asset_and_real_api_remain_distinct(build, tmp_path):
    with TestClient(create_local_app(build, tmp_path / 'private')) as client:
        index = client.get('/')
        assert index.status_code == 200 and 'Synthetic interface fixture' in index.text
        assert index.headers['content-type'].startswith('text/html')
        assert client.get('/assets/app.js').text == 'console.log("software fixture");'
        health = client.get('/api/health')
        assert health.status_code == 200
        assert health.json()['model']['status'] == 'unavailable'
        assert client.get('/api/analyses').json() == []
        assert client.get('/missing-page').status_code == 404
        assert client.get('/missing-page').headers['content-type'].startswith('application/json')


@pytest.mark.parametrize('method', ['get', 'head', 'post', 'put', 'patch', 'delete', 'options'])
def test_unknown_api_never_falls_through_to_static_files(build, tmp_path, method):
    (build / 'api').mkdir()
    (build / 'api/not-a-route').write_text('must not serve static API lookalike')
    with TestClient(create_local_app(build, tmp_path / 'private')) as client:
        response = getattr(client, method)('/api/not-a-route')
        assert response.status_code == 404
        assert response.headers['content-type'].startswith('application/json')
        assert client.get('/api').status_code == 404


@pytest.mark.parametrize('path', ['/../secret.txt', '/%2e%2e/secret.txt', '/assets/../../secret.txt', '/assets/%2e%2e/%2e%2e/secret.txt'])
def test_static_paths_cannot_escape_public_tree(build, tmp_path, path):
    (tmp_path / 'secret.txt').write_text('private fixture')
    with TestClient(create_local_app(build, tmp_path / 'private')) as client:
        response = client.get(path)
        assert response.status_code == 404 and 'private fixture' not in response.text


@pytest.mark.parametrize('missing', ['directory', 'index.html', 'assets', 'javascript'])
def test_missing_build_refused_with_build_instructions(tmp_path, missing):
    build = tmp_path / 'public'
    if missing != 'directory':
        build.mkdir()
        if missing != 'index.html':
            (build / 'index.html').write_text('build fixture')
        if missing != 'assets':
            (build / 'assets').mkdir()
            if missing != 'javascript':
                (build / 'assets/app.js').write_text('fixture')
    with pytest.raises(ValueError, match='npm ci && npm run build'):
        create_local_app(build, tmp_path / 'private')


@pytest.mark.parametrize('target', ['index.html', 'assets/app.js', 'assets/link-dir'])
def test_public_symlinks_are_refused(build, tmp_path, target):
    outside = tmp_path / 'outside'
    outside.mkdir()
    secret = outside / 'secret.js'
    secret.write_text('private fixture')
    link = build / target
    if link.exists():
        link.unlink()
    link.symlink_to(outside if target.endswith('dir') else secret)
    with pytest.raises(ValueError, match='symbolic links'):
        create_local_app(build, tmp_path / 'private')


def test_replaced_asset_symlink_cannot_escape_after_startup(build, tmp_path):
    outside = tmp_path / 'secret.txt'
    outside.write_text('private fixture')
    app = create_local_app(build, tmp_path / 'private')
    with TestClient(app) as client:
        asset = build / 'assets/app.js'
        asset.unlink()
        asset.symlink_to(outside)
        assert client.get('/assets/app.js').status_code == 404


@pytest.mark.parametrize('relation', ['same', 'private-in-public', 'public-in-private'])
def test_private_storage_and_public_build_must_not_overlap(build, tmp_path, relation):
    private = {'same': build, 'private-in-public': build / 'private', 'public-in-private': tmp_path}[relation]
    with pytest.raises(ValueError, match='separate public-build and private-data'):
        create_local_app(build, private)


def test_launcher_enforces_loopback_and_one_worker(build, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('scripts.serve_local.uvicorn.run', lambda app, **kwargs: calls.append(kwargs))
    main(['--frontend-dir', str(build), '--data-dir', str(tmp_path / 'private'), '--port', '8123'])
    assert calls == [{'host': '127.0.0.1', 'port': 8123, 'workers': 1, 'reload': False}]


@pytest.mark.parametrize('args', [['--port', '0'], ['--port', '65536'], ['--port', 'not-a-number'], ['--host', '0.0.0.0'], ['--workers', '2']])
def test_launcher_bad_port_or_binding_override_refused(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_environment_data_directory_is_used_without_public_exposure(build, tmp_path, monkeypatch):
    private = tmp_path / 'private'
    monkeypatch.setenv('SWINGSYNC_DATA_DIR', str(private))
    with TestClient(create_local_app(build)) as client:
        assert (private / 'analyses.sqlite3').exists()
        assert client.get('/analyses.sqlite3').status_code == 404
        assert client.get('/api/health').json()['model']['status'] == 'unavailable'


def test_inherited_listener_is_validated_and_original_descriptor_remains_owned(build, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('scripts.serve_local.uvicorn.run', lambda app, **kwargs: calls.append(kwargs))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        fd = listener.fileno()
        main(['--frontend-dir', str(build), '--data-dir', str(tmp_path / 'private'), '--socket-fd', str(fd)])
        assert calls == [{'fd': fd, 'host': '127.0.0.1', 'workers': 1, 'reload': False}]
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_STREAM


@pytest.mark.parametrize('kind', ['all-interfaces', 'not-listening', 'udp', 'unix', 'file'])
def test_inherited_descriptor_rejects_external_nonlistening_and_non_tcp_sockets(tmp_path, kind):
    if kind == 'file':
        with (tmp_path / 'ordinary-file').open('w') as stream:
            with pytest.raises(ValueError, match='loopback TCP socket'):
                validate_listener(stream.fileno())
        return
    if kind == 'unix':
        first, second = socket.socketpair()
        with first, second:
            with pytest.raises(ValueError, match='loopback TCP socket'):
                validate_listener(first.fileno())
        return
    family = socket.AF_INET
    socket_type = socket.SOCK_DGRAM if kind == 'udp' else socket.SOCK_STREAM
    with socket.socket(family, socket_type) as listener:
        listener.bind(('0.0.0.0' if kind == 'all-interfaces' else '127.0.0.1', 0))
        if kind not in ('udp', 'not-listening'):
            listener.listen(1)
        with pytest.raises(ValueError, match='loopback TCP socket'):
            validate_listener(listener.fileno())
        assert listener.fileno() >= 0  # Rejection must not close caller ownership.


@pytest.mark.parametrize('descriptor', [-1, True, '1', 1.0, None])
def test_inherited_descriptor_requires_an_integer(descriptor):
    with pytest.raises(ValueError, match='loopback TCP socket'):
        validate_listener(descriptor)


def test_closed_descriptor_and_conflicting_binding_refused(tmp_path):
    descriptor = os.open(tmp_path / 'ordinary-file', os.O_WRONLY | os.O_CREAT)
    os.close(descriptor)
    with pytest.raises(ValueError, match='loopback TCP socket'):
        validate_listener(descriptor)
    for args in (['--socket-fd', '-1'], ['--socket-fd', 'x'], ['--socket-fd', '0', '--port', '8000']):
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 2


@pytest.mark.parametrize('state,expected', [(b'', False), (bytes([0])+bytes(111), False), (bytes([1])+bytes(111), True), (bytes([4])+bytes(111), False)])
def test_darwin_tcp_state_requires_listen_without_mutating_socket(monkeypatch, state, expected):
    from types import SimpleNamespace
    monkeypatch.setattr('scripts.serve_local.sys.platform', 'darwin')
    monkeypatch.setattr(socket, 'TCP_CONNECTION_INFO', 262, raising=False)
    calls = []
    def read_option(*args):
        calls.append(args)
        return state
    assert is_listening(SimpleNamespace(getsockopt=read_option)) is expected
    assert calls == [(socket.IPPROTO_TCP, 262, 256)]


@pytest.mark.parametrize('state,expected', [(0, False), (1, True), (2, False)])
def test_linux_socket_listen_flag(monkeypatch, state, expected):
    from types import SimpleNamespace
    monkeypatch.setattr('scripts.serve_local.sys.platform', 'linux')
    calls = []
    def read_option(*args):
        calls.append(args)
        return state
    assert is_listening(SimpleNamespace(getsockopt=read_option)) is expected
    assert calls == [(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)]


def test_replaced_public_root_symlink_cannot_rebase_serving_after_startup(build, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'index.html').write_text('private sentinel')
    app = create_local_app(build, tmp_path / 'private')
    with TestClient(app) as client:
        build.rename(tmp_path / 'original-public')
        build.symlink_to(outside, target_is_directory=True)
        response = client.get('/')
        assert response.status_code == 404 and 'private sentinel' not in response.text
        assert client.get('/api/health').json()['model']['status'] == 'unavailable'
