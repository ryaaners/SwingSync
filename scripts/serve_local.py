"""Serve the built SwingSync interface and local API from one loopback process.

This serves the existing manual inspection application. It never loads research
models or promotes their availability. Build frontend/dist before launching.
"""
import argparse
import os
from pathlib import Path
import socket
import sys

from fastapi import HTTPException
from starlette.staticfiles import StaticFiles
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PinnedStaticFiles(StaticFiles):
    """Do not allow a later root-directory symlink to rebase public serving."""
    def __init__(self, directory):
        self.initial_root = Path(directory).resolve()
        super().__init__(directory=self.initial_root, html=True, follow_symlink=False)

    def lookup_path(self, path):
        if self.initial_root.is_symlink() or self.initial_root.resolve() != self.initial_root:
            return '', None
        full_path, metadata = super().lookup_path(path)
        if full_path and not Path(full_path).is_relative_to(self.initial_root):
            return '', None
        return full_path, metadata


def create_local_app(frontend_dir=None, data_dir=None):
    """Require an actual public build and keep private state outside that tree."""
    from backend.swingsync.app import create_app

    supplied = Path(frontend_dir) if frontend_dir is not None else ROOT / 'frontend/dist'
    public = supplied.resolve()
    if (supplied.is_symlink() or not public.is_dir() or not (public / 'index.html').is_file()
            or not (public / 'assets').is_dir() or not any((public / 'assets').glob('*.js'))):
        raise ValueError('Built interface unavailable. Run cd frontend && npm ci && npm run build, '
                         f'then use its dist directory (received {supplied}).')
    if any(path.is_symlink() for path in public.rglob('*')):
        raise ValueError('The public interface build must not contain symbolic links. Rebuild frontend/dist.')
    private = Path(data_dir if data_dir is not None else os.environ.get('SWINGSYNC_DATA_DIR', ROOT / 'data/local')).resolve()
    if private == public or private.is_relative_to(public) or public.is_relative_to(private):
        raise ValueError('Choose separate public-build and private-data directories; neither may contain the other.')
    app = create_app(data_dir=private)

    # Existing API routes precede these guards. Unknown API URLs must never
    # resolve to a coincidentally named static file or an HTML fallback.
    @app.api_route('/api', methods=['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'], include_in_schema=False)
    @app.api_route('/api/{path:path}', methods=['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'], include_in_schema=False)
    def unknown_api(path=''):
        raise HTTPException(404, 'API route not found.')

    app.mount('/', PinnedStaticFiles(public), name='interface')
    return app


def port_number(value):
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Port must be a whole number from1to65535.') from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError('Port must be a whole number from1to65535.')
    return port


def descriptor_number(value):
    try:
        descriptor = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Socket descriptor must be a nonnegative whole number.') from exc
    if descriptor < 0:
        raise argparse.ArgumentTypeError('Socket descriptor must be a nonnegative whole number.')
    return descriptor


def is_listening(listener):
    if sys.platform == 'darwin':
        # Apple's public tcp.h defines TCP_CONNECTION_INFO and its first byte
        # tcpi_state; tcp_fsm.h defines TCPS_LISTEN=1. SO_ACCEPTCONN inspection
        # is unavailable on the verified macOS runtime. No socket is mutated.
        # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/netinet/tcp.h
        # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/netinet/tcp_fsm.h
        info = listener.getsockopt(socket.IPPROTO_TCP, socket.TCP_CONNECTION_INFO, 256)
        return len(info) >= 1 and info[0] == 1
    return listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1


def validate_listener(descriptor):
    """Check a duplicate descriptor without closing the caller-owned listener."""
    if type(descriptor) is not int or descriptor < 0:
        raise ValueError('Expected an inherited listening IPv4 loopback TCP socket.')
    duplicate = None
    try:
        duplicate = os.dup(descriptor)
        with socket.socket(fileno=duplicate) as listener:
            duplicate = None  # The socket context now owns this duplicate.
            address = listener.getsockname()
            if (listener.family != socket.AF_INET
                    or listener.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM
                    or not is_listening(listener)
                    or address[0] != '127.0.0.1' or not 1 <= address[1] <= 65535):
                raise ValueError('Expected an inherited listening IPv4 loopback TCP socket.')
    except OSError as exc:
        raise ValueError('Expected an inherited listening IPv4 loopback TCP socket.') from exc
    finally:
        if duplicate is not None:
            os.close(duplicate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend-dir', type=Path, default=ROOT / 'frontend/dist')
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Private storage; defaults to SWINGSYNC_DATA_DIR or project data/local.')
    binding = parser.add_mutually_exclusive_group()
    binding.add_argument('--port', type=port_number, default=None)
    binding.add_argument('--socket-fd', type=descriptor_number, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.socket_fd is not None:
            validate_listener(args.socket_fd)
        app = create_local_app(args.frontend_dir, args.data_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    # One process owns the persisted worker. Host and worker count deliberately
    # have no command-line/environment override in this local launcher.
    if args.socket_fd is None:
        uvicorn.run(app, host='127.0.0.1', port=args.port or 8000, workers=1, reload=False)
    else:
        uvicorn.run(app, fd=args.socket_fd, host='127.0.0.1', workers=1, reload=False)


if __name__ == '__main__':
    main()
