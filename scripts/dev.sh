#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
uv run uvicorn backend.swingsync.app:app --host 127.0.0.1 --port 8000 &
backend_pid=$!
trap 'kill "$backend_pid" 2>/dev/null || true' EXIT INT TERM
cd frontend
npm run dev -- --host 127.0.0.1
