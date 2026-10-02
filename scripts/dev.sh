#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ACTION="${1:-help}"
PORT="${PORT:-8000}"
PY="${PROJECT_PYTHON:-$ROOT/.venv/bin/python}"
case "$ACTION" in
  setup)
    if [[ ! -x "$PY" ]]; then "${PYTHON:-python3}" -m venv .venv; fi
    "$PY" -m pip install --only-binary=:all: -r requirements-dev.txt
    ;;
  test|serve|health)
    if [[ ! -x "$PY" ]]; then echo 'Run: bash scripts/dev.sh setup' >&2; exit 1; fi
    case "$ACTION" in
      test) "$PY" -m unittest discover -s tests -v ;;
      serve) exec "$PY" -m uvicorn api:app --host 127.0.0.1 --port "$PORT" ;;
      health) "$PY" -c 'import json,sys,urllib.request; response=urllib.request.urlopen("http://127.0.0.1:"+sys.argv[1]+"/health",timeout=5); body=json.load(response); assert body.get("status")=="ok", body; print("Health check passed")' "$PORT" ;;
    esac
    ;;
  help) echo 'Usage: bash scripts/dev.sh {setup|test|serve|health} (optional PORT, PROJECT_PYTHON)' ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
