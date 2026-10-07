#!/usr/bin/env bash
# Dev entrypoint; the container uses uvicorn directly.
cd "$(dirname "$0")"
exec ./.venv/bin/uvicorn app:app --host 0.0.0.0 --port "${NLP_PORT:-8081}" "$@"
