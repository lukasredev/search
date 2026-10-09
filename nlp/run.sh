#!/usr/bin/env bash
# Dev entrypoint; the container uses uvicorn directly.
cd "$(dirname "$0")"
# Loopback only: the sidecar has no auth. Set NLP_HOST=0.0.0.0 to reach it from elsewhere.
exec ./.venv/bin/uvicorn app:app --host "${NLP_HOST:-127.0.0.1}" --port "${NLP_PORT:-8081}" "$@"
