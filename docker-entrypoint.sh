#!/bin/sh
set -e
trap 'kill 0' TERM INT

python -m uvicorn langchain_backend.main:app --host 0.0.0.0 --port 8001 &

cd langgraph-backend
langgraph dev --host 0.0.0.0 --port 2024 --no-browser --no-reload --allow-blocking &

wait
