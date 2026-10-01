#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-help}" in
  start)
    if [ ! -f .runtime/anythingllm.env ]; then
      mkdir -p .runtime
      chmod 700 .runtime
      cp config/anythingllm.env.example .runtime/anythingllm.env
      chmod 600 .runtime/anythingllm.env
    fi
    open -a Docker
    for attempt in {1..60}; do
      if docker info >/dev/null 2>&1; then break; fi
      sleep 2
    done
    docker info >/dev/null
    brew services start ollama
    docker compose up -d
    for attempt in {1..60}; do
      if curl -fsS http://127.0.0.1:3001/api/ping >/dev/null; then
        open http://localhost:3001
        exit 0
      fi
      sleep 2
    done
    echo 'AnythingLLM is not ready yet. Run: ./lab.sh status'
    exit 1
    ;;
  stop)
    docker compose stop
    echo 'AnythingLLM stopped; documents and chat history are preserved. Ollama remains running.'
    ;;
  stop-ollama)
    brew services stop ollama
    ;;
  restart)
    docker compose restart
    ;;
  status)
    docker compose ps
    curl -fsS --max-time 5 http://127.0.0.1:3001/api/ping || true
    echo
    curl -fsS --max-time 5 http://127.0.0.1:11434/api/version || true
    echo
    ollama list
    ollama ps
    ;;
  logs)
    docker compose logs --tail=100 anythingllm
    ;;
  model)
    selected_model="${2:-}"
    case "$selected_model" in
      qwen3.5:4b|qwen2.5:7b) ;;
      *) echo 'Choose: ./lab.sh model qwen3.5:4b or ./lab.sh model qwen2.5:7b'; exit 1 ;;
    esac
    mkdir -p .runtime
    chmod 700 .runtime
    curl -fsS --max-time 30 -H 'Content-Type: application/json' \
      --data "{\"chatProvider\":\"ollama\",\"chatModel\":\"$selected_model\"}" \
      http://127.0.0.1:3001/api/workspace/company-policy-lab/update \
      -o .runtime/model-switch.json
    python3 - "$selected_model" <<'PY'
import json
import sys
from pathlib import Path
result = json.loads(Path('.runtime/model-switch.json').read_text())
if (result.get('workspace') or {}).get('chatModel') != sys.argv[1]:
    raise SystemExit('Failed to switch model: ' + str(result.get('message')))
print('Company Policy Lab model: ' + sys.argv[1] + '. Refresh the browser page.')
PY
    ;;
  *)
    echo 'Usage: ./lab.sh {start|stop|stop-ollama|restart|status|logs|model MODEL}'
    ;;
esac
