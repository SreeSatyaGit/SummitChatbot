#!/usr/bin/env bash
# Simple reproducible E2E test: send a message to Rasa REST webhook (inside container) and validate action output
set -euo pipefail
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

# Start by ensuring services are up (compose already manages this in the environment)
# We'll exec into the rasa container to call its internal REST webhook so network resolution works.

RASA_CONTAINER=$(docker-compose ps -q rasa)
if [ -z "$RASA_CONTAINER" ]; then
  echo "rasa container not running"
  exit 2
fi

# Wait for rag_server to report healthy
echo "Waiting for rag_server health..."
for i in {1..20}; do
  OK=$(docker-compose exec -T rasa_action_server bash -lc "python -c 'import requests,sys;print(requests.get(\"http://rag_server:8000/health\").status_code if True else 0)' 2>/dev/null || true") || true
  if echo "$OK" | grep -q "200"; then
    echo "rag_server healthy"
    break
  fi
  echo "sleeping..."
  sleep 2
done

# Send a message that triggers the 'suggestion' story
RESP=$(docker-compose exec -T rasa bash -lc "curl -sS -X POST 'http://localhost:5005/webhooks/rest/webhook' -H 'Content-Type: application/json' -d '{\"sender\":\"e2e_test_user\",\"message\":\"I need a suggestion for my bio\"}'")

echo "--- Rasa webhook response ---"
echo "$RESP"

# Basic assertion: expect 'Profile Draft' or 'Clarifying question' in the output
if echo "$RESP" | grep -q -i "Profile Draft\|Clarifying question\|Profile"; then
  echo "E2E test PASSED"
  exit 0
else
  echo "E2E test FAILED: expected profile draft or clarifying question in response"
  exit 3
fi

echo "\n-- Now verifying alternate key via X-API-KEY header (second key) --"
# Test contacting the action server's /test endpoint which forwards to RAG
TEST_RESP=$(docker-compose exec -T rasa_action_server bash -lc "curl -sS -X POST http://localhost:5055/test -H 'Content-Type: application/json' -d '{\"text\":\"Hello\"}'") || true
echo "Action server /test response: $TEST_RESP"

if echo "$TEST_RESP" | grep -q "Unauthorized\|401"; then
  echo "Alternate key test FAILED (action server couldn't contact RAG)"
  exit 4
fi

echo "Alternate key test PASSED"
