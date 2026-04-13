#!/bin/bash
# PickLedger — "run models" command
# Usage: bash scripts/run_models.sh [YYYY-MM-DD]
# Runs all prediction models locally and saves results to Firebase via /save-admin-picks.
set -euo pipefail

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

DATE=${1:-$(date +%Y-%m-%d)}
PORT=${PORT:-10000}
export PORT
BASE_URL="http://localhost:$PORT"
SECRET=${ADMIN_PICKS_SECRET:-""}
SERVER_PID=""

echo "=========================================="
echo " PickLedger Model Runner"
echo " Date: $DATE"
echo " Backend: $BASE_URL"
echo "=========================================="

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    echo "Stopping local server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

if [ -z "$SECRET" ]; then
  echo "ERROR: ADMIN_PICKS_SECRET is not set in the environment."
  echo "Add it to your local .env before running this script."
  exit 1
fi

if ! curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
  echo "[1/6] Starting local server..."
  python3 pickgrader_server.py &
  SERVER_PID=$!
  echo "      PID: $SERVER_PID"
  sleep 4
  if ! curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Server failed to start. Check pickgrader_server.py."
    exit 1
  fi
  echo "      Server ready."
else
  echo "[1/6] Server already running — skipping start."
fi

run_model() {
  local label=$1
  local method=$2
  local endpoint=$3
  local fb_key=$4
  local body=$5

  echo ""
  echo "[$label] Running $endpoint..."

  local result=""
  if [ "$method" = "GET" ]; then
    result=$(curl -sf "$BASE_URL$endpoint" 2>&1) || {
      echo "  WARN: $endpoint failed or timed out — skipping."
      return
    }
  else
    result=$(curl -sf -X "$method" "$BASE_URL$endpoint" \
      -H "Content-Type: application/json" \
      -d "$body" 2>&1) || {
      echo "  WARN: $endpoint failed or timed out — skipping."
      return
    }
  fi

  echo "  Done. Saving to Firebase as '$fb_key'..."
  curl -sf -X POST "$BASE_URL/save-admin-picks" \
    -H "Content-Type: application/json" \
    -d "{\"secret\":\"$SECRET\",\"model\":\"$fb_key\",\"date\":\"$DATE\",\"picks\":$result}" > /dev/null 2>&1 || {
      echo "  WARN: Firebase save failed for $fb_key."
      return
    }
  echo "  ✓ Saved: $fb_key"
}

run_model "2/6" "POST" "/run-nba-model" "nba" "{\"date\":\"$DATE\"}"
run_model "3/6" "POST" "/run-nba-model" "nba_new" "{\"date\":\"$DATE\",\"model\":\"new\"}"
run_model "4/6" "POST" "/run-mlb-model" "mlb" "{\"date\":\"$DATE\"}"
run_model "5/6" "GET" "/api/ipl" "ipl" "{}"
run_model "6/6" "POST" "/run-nba-props-model" "props" "{\"date\":\"$DATE\"}"

echo ""
echo "=========================================="
echo " All models complete for $DATE"
echo "=========================================="
