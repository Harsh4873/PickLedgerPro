#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GRADER_PLIST_SRC="$REPO_DIR/scripts/com.pickledger.grader.plist"
GRADER_PLIST_DEST="$HOME/Library/LaunchAgents/com.pickledger.grader.plist"
MODELS_PLIST_SRC="$REPO_DIR/scripts/com.pickledger.models.plist"
MODELS_PLIST_DEST="$HOME/Library/LaunchAgents/com.pickledger.models.plist"

# Load repo-local env files when variables were sourced without export.
for env_file in "$REPO_DIR/.env" "$REPO_DIR/.env.local"; do
  if [ -f "$env_file" ] && [ -z "${ADMIN_PICKS_SECRET:-}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done

SECRET="${ADMIN_PICKS_SECRET:-}"
GOOGLE_CREDS="${GOOGLE_APPLICATION_CREDENTIALS:-}"

if [ -z "$SECRET" ]; then
  echo "ERROR: source .env first"
  exit 1
fi

escape_sed() {
  printf '%s' "$1" | sed 's/[\\/&|]/\\&/g'
}

REPO_DIR_ESCAPED="$(escape_sed "$REPO_DIR")"
SECRET_ESCAPED="$(escape_sed "$SECRET")"
GOOGLE_CREDS_ESCAPED="$(escape_sed "$GOOGLE_CREDS")"

mkdir -p "$(dirname "$GRADER_PLIST_DEST")"

chmod +x "$REPO_DIR/scripts/grader_loop.py" "$REPO_DIR/scripts/run_models.sh" "$REPO_DIR/scripts/install_grader_agent.sh"

sed -e "s|REPO_PATH_PLACEHOLDER|$REPO_DIR_ESCAPED|g" \
    -e "s|PLACEHOLDER_SET_BY_INSTALL_SCRIPT|$SECRET_ESCAPED|g" \
    -e "s|GOOGLE_APPLICATION_CREDENTIALS_PLACEHOLDER|$GOOGLE_CREDS_ESCAPED|g" \
    "$GRADER_PLIST_SRC" > "$GRADER_PLIST_DEST"

launchctl enable "gui/$(id -u)/com.pickledger.grader" 2>/dev/null || true
launchctl unload "$GRADER_PLIST_DEST" 2>/dev/null || true
launchctl load "$GRADER_PLIST_DEST"

sed -e "s|REPO_PATH_PLACEHOLDER|$REPO_DIR_ESCAPED|g" \
    -e "s|PLACEHOLDER_SET_BY_INSTALL_SCRIPT|$SECRET_ESCAPED|g" \
    -e "s|GOOGLE_APPLICATION_CREDENTIALS_PLACEHOLDER|$GOOGLE_CREDS_ESCAPED|g" \
    "$MODELS_PLIST_SRC" > "$MODELS_PLIST_DEST"

launchctl enable "gui/$(id -u)/com.pickledger.models" 2>/dev/null || true
launchctl unload "$MODELS_PLIST_DEST" 2>/dev/null || true
launchctl load "$MODELS_PLIST_DEST"

echo "✓ Grader agent installed. Runs every 15 min."
echo "✓ Models agent installed. Runs daily at 8:30 AM."
echo "  Logs: ~/Library/Logs/pickledger_grader.log"
echo "        ~/Library/Logs/pickledger_models.log"
if [ -z "$GOOGLE_CREDS" ]; then
  echo "  Warning: GOOGLE_APPLICATION_CREDENTIALS is not set; LaunchAgents must rely on ambient ADC."
fi
echo "  Status: launchctl list | grep pickledger"
