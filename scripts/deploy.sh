#!/usr/bin/env bash
# Deploy to Fly.io from Git Bash.
#   bash scripts/deploy.sh            # create app/volume if missing, push secrets, deploy, check the page
#   bash scripts/deploy.sh --secrets  # push .env secrets only and restart
# Needs: flyctl logged in, .env with MODEL_API_KEY and/or CSR_VLM_API_KEY
set -euo pipefail
cd "$(dirname "$0")/.."

FLY="${FLY:-$(command -v flyctl || command -v fly || echo "$LOCALAPPDATA/Microsoft/WinGet/Packages/Fly-io.flyctl_Microsoft.Winget.Source_8wekyb3d8bbwe/flyctl.exe")}"
APP="${FLY_APP:-$(grep -E '^app *=' fly.toml | sed -E 's/.*"(.*)".*/\1/')}"
REGION="$(grep -E '^primary_region *=' fly.toml | sed -E 's/.*"(.*)".*/\1/')"

"$FLY" auth whoami >/dev/null 2>&1 || { echo "flyctl is not logged in: \"$FLY\" auth login"; exit 1; }

# 1) App and volume, only when missing
if ! "$FLY" status -a "$APP" >/dev/null 2>&1; then
  echo "> creating app $APP"
  "$FLY" apps create "$APP" --org personal
fi
if ! "$FLY" volumes list -a "$APP" 2>/dev/null | grep -q standin_runs; then
  echo "> creating volume standin_runs (1GB, $REGION)"
  "$FLY" volumes create standin_runs -a "$APP" -r "$REGION" -s 1 -y
fi

# 2) Secrets: only the keys the server reads, values never printed
push_secrets() {
  local keys="MODEL_API_KEY CSR_VLM_API_KEY CSR_VLM_BASE_URL STANDIN_WRITER_MODEL STANDIN_ACTOR_MODEL
    STANDIN_CHECK_MODEL STANDIN_IMAGE_MODEL STANDIN_TTS_MODEL STANDIN_HOURLY_LIMIT
    GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET STANDIN_SESSION_SECRET STANDIN_PUBLIC_URL
    STANDIN_ADMIN_EMAILS STANDIN_TEST_TOKEN STANDIN_CAP_TURN STANDIN_CAP_VOICE STANDIN_CAP_CREATE STANDIN_CAP_CHAT STANDIN_CAP_SCENE"
  local tmp; tmp=$(mktemp)
  trap 'rm -f "${tmp:-}"' RETURN EXIT
  for k in $keys; do
    grep -E "^$k=.+" .env | tr -d '\r' >> "$tmp" || true
  done
  [ -s "$tmp" ] || { echo "no keys found in .env"; exit 1; }
  echo "> secrets: $(cut -d= -f1 "$tmp" | tr '\n' ' ')"
  "$FLY" secrets import -a "$APP" --stage < "$tmp"
  rm -f "$tmp"
}
push_secrets
[ "${1:-}" = "--secrets" ] && { "$FLY" secrets deploy -a "$APP"; exit 0; }

# 3) Deploy on Fly's remote builder, so local Docker is not needed
echo "> deploying..."
"$FLY" deploy -a "$APP" --remote-only --ha=false --strategy immediate

# 4) The home page should answer
URL="https://$APP.fly.dev"
for _ in $(seq 1 20); do
  if curl -fsS -m 5 -o /dev/null "$URL/"; then
    echo "deployed: $URL"
    exit 0
  fi
  sleep 3
done
echo "home page did not answer. Logs: \"$FLY\" logs -a $APP"
exit 1
