#!/usr/bin/env bash
# Called by GitHub Actions over SSH. Required environment variables are validated
# before any application container is changed.
set -Eeuo pipefail

required=(API_IMAGE MLFLOW_IMAGE EC2_APP_DIR GHCR_TOKEN GHCR_USERNAME)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Required environment variable $variable is not set" >&2
    exit 1
  fi
done

if [[ ! -f "$EC2_APP_DIR/docker-compose.prod.yml" ]]; then
  echo "Missing $EC2_APP_DIR/docker-compose.prod.yml. Complete the one-time EC2 setup first." >&2
  exit 1
fi

sudo_cmd=()
if [[ "$(id -u)" -ne 0 ]]; then
  sudo_cmd=(sudo)
fi

if ! command -v docker >/dev/null 2>&1; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Docker is missing and this host is not Debian/Ubuntu. Install Docker Engine and Compose v2 first." >&2
    exit 1
  fi
  "${sudo_cmd[@]}" apt-get update
  "${sudo_cmd[@]}" apt-get install -y ca-certificates curl
  "${sudo_cmd[@]}" install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | "${sudo_cmd[@]}" gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  "${sudo_cmd[@]}" chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" | "${sudo_cmd[@]}" tee /etc/apt/sources.list.d/docker.list >/dev/null
  "${sudo_cmd[@]}" apt-get update
  "${sudo_cmd[@]}" apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

if ! command -v curl >/dev/null 2>&1; then
  "${sudo_cmd[@]}" apt-get update
  "${sudo_cmd[@]}" apt-get install -y curl
fi

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  DOCKER=("${sudo_cmd[@]}" docker)
fi

if ! "${DOCKER[@]}" compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required but unavailable." >&2
  exit 1
fi

echo "$GHCR_TOKEN" | "${DOCKER[@]}" login ghcr.io --username "$GHCR_USERNAME" --password-stdin
cd "$EC2_APP_DIR"

# Keep runtime configuration out of the image and repository.
umask 077
cat > .env <<EOF
API_IMAGE=$API_IMAGE
MLFLOW_IMAGE=$MLFLOW_IMAGE
API_PORT=${API_PORT:-8000}
EOF

# Pull before replacing the running container, keeping interruption minimal.
export API_IMAGE
export MLFLOW_IMAGE
"${DOCKER[@]}" compose -f docker-compose.prod.yml pull
"${DOCKER[@]}" compose -f docker-compose.prod.yml up -d --remove-orphans

for attempt in {1..24}; do
  if curl --fail --silent --show-error --max-time 5 http://127.0.0.1:"${API_PORT:-8000}"/health >/dev/null; then
    "${DOCKER[@]}" image prune -f --filter "until=168h"
    echo "Deployment healthy: $API_IMAGE"
    exit 0
  fi
  sleep 5
done

echo "Health check failed; the newly deployed container was not accepted." >&2
"${DOCKER[@]}" compose -f docker-compose.prod.yml ps
exit 1
