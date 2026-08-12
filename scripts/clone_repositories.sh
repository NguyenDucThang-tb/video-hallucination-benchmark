#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$PROJECT_DIR/external"

clone_if_missing() {
  local url="$1"
  local destination="$2"
  local commit="$3"
  if [[ -d "$destination/.git" ]]; then
    echo "Present: $destination ($(git -C "$destination" rev-parse HEAD))"
  else
    git clone "$url" "$destination"
  fi
  git -C "$destination" checkout --detach "$commit"
}

clone_if_missing https://github.com/patrick-tssn/VideoHallucer "$PROJECT_DIR/external/VideoHallucer" 8b785d1680465911cd2ce80c9f652837c0ba2abd
clone_if_missing https://github.com/Stevetich/EventHallusion "$PROJECT_DIR/external/EventHallusion" aa544c21c7cd93b4685423cb94f77ab441f754bc
clone_if_missing https://github.com/CyL97/VidHalluc "$PROJECT_DIR/external/VidHalluc" e753864f5c2500c38523f97992355e2352bf8732
