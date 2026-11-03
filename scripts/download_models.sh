#!/usr/bin/env bash
# Download MediaPipe Pose Landmarker model bundles into models/.
# Run on a machine with internet access. URLs follow the pattern in Google's
# MediaPipe Pose Landmarker docs (ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).
# NOT verified from the sandbox where this repo was written (storage.googleapis.com was blocked).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
BASE="https://storage.googleapis.com/mediapipe-models/pose_landmarker"
# Usage: scripts/download_models.sh [lite] [full] [heavy]   (default: lite full)
variants=("$@")
[ ${#variants[@]} -eq 0 ] && variants=(lite full)
for v in "${variants[@]}"; do
  url="$BASE/pose_landmarker_${v}/float16/latest/pose_landmarker_${v}.task"
  echo "GET $url"
  curl -fL --retry 3 -o "models/pose_landmarker_${v}.task" "$url"
  (shasum -a 256 "models/pose_landmarker_${v}.task" 2>/dev/null || sha256sum "models/pose_landmarker_${v}.task")
done
