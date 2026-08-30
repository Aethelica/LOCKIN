#!/usr/bin/env bash
# Downloads the MediaPipe face landmarker model (~3.6 MB).
# Gitignored, so a fresh clone needs this once before running the demo.
set -euo pipefail

DEST="models/face_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

mkdir -p models
if [ -f "$DEST" ]; then
  echo "already present: $DEST"
  exit 0
fi

echo "downloading face landmarker model..."
curl -fL --progress-bar -o "$DEST" "$URL"
echo "saved to $DEST"
