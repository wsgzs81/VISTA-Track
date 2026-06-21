#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"
BLENDER_BIN="${BLENDER:-blender}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-output/SynMVTrack_v7_3_true_mv_photometric_pilot20_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

"$PYTHON_BIN" generator/generate_dataset.py \
  --config configs/dataset_true_mv_v7_3_photometric_pilot20.yaml \
  --output "$OUT" \
  --render \
  --blender "$BLENDER_BIN" \
  --verbose 2>&1 | tee "$LOG_DIR/v7_3_pilot20_${RUN_STAMP}.log"

"$PYTHON_BIN" generator/audit_render_quality.py "$OUT" \
  --frames-per-seq 12 \
  --min-mask-pixels 48 \
  --color-max-rgb-dist 72.0 \
  --bbox-min-iou 0.58 \
  --bbox-max-area-ratio 1.9 2>&1 | tee "$LOG_DIR/v7_3_pilot20_${RUN_STAMP}.audit.txt"

