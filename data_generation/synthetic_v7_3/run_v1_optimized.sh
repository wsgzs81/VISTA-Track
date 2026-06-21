#!/bin/bash
set -euo pipefail

# Optimized SynMVTrack V1 pipeline for the current /root/datasets server layout.
# Environment overrides:
#   NUM_SEQUENCES=500 OUTPUT=output/SynMVTrack_v1_full GPUS=0,1 SAMPLES=16 bash run_v1_optimized.sh

ROOT="${SYNMVTRACK_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"

BLENDER="${BLENDER:-/root/datasets/tools/blender/blender-4.2.1-linux-x64/blender}"
CONFIG="${CONFIG:-configs/dataset_v1.yaml}"
OUTPUT="${OUTPUT:-output/SynMVTrack_v1_full}"
NUM_SEQUENCES="${NUM_SEQUENCES:-500}"
GPUS_CSV="${GPUS:-0,1}"
SAMPLES="${SAMPLES:-16}"
SEED="${SEED:-42}"
FRAMES_CHOICES="${FRAMES_CHOICES:-}"
RESOLUTION_CHOICES="${RESOLUTION_CHOICES:-}"
SAMPLES_CHOICES="${SAMPLES_CHOICES:-}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "No GPUs configured. Set GPUS=0,1"
  exit 1
fi

if [ ! -x "$BLENDER" ]; then
  echo "Blender not found or not executable: $BLENDER"
  exit 1
fi

TMP_CONFIG="$ROOT/configs/.synmvtrack_v1_optimized_${NUM_SEQUENCES}.yaml"
python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
cfg["dataset"]["num_sequences"] = int("$NUM_SEQUENCES")
if "$FRAMES_CHOICES":
    cfg["dataset"]["frames_per_sequence_choices"] = [int(x) for x in "$FRAMES_CHOICES".split(",") if x]
if "$RESOLUTION_CHOICES":
    cfg["render"]["resolution_choices"] = [
        [int(v) for v in item.lower().split("x")]
        for item in "$RESOLUTION_CHOICES".split(",") if item
    ]
if "$SAMPLES_CHOICES":
    cfg["render"]["samples_choices"] = [int(x) for x in "$SAMPLES_CHOICES".split(",") if x]
open("$TMP_CONFIG", "w").write(yaml.safe_dump(cfg, sort_keys=False))
print("Wrote", "$TMP_CONFIG")
PY

echo "== Metadata =="
python3 generator/generate_dataset.py \
  --config "$TMP_CONFIG" \
  --output "$OUTPUT" \
  --step metadata \
  --seed "$SEED"

echo "== Render =="
mapfile -t SEQS < <(find "$OUTPUT" -maxdepth 1 -type d -name 'seq_*' -printf '%f\n' | sort)
TOTAL="${#SEQS[@]}"
echo "Sequences: $TOTAL"

render_seq() {
  local seq="$1"
  local gpu="$2"
  local seq_dir="$OUTPUT/$seq"
  local res
  res=$(python3 - <<PY
import json
a = json.load(open("$seq_dir/attributes.json"))
print(a.get("resolution", [640, 480])[0], a.get("resolution", [640, 480])[1])
PY
)
  read -r res_w res_h <<< "$res"
  CUDA_VISIBLE_DEVICES="$gpu" "$BLENDER" --background --python generator/render_sequence.py -- \
    --seq-dir "$(pwd)/$seq_dir" \
    --resolution "$res_w" "$res_h" \
    --samples "$SAMPLES" \
    --save-mask \
    > "/tmp/render_${seq}_gpu${gpu}.log" 2>&1
  echo "rendered $seq on GPU $gpu"
}

running=0
for i in "${!SEQS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  render_seq "${SEQS[$i]}" "$gpu" &
  running=$((running + 1))
  if [ "$running" -ge "${#GPUS[@]}" ]; then
    wait
    running=0
  fi
done
wait

echo "== Export annotations =="
python3 generator/export_annotations.py --output-dir "$OUTPUT"

echo "== MVTrack format and initial splits =="
python3 - <<PY
import os
import yaml
from generator.write_mvtrack_format import write_mvtrack_format, generate_splits

output = "$OUTPUT"
cfg = yaml.safe_load(open("$TMP_CONFIG"))
seq_ids = sorted(d for d in os.listdir(output) if d.startswith("seq_") and os.path.isdir(os.path.join(output, d)))
for i, sid in enumerate(seq_ids, 1):
    write_mvtrack_format(os.path.join(output, sid), cfg=cfg)
    if i % 25 == 0:
        print(f"  formatted {i}/{len(seq_ids)}")
generate_splits(output, split_cfg=cfg.get("split"))
PY

echo "== Validate and discard bad sequences =="
python3 validation/run_validation.py --output-dir "$OUTPUT" --discard || true

echo "== Regenerate splits after discard =="
python3 - <<PY
import os
import yaml
from generator.write_mvtrack_format import generate_splits
cfg = yaml.safe_load(open("$TMP_CONFIG"))
generate_splits("$OUTPUT", split_cfg=cfg.get("split"))
PY

echo "V1 optimized pipeline complete: $OUTPUT"
