#!/usr/bin/env bash
set -euo pipefail

workspace="${VISTA_TRAIN_WORKSPACE:-$(pwd)}"
config="${VISTA_STAGE1_CONFIG:-stage1_real_qclean}"
save_dir="${VISTA_SAVE_DIR:-./output}"
gpus="${VISTA_GPUS:-2}"

cd "$workspace"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

python tracking/train.py \
  --script vista_stage1 \
  --config "$config" \
  --save_dir "$save_dir" \
  --mode multiple \
  --nproc_per_node "$gpus" \
  --use_lmdb 0 \
  --use_wandb 0
