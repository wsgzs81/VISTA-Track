#!/usr/bin/env bash
set -euo pipefail

workspace="${VISTA_TRAIN_WORKSPACE:-$(pwd)}"
save_dir="${VISTA_SAVE_DIR:-./output}"
warmup_ckpt="${VISTA_WARMUP_CKPT:-$save_dir/checkpoints/train/vista_stage1/stage1_real_warmup/VISTATrackStage1_ep0020.pth.tar}"

cd "$workspace"

if [[ ! -f "$warmup_ckpt" ]]; then
  echo "waiting_for=$warmup_ckpt"
  exit 2
fi

if pgrep -af "run_training.py.*stage1_real_qclean|tracking/train.py.*stage1_real_qclean" >/dev/null; then
  echo "stage1_real_qclean_already_running"
  exit 0
fi

if pgrep -af "run_training.py.*stage1_real_warmup|tracking/train.py.*stage1_real_warmup" >/dev/null; then
  echo "stopping_warmup_after_ep0020"
  pkill -f "run_training.py.*stage1_real_warmup|tracking/train.py.*stage1_real_warmup" || true
  sleep 10
fi

nohup bash training/scripts/train_stage1_real_qclean.sh \
  > logs/train_stage1_real_qclean.log 2>&1 &
pid=$!
echo "$pid" > logs/train_stage1_real_qclean.pid
echo "started_pid=$pid"
