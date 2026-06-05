#!/usr/bin/env bash
set -o pipefail

ROOT_DIR=/data/fzj/RadioDiff
CFG=./configs/first_dynamic_radio.yaml
SESSION_LOG="$ROOT_DIR/logs/watch_dynamic_radio_vae_gpu6.log"
RESTART_DELAY_SECONDS=30

mkdir -p "$ROOT_DIR/logs"
cd "$ROOT_DIR" || exit 1

echo "WATCH_START=$(date)" >> "$SESSION_LOG"
echo "CFG=$CFG" >> "$SESSION_LOG"

while true; do
  run_id=$(date +%Y%m%d_%H%M%S)
  run_log="$ROOT_DIR/logs/train_dynamic_radio_vae_gpu6_run_${run_id}.log"

  {
    echo "RUN_START=$(date)"
    echo "RUN_LOG=$run_log"
    echo "CFG=$CFG"
  } >> "$SESSION_LOG"

  source /home/zjlab/miniconda3/etc/profile.d/conda.sh
  export CONDA_ENVS_PATH=/data/fzj/conda_envs
  conda activate radiodiff
  export CUDA_VISIBLE_DEVICES=6
  export PYTHONUNBUFFERED=1

  {
    echo "START_TIME=$(date)"
    echo "CONFIG=$CFG"
    python -u train_vae.py --cfg "$CFG"
    code=$?
    echo "EXIT_CODE=$code"
    echo "END_TIME=$(date)"
  } >> "$run_log" 2>&1

  code=$(tail -n 20 "$run_log" | sed -n 's/^EXIT_CODE=//p' | tail -n 1)
  code=${code:-unknown}

  {
    echo "RUN_END=$(date)"
    echo "EXIT_CODE=$code"
    echo
  } >> "$SESSION_LOG"

  if [[ "$code" == "0" ]]; then
    echo "WATCH_STOP=$(date) normal training completion" >> "$SESSION_LOG"
    exit 0
  fi

  sleep "$RESTART_DELAY_SECONDS"
done
