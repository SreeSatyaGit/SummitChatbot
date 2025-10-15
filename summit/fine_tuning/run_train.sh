#!/usr/bin/env bash
set -euo pipefail

# Example runner for fine_tuning/train_gpt2.py
# Usage: run_train.sh [train_file] [output_dir] [model] [batch] [epochs] [max_seq] [lr] [weight_decay] [save_steps] [fp16]
# Defaults are tuned for the onboarding dataset in fine_tuning/data/
DATA=${1:-fine_tuning/data/onboarding_train.jsonl}
OUT=${2:-fine_tuning/models/gpt2-onboarding}
MODEL=${3:-gpt2}
BATCH=${4:-4}
EPOCHS=${5:-3}
MAX_SEQ=${6:-512}
LR=${7:-5e-5}
WEIGHT_DECAY=${8:-0.0}
SAVE_STEPS=${9:-500}
FP16=${10:-false}

mkdir -p "$OUT"

cmd=(python3 fine_tuning/train_gpt2.py
  --train-file "$DATA"
  --output-dir "$OUT"
  --model-name-or-path "$MODEL"
  --per-device-train-batch-size "$BATCH"
  --num-train-epochs "$EPOCHS"
  --max-seq-length "$MAX_SEQ"
  --learning-rate "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --save-steps "$SAVE_STEPS"
)

if [ "$FP16" = "true" ] || [ "$FP16" = "1" ]; then
  cmd+=(--fp16)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
