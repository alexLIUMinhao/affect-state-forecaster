#!/usr/bin/env bash

set -euo pipefail

TRAIN_PATH="${TRAIN_PATH:-data/processed/pheme_forecast_ratio_05_train.jsonl}"
VAL_PATH="${VAL_PATH:-data/processed/pheme_forecast_ratio_05_val.jsonl}"
TEST_PATH="${TEST_PATH:-data/processed/pheme_forecast_ratio_05_test.jsonl}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/first_round_ratio_05}"

if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(text_baseline temporal_baseline structure_baseline affect_state_forecaster)
fi

mkdir -p "$OUTPUT_ROOT"

for model in "${MODELS[@]}"; do
  artifact_dir="$OUTPUT_ROOT/$model/artifacts"
  eval_dir="$OUTPUT_ROOT/$model/eval"
  mkdir -p "$artifact_dir" "$eval_dir"

  echo "==> training model=$model device=$DEVICE"
  python src/train.py \
    --train_path "$TRAIN_PATH" \
    --val_path "$VAL_PATH" \
    --model "$model" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$artifact_dir"

  prefix="${model}_$(basename "$TRAIN_PATH" .jsonl)"
  model_path="$artifact_dir/${prefix}.pt"
  config_path="$artifact_dir/${prefix}.json"

  echo "==> evaluating model=$model"
  python src/evaluate.py \
    --data_path "$TEST_PATH" \
    --model "$model" \
    --model_path "$model_path" \
    --config_path "$config_path" \
    --device "$DEVICE" \
    --output_dir "$eval_dir"
done

echo "completed_first_round=$OUTPUT_ROOT"
