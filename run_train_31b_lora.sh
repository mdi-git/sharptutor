#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

. .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python scripts/train_gemma_sft.py \
  --model_name google/gemma-4-31B-it \
  --train_file data/sample_train.jsonl \
  --eval_file data/sample_eval.jsonl \
  --output_dir outputs/gemma4-31b-it-lora \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 2 \
  --save_steps 50 \
  --eval_steps 50 \
  --logging_steps 10
