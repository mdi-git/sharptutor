#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

source .venv/bin/activate

python scripts/merge_lora.py \
  --base_model google/gemma-4-31B-it \
  --adapter_path outputs/gemma4-31b-it-lora \
  --output_dir outputs/gemma4-31b-it-merged \
  --use_bf16
