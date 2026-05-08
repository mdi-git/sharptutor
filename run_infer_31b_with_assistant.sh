#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

source .venv/bin/activate

python scripts/infer_with_assistant.py \
  --target_model outputs/gemma4-31b-it-merged \
  --assistant_model google/gemma-4-31B-it-assistant \
  --prompt "환불 정책 문의에 답하는 가이드를 짧게 써줘."
