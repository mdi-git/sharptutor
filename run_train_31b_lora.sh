#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

. .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python -c "import sys, torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "CUDA is not available in the current Python environment."
  echo "Check nvidia-smi, torch.version.cuda, and reinstall a driver-compatible PyTorch wheel before training."
  echo "Recommended reinstall example:"
  echo "  rm -rf .venv"
  echo "  python3 -m venv .venv"
  echo "  . .venv/bin/activate"
  echo "  pip install --upgrade pip"
  echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
  echo "  pip install -r requirements.txt"
  exit 1
}

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
