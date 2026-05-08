import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base Gemma model.")
    parser.add_argument("--base_model", default="google/gemma-4-31B-it")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_bf16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
