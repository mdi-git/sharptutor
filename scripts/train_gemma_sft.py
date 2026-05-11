import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA SFT for Gemma 4 chat models.")
    parser.add_argument("--model_name", default="google/gemma-4-31B-it")
    parser.add_argument("--train_file", default="data/sample_train.jsonl")
    parser.add_argument("--eval_file", default="data/sample_eval.jsonl")
    parser.add_argument("--output_dir", default="outputs/gemma4-31b-it-lora")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    return parser.parse_args()


def format_conversation(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    bf16_enabled = bool(args.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    fp16_enabled = bool(args.use_fp16 and not bf16_enabled)
    compute_dtype = torch.bfloat16 if bf16_enabled else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        dtype=compute_dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "gate_proj.linear",
            "up_proj.linear",
            "down_proj.linear",
        ],
    )

    data_files = {"train": args.train_file}
    if args.eval_file and os.path.exists(args.eval_file):
        data_files["eval"] = args.eval_file

    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=max(1, int(args.warmup_ratio * len(dataset["train"]))),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if "eval" in dataset else "no",
        save_strategy="steps",
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        max_length=args.max_seq_length,
        dataset_text_field="text",
    )

    if args.use_bf16 and not bf16_enabled:
        print("bf16 requested but not supported on this setup; falling back to fp16.")
        fp16_enabled = True
        training_args.fp16 = True
        training_args.bf16 = False

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"] if "eval" in dataset else None,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=lambda example: example["text"],
    )

    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable_params={trainable_params} total_params={total_params}")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
