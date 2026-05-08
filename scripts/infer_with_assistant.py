import argparse

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gemma 4 target model with MTP assistant model.")
    parser.add_argument("--target_model", default="google/gemma-4-31B-it")
    parser.add_argument("--assistant_model", default="google/gemma-4-31B-it-assistant")
    parser.add_argument("--prompt", default="한국어 고객지원 챗봇의 응답 원칙을 3문장으로 설명해줘.")
    parser.add_argument("--enable_thinking", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.target_model)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    assistant_model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    system_text = "You are a helpful Korean assistant."
    if args.enable_thinking:
        system_text = "<|think|>\n" + system_text

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": args.prompt},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    inputs = processor(text=text, return_tensors="pt").to(target_model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = target_model.generate(
        **inputs,
        assistant_model=assistant_model,
        max_new_tokens=256,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    print(processor.parse_response(response))


if __name__ == "__main__":
    main()
