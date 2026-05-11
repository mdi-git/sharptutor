# Gemma 4 31B Fine-Tuning Starter

이 작업공간은 `google/gemma-4-31B-it` 기준의 실전형 QLoRA 학습 템플릿입니다.

중요한 점:

- `google/gemma-4-31B-it-assistant`는 일반적인 메인 챗 모델이 아니라 MTP speculative decoding용 drafter입니다.
- 즉 일반적인 성능 향상용 파인튜닝은 보통 `google/gemma-4-31B-it`를 대상으로 하고, `-assistant`는 추론 가속에 붙입니다.
- 현재 공개 상태는 2026-05-07 기준 Hugging Face 모델 카드에서 확인했습니다.

## 파일 구성

- `requirements.txt`: 학습 의존성
- `data/sample_train.jsonl`: 학습 예시
- `data/sample_eval.jsonl`: 평가 예시
- `scripts/train_gemma_sft.py`: QLoRA SFT 학습
- `scripts/merge_lora.py`: LoRA 병합
- `scripts/infer_with_assistant.py`: 본체 + assistant drafter 추론 예시
- `run_train_31b_lora.sh`: 학습 실행 래퍼
- `run_merge_31b_lora.sh`: 병합 실행 래퍼
- `run_infer_31b_with_assistant.sh`: 추론 실행 래퍼
- `docs/leaderboard.md`: Open LLM Leaderboard 제출 가이드

## 빠른 시작

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

중요:

- `torch`는 서버 드라이버와 CUDA 호환성이 가장 민감하므로 `requirements.txt`에 넣지 않았습니다.
- 드라이버가 오래된 서버에서는 최신 PyTorch가 GPU를 못 잡을 수 있습니다.
- 설치 직후 아래 검사를 통과해야 학습을 시작해야 합니다.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
nvidia-smi
```

`torch.cuda.is_available()`가 `True`가 아니면 학습을 돌리지 말고 PyTorch CUDA wheel 또는 NVIDIA 드라이버를 먼저 맞춰야 합니다.

학습:

```bash
python scripts/train_gemma_sft.py \
  --model_name google/gemma-4-31B-it \
  --train_file data/sample_train.jsonl \
  --eval_file data/sample_eval.jsonl \
  --output_dir outputs/gemma4-31b-it-lora
```

병합:

```bash
python scripts/merge_lora.py \
  --base_model google/gemma-4-31B-it \
  --adapter_path outputs/gemma4-31b-it-lora \
  --output_dir outputs/gemma4-31b-it-merged
```

assistant drafter를 붙여 추론:

```bash
python scripts/infer_with_assistant.py \
  --target_model outputs/gemma4-31b-it-merged \
  --assistant_model google/gemma-4-31B-it-assistant \
  --prompt "환불 정책 문의에 답하는 가이드를 짧게 써줘."
```

또는 래퍼 스크립트 사용:

```bash
bash run_train_31b_lora.sh
bash run_merge_31b_lora.sh
bash run_infer_31b_with_assistant.sh
```

## 데이터 형식

권장 형식은 1줄 1샘플 JSONL입니다.

```json
{"messages":[
  {"role":"system","content":"You are a concise Korean enterprise assistant."},
  {"role":"user","content":"고객 질문"},
  {"role":"assistant","content":"모범 답변"}
]}
```

이 형식은 TRL의 대화형 SFT 흐름과 잘 맞고, Gemma 4의 `system/user/assistant` 역할 구조를 직접 반영하기 좋습니다.

## 하드웨어 메모

- QLoRA 4bit 기준으로도 31B는 가볍지 않습니다.
- 보수적으로는 `H100 80GB` 또는 `A100 80GB` 1장 이상을 권장합니다.
- 더 작은 GPU에서는 `max_seq_length`, 배치 크기, gradient accumulation을 더 공격적으로 낮춰야 합니다.
- 풀 파인튜닝은 별도 분산 학습 구성이 사실상 필요합니다.
- 현재 이 머신은 `RTX A5000 24GB` 1장이라 `gemma-4-31B-it` 학습을 바로 돌리기에는 메모리 여유가 부족합니다.

위 하드웨어 수치는 공개 문서의 직접 수치가 아니라 31B dense 모델 크기와 QLoRA 메모리 특성에 근거한 실무 추정치입니다.
