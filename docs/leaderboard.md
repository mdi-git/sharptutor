# Leaderboard Guide

## 어떤 리더보드를 노릴지 먼저 정리

가장 보편적인 공개 리더보드는 Hugging Face Open LLM Leaderboard입니다. 다만 이 리더보드는 범용 지식, 추론, 수학, 코딩 계열 점수를 봅니다. 고객지원, 사내 문서 QA, 특정 한국어 도메인 같은 좁은 목적을 강하게 학습하면 오히려 범용 점수가 떨어질 수 있습니다.

즉:

- 범용 리더보드 점수가 목표면 데이터는 좁은 업무형 답변만 넣지 말고 일반 추론, 수학, 코딩, 한국어/영어 혼합 지시 데이터를 같이 넣어야 합니다.
- 실제 제품 성능이 목표면 내부 태스크 기준 평가셋을 따로 만들고 그 점수를 우선해야 합니다.

## Open LLM Leaderboard 제출 절차

2026-05-07 기준 Hugging Face 제출 문서 핵심 조건:

- 모델은 public 이어야 합니다.
- `AutoConfig`, `AutoModel`, `AutoTokenizer`로 로드 가능해야 합니다.
- `Safetensors` 형식이 권장됩니다.
- `use_remote_code=True`가 필요하면 자동 평가가 제한될 수 있습니다.

실무 순서:

1. LoRA adapter만 올릴지, 병합 모델까지 올릴지 결정합니다.
2. 리더보드 제출은 보통 병합 모델이 더 단순합니다.
3. Hugging Face Hub에 모델 카드와 라이선스를 채워 넣습니다.
4. 로컬에서 아래 로드 테스트를 먼저 통과시킵니다.

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

model_id = "your-org/your-model"
config = AutoConfig.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

5. 리더보드 제출 페이지에서 모델명, precision, revision, chat template 옵션을 맞춰 제출합니다.

제출 문서:

- https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/submitting

## 점수를 올리기 위한 조언

- `gemma-4-31B-it` 자체가 이미 강한 범용 모델이므로, 소량의 편향된 데이터로 세게 학습하면 평균 점수는 쉽게 떨어집니다.
- 리더보드 목적이면 `full SFT`보다 먼저 `small high-quality SFT` 또는 `DPO/ORPO` 계열 미세 정렬을 검토하는 편이 안전합니다.
- 좁은 사내 고객지원 데이터만으로는 리더보드 상승보다 제품 특화 성능 개선 쪽이 더 현실적입니다.
- 공개 벤치 상승을 원하면 합성 데이터로 일반 추론, 코딩, 수학, 한국어 지시 수행 데이터를 섞어야 합니다.
- 데이터 품질이 양보다 중요합니다. 중복, 장황한 chain-of-thought, 잘못된 정답은 바로 점수를 깎습니다.

## 추천 검증 루틴

1. 내부 dev set 200~1000개를 먼저 고정합니다.
2. zero-shot base vs LoRA tuned 를 exact match, win rate, human eval로 비교합니다.
3. 개선이 확인되면 그때만 학습량과 데이터량을 늘립니다.
4. 마지막에 병합 모델을 만들어 공개 벤치 제출용으로 고정합니다.
