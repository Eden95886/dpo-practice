# DPO 실습

GitHub: https://github.com/Eden95886/dpo-practice

Anthropic/hh-rlhf 데이터셋으로 DPO(Direct Preference Optimization)를 실습한 기록.
모델 가중치(수 GB~수십 GB)는 크기 때문에 이 저장소에 포함하지 않았고, 스크립트/학습 로그/결과 그래프만 커밋되어 있다.

## 실험 순서

1. `dpo_practice.py` — gpt2 full fine-tuning 1차/2차 시도 (1,000샘플/1epoch → 5,000샘플/2epoch).
   2차에서 MPS 메모리 누적으로 OOM 발생, `torch.mps.empty_cache()` 콜백 추가로 해결.
   로그: `dpo_train_v2.log`
2. gpt2, 3,000샘플, early stopping 추가 — eval loss가 step 550에서 최저점(0.638)을 찍고
   과적합으로 반등하는 지점을 관찰. 로그: `dpo_train_v3.log`
3. `dpo_practice_split.py` — helpful-base / harmless-base 단독 학습 비교.
   로그: `dpo_train_helpful.log`, `dpo_train_harmless.log`
   (참고: `load_dataset("Anthropic/hh-rlhf")`를 `data_dir` 없이 호출하면 harmless-base가
   맨 앞에 오도록 concat되는 버그성 동작이 있어, 2번 실험의 "mixed" 모델은 사실상
   harmless-base만 학습한 것과 동일했다.)
4. `dpo_practice_mixed.py` — helpful-base + harmless-base를 예제 단위로 셔플해 진짜로 섞은 실험.
   best eval loss 0.694로 거의 랜덤 수준 (학습이 거의 안 됨). 로그: `dpo_train_mixed_fixed.log`
5. `dpo_practice_alternating.py` — optimizer step(8샘플) 단위로 helpful/harmless를 번갈아 배치.
   `SequentialSampler`로 재셔플을 막음. best eval loss 0.670. 로그: `dpo_train_alternating.log`
6. `dpo_practice_curriculum.py` — helpful-only 체크포인트에서 harmless-base로 이어서 학습(curriculum).
   eval loss가 계속 악화(catastrophic forgetting)되며 5가지 중 최악의 결과.
   로그: `dpo_train_curriculum.log`
7. `dpo_practice_qwen_lora.py` — Qwen2.5-1.5B-Instruct + LoRA로 helpful-base만 학습.
   gpt2 helpful-only(0.624)보다 적은 데이터(1,000개)/적은 step(200)으로 더 낮은 best eval
   loss(0.589) 달성. 로그: `dpo_train_qwen_lora.log`

## 결과 요약 (best eval loss, 낮을수록 좋음)

| 실험 | best eval loss |
|---|---|
| Qwen2.5-1.5B + LoRA (helpful-only) | 0.589 |
| gpt2 helpful-only | 0.624 |
| gpt2 harmless-only | 0.638 |
| gpt2 mixed (step 단위 번갈아) | 0.670 |
| gpt2 mixed (예제 단위 셔플) | 0.694 |
| gpt2 curriculum (helpful→harmless) | 0.716 (계속 악화) |

## 응답 비교 스크립트

- `dpo_generate_test.py`, `dpo_generate_compare.py`, `dpo_generate_helpful_check.py`,
  `dpo_generate_qwen_compare.py` — base/DPO 모델 응답 비교용 생성 스크립트

## 그래프

`dpo_loss.html` — 6개 실험의 eval loss 곡선 비교 (인터랙티브 차트).

## 핵심 결론

1. Early stopping으로 과적합 turning point를 정량적으로 탐지할 수 있었다.
2. Helpfulness/harmlessness는 서로 상충하는 목표라, 단순 예제 셔플 mix나 순차 curriculum으로는
   두 목표를 동시에 잘 학습시키기 어려웠다. step 단위로 소스를 번갈아 넣는 방식이 셔플보다는 나았다.
3. gpt2(124M)에서 관찰된 응답 반복/헛소리 문제는 DPO 방법론의 결함이 아니라 base 모델 자체의
   표현력 한계였다 — Qwen2.5-1.5B + LoRA로 같은 레시피를 재현하자 훨씬 안정적인 결과를 얻었다.

## 영어 Wikipedia (이어서 할 작업)

`wikipedia/` — Hugging Face `wikimedia/wikipedia` / `20231101.en` 메모와 스트리밍 스크립트.

- 2023-11-01 영어 위키 덤프. 약 641만 문서, 11.6GB. 필드는 `id`, `url`, `title`, `text`.
- Windows PC에서 `streaming=True`로 첫 문서(`Anarchism`)까지 확인함. 전체 다운로드는 하지 않음.
- 영어판은 용량 때문에 익명이면 403이 나기 쉬움. Hugging Face 로그인 필요.
- Windows PC는 CUDA GPU가 없어서 Qwen+LoRA 학습은 비추천. 맥북(MPS)에서 소량 샘플로 이어가는 용도.

```bash
python wikipedia/stream_wikipedia.py --n 20
```
