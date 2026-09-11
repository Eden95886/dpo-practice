# 영어 Wikipedia 데이터셋 메모

Hugging Face `wikimedia/wikipedia` 의 영어 subset을 맥북에서 이어서 쓰기 위한 기록.

## 데이터셋

```python
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
```

| 항목 | 값 |
|------|------|
| repo | `wikimedia/wikipedia` |
| config | `20231101.en` |
| 스냅샷 | 2023-11-01 |
| 문서 수 | 약 641만 |
| 용량 | 약 11.6 GB (parquet 41개) |
| 필드 | `id`, `url`, `title`, `text` |
| 라이선스 | CC BY-SA 3.0 + GFDL |

같은 dump의 소수 언어 예: `20231101.ab`(압하지야어), `20231101.ace`(아체어), `20231101.ady`(아디게어). 영어는 `20231101.en`.

옛 이름 `load_dataset("wikipedia", ...)` 는 deprecated. 반드시 `wikimedia/wikipedia` 를 쓴다.

## 확인한 것 (Windows PC)

- Hugging Face CLI 로그인: `EdenKim955`
- `streaming=True` 로 전체 다운로드 없이 첫 문서 확인됨
- 첫 문서: `Anarchism` (id 12, 약 46,064자)
- 앞 문서 제목은 주제순이 아니라 page id 순. 사상/과학/지리/문화/스포츠가 섞여 있음

403이 나면 데이터셋이 비공개인 게 아니다. 영어판은 파일이 커서 익명 다운로드가 자주 거절된다. `hf auth login` 후 재시도.

## 언제 쓰면 좋은가

쓰기 좋음:

- 영어 LM pretrain / continued pretraining / tokenizer 학습
- RAG, 청킹, 임베딩, 검색 실험
- 긴 영어 문서로 전처리 파이프라인 점검

쓰지 않는 편이 나음:

- 2023-11-01 이후 최신 정보
- QA/분류 라벨이 필요한 벤치마크 (SQuAD, GLUE 등)
- 대화체, 리뷰, 한국어 작업
- 위키 전체로 Qwen LoRA를 이 PC에서 학습하는 것

주제 라벨은 없다. 과학만 필요하면 `title`/`text` 키워드로 걸러야 한다. 문서 수 기준으로는 인물, 지명, 생물 종, 영화/앨범, 스포츠가 많고, 철학·수학 같은 핵심 항목은 수는 적지만 본문이 길다.

## Qwen + LoRA 학습 가능 여부

Windows PC (`i7-14700K`, RAM 32GB, Intel UHD 770, `torch 2.9.1+cpu`, CUDA 없음):

- 실용 학습 불가. LoRA/QLoRA는 NVIDIA GPU(또는 맥의 MPS)가 필요함
- 위키 전체(641만 문서)는 GPU가 있어도 과함. 샘플/주제 필터가 맞음
- 이 PC는 전처리·스트리밍 확인용

맥북: 이 저장소의 기존 DPO 실험이 MPS에서 돌아갔음 (`dpo_practice.py` 의 MPS OOM 기록 참고). Qwen2.5-1.5B + LoRA는 맥에서 소량 샘플로 시도하는 편이 맞음. 위키 원문 그대로보다 instruction/completion 형태로 바꾸는 게 DPO/SFT에 적합함.

## 맥북에서 열기

```bash
git clone https://github.com/Eden95886/dpo-practice.git
cd dpo-practice
pip install datasets huggingface_hub
hf auth login
python wikipedia/stream_wikipedia.py --n 20
```

앞 1000개만 디스크에 받으려면:

```python
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1000]")
```
