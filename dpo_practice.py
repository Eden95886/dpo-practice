"""
DPO(Direct Preference Optimization) 실습 스크립트
- 모델: gpt2 (124M, 가벼워서 CPU/MPS에서도 빠르게 돌아감)
- 데이터: Anthropic/hh-rlhf (앞부분 1000개만 사용, 실습용)
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "gpt2"
# 2차 실습(5000개, 2 epoch)에서는 eval loss가 끝까지 완만하게 계속 줄어서 과적합 turn을
# 못 봤다. epoch을 늘려 실제로 eval loss가 반등하는 지점을 early stopping으로 잡아본다.
N_TRAIN_SAMPLES = 3000
N_EVAL_SAMPLES = 300
NUM_EPOCHS = 6
OUTPUT_DIR = "/private/tmp/claude-501/-Users-kjw/8a9d94ac-4e2d-4a10-9950-600f5bd81ae0/scratchpad/dpo-gpt2-output-v3"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 1. 데이터 로드 (chosen/rejected는 "Human: ... Assistant: ..." 전체 대화 문자열)
#    DPOTrainer는 prompt / chosen / rejected 세 컬럼을 기대하므로
#    마지막 "Assistant:" 위치를 기준으로 prompt와 response를 분리한다.
def split_prompt_response(example):
    def split(text):
        idx = text.rfind("\n\nAssistant:")
        prompt = text[: idx + len("\n\nAssistant:")]
        response = text[idx + len("\n\nAssistant:") :]
        return prompt, response

    prompt_c, chosen = split(example["chosen"])
    _, rejected = split(example["rejected"])
    return {"prompt": prompt_c, "chosen": chosen, "rejected": rejected}

raw = load_dataset("Anthropic/hh-rlhf")
train_ds = raw["train"].select(range(N_TRAIN_SAMPLES)).map(split_prompt_response)
eval_ds = raw["test"].select(range(N_EVAL_SAMPLES)).map(split_prompt_response)

# 2. 모델 / 토크나이저
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 3. DPO 설정
config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=5e-6,
    beta=0.1,                 # DPO의 KL penalty 계수 (preference 강도 조절)
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,             # load_best_model_at_end을 쓰려면 save_steps가 eval_steps의 배수여야 함
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=[],
    max_length=384,
    max_prompt_length=192,
    bf16=False,
    use_mps_device=(device == "mps"),
)


# MPS는 step마다 캐시가 누적되어 장시간 학습 시 OOM이 나는 경우가 있어 주기적으로 비워준다.
class MpsCacheClearCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if device == "mps":
            torch.mps.empty_cache()


trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    callbacks=[
        MpsCacheClearCallback(),
        # 최근 3번의 eval(=150 step) 동안 eval_loss가 0.0005 이상 개선되지 않으면 중단
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0005),
    ],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Saved fine-tuned model to {OUTPUT_DIR}")
print(f"Best eval_loss: {trainer.state.best_metric} at step {trainer.state.best_global_step if hasattr(trainer.state, 'best_global_step') else '?'}")
