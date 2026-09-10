"""
DPO 실습 - LoRA + Qwen2.5-1.5B-Instruct (helpful-base)
gpt2(124M) full fine-tuning의 반복/헛소리 문제가 모델 용량 한계였는지 확인하기 위해
instruction-tuned된 더 큰 모델로 같은 helpful-base 레시피를 재현한다.
LoRA를 쓰면 reference 모델을 따로 메모리에 올리지 않고(어댑터를 끈 base가 reference 역할)
학습 파라미터도 적어 MPS 메모리 부담이 훨씬 적다.
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_TRAIN_SAMPLES = 1000
N_EVAL_SAMPLES = 150
NUM_EPOCHS = 4
OUTPUT_DIR = "/private/tmp/claude-501/-Users-kjw/8a9d94ac-4e2d-4a10-9950-600f5bd81ae0/scratchpad/dpo-qwen15b-lora-helpful"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def split_prompt_response(example):
    def split(text):
        idx = text.rfind("\n\nAssistant:")
        prompt = text[: idx + len("\n\nAssistant:")]
        response = text[idx + len("\n\nAssistant:") :]
        return prompt, response

    prompt_c, chosen = split(example["chosen"])
    _, rejected = split(example["rejected"])
    return {"prompt": prompt_c, "chosen": chosen, "rejected": rejected}

raw = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
train_ds = raw["train"].select(range(N_TRAIN_SAMPLES)).map(split_prompt_response)
eval_ds = raw["test"].select(range(N_EVAL_SAMPLES)).map(split_prompt_response)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=1e-4,        # LoRA는 학습 파라미터가 적어 full FT보다 LR을 높게 잡는 게 일반적
    beta=0.1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
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
    peft_config=lora_config,
    callbacks=[
        MpsCacheClearCallback(),
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0005),
    ],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Saved LoRA adapter to {OUTPUT_DIR}")
print(f"Best eval_loss: {trainer.state.best_metric} at step {getattr(trainer.state, 'best_global_step', '?')}")
