"""
DPO 실습 - curriculum: helpful-base로 먼저 학습한 best 체크포인트에
이어서 harmless-base를 추가 학습.
평가는 helpful+harmless를 합친 combined eval set으로 해서,
harmless를 배우는 동안 helpful 능력을 얼마나 유지하는지 같이 본다.
"""
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from trl import DPOTrainer, DPOConfig

STAGE1_MODEL_DIR = "/private/tmp/claude-501/-Users-kjw/8a9d94ac-4e2d-4a10-9950-600f5bd81ae0/scratchpad/dpo-gpt2-output-helpful"
N_TRAIN_SAMPLES = 3000       # stage 2: harmless-base 3000개
N_EVAL_PER_SUBSET = 150      # helpful 150 + harmless 150 = combined eval 300 (다른 실험들과 동일)
NUM_EPOCHS = 6
OUTPUT_DIR = "/private/tmp/claude-501/-Users-kjw/8a9d94ac-4e2d-4a10-9950-600f5bd81ae0/scratchpad/dpo-gpt2-output-curriculum"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Stage 1 checkpoint: {STAGE1_MODEL_DIR}")

def split_prompt_response(example):
    def split(text):
        idx = text.rfind("\n\nAssistant:")
        prompt = text[: idx + len("\n\nAssistant:")]
        response = text[idx + len("\n\nAssistant:") :]
        return prompt, response

    prompt_c, chosen = split(example["chosen"])
    _, rejected = split(example["rejected"])
    return {"prompt": prompt_c, "chosen": chosen, "rejected": rejected}

helpful = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")

# stage 2 학습 데이터: harmless-base만
train_ds = harmless["train"].select(range(N_TRAIN_SAMPLES)).map(split_prompt_response)

# combined eval: helpful + harmless 둘 다 확인
eval_ds = concatenate_datasets([
    helpful["test"].select(range(N_EVAL_PER_SUBSET)),
    harmless["test"].select(range(N_EVAL_PER_SUBSET)),
]).shuffle(seed=42).map(split_prompt_response)

print(f"stage2 train (harmless-base): {len(train_ds)}, combined eval: {len(eval_ds)}")

tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(STAGE1_MODEL_DIR)

config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=5e-6,
    beta=0.1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
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
    callbacks=[
        MpsCacheClearCallback(),
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0005),
    ],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Saved fine-tuned model to {OUTPUT_DIR}")
print(f"Best eval_loss: {trainer.state.best_metric} at step {getattr(trainer.state, 'best_global_step', '?')}")
