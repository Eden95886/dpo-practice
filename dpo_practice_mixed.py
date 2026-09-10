"""
DPO 실습 - helpful-base + harmless-base를 실제로 균형있게 섞어서 학습.
(이전 "mixed(v3)"는 load_dataset 기본 concat 순서 버그로 harmless-base만 학습된 것이었음)
"""
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "gpt2"
N_PER_SUBSET_TRAIN = 1500   # helpful 1500 + harmless 1500 = 3000 (이전 실험들과 동일 규모)
N_PER_SUBSET_EVAL = 150     # 150 + 150 = 300
NUM_EPOCHS = 6
OUTPUT_DIR = "/private/tmp/claude-501/-Users-kjw/8a9d94ac-4e2d-4a10-9950-600f5bd81ae0/scratchpad/dpo-gpt2-output-mixed"

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

helpful = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")

train_ds = concatenate_datasets([
    helpful["train"].select(range(N_PER_SUBSET_TRAIN)),
    harmless["train"].select(range(N_PER_SUBSET_TRAIN)),
]).shuffle(seed=42).map(split_prompt_response)

eval_ds = concatenate_datasets([
    helpful["test"].select(range(N_PER_SUBSET_EVAL)),
    harmless["test"].select(range(N_PER_SUBSET_EVAL)),
]).shuffle(seed=42).map(split_prompt_response)

print(f"train: {len(train_ds)}, eval: {len(eval_ds)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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
