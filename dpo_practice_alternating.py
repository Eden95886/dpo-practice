"""
DPO 실습 - helpful-base / harmless-base를 "optimizer step 단위"로 번갈아 넣기.
(이전 실습: 예제 단위로 shuffle한 balanced mix는 거의 학습이 안 됐음 - helpful/harmless gradient가
 매 step마다 서로 반대 방향으로 밀려서 상쇄된 것으로 추정)

여기서는 각 최적화 step(=per_device_train_batch_size * gradient_accumulation_steps 개 샘플)을
통째로 helpful 또는 harmless 하나의 소스로만 채우고, step마다 소스를 번갈아 가며 학습한다.
Trainer가 기본으로 매 epoch마다 RandomSampler로 다시 섞어버리기 때문에, 이 순서를 유지하려면
SequentialSampler를 쓰도록 Trainer를 오버라이드해야 한다.
"""
import torch
from torch.utils.data import SequentialSampler
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "gpt2"
N_PER_SUBSET_TRAIN = 1500
N_PER_SUBSET_EVAL = 150
NUM_EPOCHS = 6
BATCH = 2
GRAD_ACCUM = 4
CHUNK = BATCH * GRAD_ACCUM  # 8개 = optimizer step 1회 분량
OUTPUT_DIR = "/private/tmp/claude-501/-Users-kjw/8a9d94ac-4e2d-4a10-9950-600f5bd81ae0/scratchpad/dpo-gpt2-output-alternating"

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

helpful_train = helpful["train"].select(range(N_PER_SUBSET_TRAIN))
harmless_train = harmless["train"].select(range(N_PER_SUBSET_TRAIN))

# CHUNK(=8)개씩 helpful -> harmless -> helpful -> ... 번갈아 이어붙이기
n_chunks = N_PER_SUBSET_TRAIN // CHUNK
interleaved_indices_helpful = []
interleaved_indices_harmless = []
order = []  # 디버깅/확인용: 어느 소스가 몇 번째 chunk인지
for i in range(n_chunks):
    order.append(("helpful", i))
    order.append(("harmless", i))

pieces = []
for src, i in order:
    ds = helpful_train if src == "helpful" else harmless_train
    pieces.append(ds.select(range(i * CHUNK, (i + 1) * CHUNK)))

train_ds = concatenate_datasets(pieces).map(split_prompt_response)

eval_ds = concatenate_datasets([
    helpful["test"].select(range(N_PER_SUBSET_EVAL)),
    harmless["test"].select(range(N_PER_SUBSET_EVAL)),
]).shuffle(seed=42).map(split_prompt_response)

print(f"train: {len(train_ds)} ({n_chunks} chunks x 2 sources x {CHUNK}), eval: {len(eval_ds)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
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


class SequentialDPOTrainer(DPOTrainer):
    # 기본 Trainer는 RandomSampler를 써서 매 epoch마다 순서를 다시 섞는다.
    # helpful/harmless가 optimizer step 단위로 번갈아 오도록 만든 순서를 유지하려면
    # SequentialSampler를 강제해야 한다.
    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)


trainer = SequentialDPOTrainer(
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
