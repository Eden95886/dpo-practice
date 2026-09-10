"""
DPO best 모델(step 550) vs 원본 gpt2 응답 비교
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/private/tmp/claude-501/-Users-kjw/8a9d94ac-4e2d-4a10-9950-600f5bd81ae0/scratchpad/dpo-gpt2-output-v3"
BASE_MODEL = "gpt2"
N_PROMPTS = 5

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dpo_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()

raw = load_dataset("Anthropic/hh-rlhf")["test"]

def get_prompt(example):
    idx = example["chosen"].rfind("\n\nAssistant:")
    return example["chosen"][: idx + len("\n\nAssistant:")]

# 학습에 쓰지 않은 뒷부분 test 샘플 사용 (앞 300개는 eval에 사용됐음)
prompts = [get_prompt(raw[i]) for i in range(400, 400 + N_PROMPTS)]

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=192).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()

for i, prompt in enumerate(prompts):
    print(f"\n{'='*80}\n[{i}] PROMPT:\n{prompt}")
    print(f"\n--- base gpt2 ---\n{generate(base_model, prompt)}")
    print(f"\n--- DPO (best, step 550) ---\n{generate(dpo_model, prompt)}")
