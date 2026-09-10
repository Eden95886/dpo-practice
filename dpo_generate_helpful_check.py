"""
helpful-only DPO 모델(best, step 300) 응답 품질 점검
- helpful-base test split에서 다양한 인덱스로 8개 프롬프트 샘플링 (실사용에 가까운 일반 질문 위주)
- base gpt2 vs DPO helpful-only 비교
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DPO_DIR = "/private/tmp/claude-501/-Users-kjw/8a9d94ac-4e2d-4a10-9950-600f5bd81ae0/scratchpad/dpo-gpt2-output-helpful"
BASE_MODEL = "gpt2"
INDICES = [0, 50, 150, 250, 350, 450, 550, 650]

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
dpo_model = AutoModelForCausalLM.from_pretrained(DPO_DIR).to(device).eval()

raw = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")["test"]

def get_prompt(example):
    idx = example["chosen"].rfind("\n\nAssistant:")
    return example["chosen"][: idx + len("\n\nAssistant:")]

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=192).to(device)
    torch.manual_seed(0)
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

for idx in INDICES:
    prompt = get_prompt(raw[idx])
    print(f"\n{'='*90}\n[test idx {idx}] PROMPT:\n{prompt}\n")
    print(f"--- base gpt2 ---\n{generate(base_model, prompt)}\n")
    print(f"--- DPO helpful-only ---\n{generate(dpo_model, prompt)}\n")
