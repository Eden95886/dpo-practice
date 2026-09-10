"""
base gpt2 vs DPO(mixed v3) vs DPO(helpful-only) vs DPO(harmless-only) 응답 비교
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/private/tmp/claude-501/-Users-kjw/8a9d94ac-4e2d-4a10-9950-600f5bd81ae0/scratchpad"
MODELS = {
    "base gpt2": "gpt2",
    "DPO mixed (v3)": f"{BASE}/dpo-gpt2-output-v3",
    "DPO helpful-only": f"{BASE}/dpo-gpt2-output-helpful",
    "DPO harmless-only": f"{BASE}/dpo-gpt2-output-harmless",
}
N_PROMPTS = 5

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

models = {name: AutoModelForCausalLM.from_pretrained(path).to(device).eval() for name, path in MODELS.items()}

raw = load_dataset("Anthropic/hh-rlhf")["test"]

def get_prompt(example):
    idx = example["chosen"].rfind("\n\nAssistant:")
    return example["chosen"][: idx + len("\n\nAssistant:")]

prompts = [get_prompt(raw[i]) for i in range(400, 400 + N_PROMPTS)]

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

for i, prompt in enumerate(prompts):
    print(f"\n{'='*90}\n[{i}] PROMPT:\n{prompt}\n")
    for name in MODELS:
        print(f"--- {name} ---\n{generate(models[name], prompt)}\n")
