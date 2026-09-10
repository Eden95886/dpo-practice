"""
gpt2 helpful-only DPO vs Qwen2.5-1.5B-Instruct base vs Qwen2.5-1.5B-Instruct + LoRA DPO
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/private/tmp/claude-501/-Users-kjw/8a9d94ac-4e2d-4a10-9950-600f5bd81ae0/scratchpad"
GPT2_HELPFUL_DIR = f"{BASE}/dpo-gpt2-output-helpful"
QWEN_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_LORA_DIR = f"{BASE}/dpo-qwen15b-lora-helpful"
INDICES = [0, 50, 150, 250, 350, 450, 550, 650]

device = "mps" if torch.backends.mps.is_available() else "cpu"

gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
if gpt2_tok.pad_token is None:
    gpt2_tok.pad_token = gpt2_tok.eos_token
gpt2_helpful = AutoModelForCausalLM.from_pretrained(GPT2_HELPFUL_DIR).to(device).eval()

qwen_tok = AutoTokenizer.from_pretrained(QWEN_BASE)
# PeftModel.from_pretrained mutates the passed-in module in place (replaces Linear
# layers with LoRA-wrapped ones), so the "base" and "+LoRA" comparisons need two
# separate model instances loaded from disk - reusing one object would silently
# make the "base" generation go through the adapter-injected layers too.
qwen_base = AutoModelForCausalLM.from_pretrained(QWEN_BASE, dtype=torch.float32).to(device).eval()
qwen_lora_base = AutoModelForCausalLM.from_pretrained(QWEN_BASE, dtype=torch.float32).to(device).eval()
qwen_lora = PeftModel.from_pretrained(qwen_lora_base, QWEN_LORA_DIR).to(device).eval()

raw = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")["test"]

def get_prompt(example):
    idx = example["chosen"].rfind("\n\nAssistant:")
    return example["chosen"][: idx + len("\n\nAssistant:")]

def generate(model, tok, prompt, max_new_tokens=80):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=192).to(device)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()

for idx in INDICES:
    prompt = get_prompt(raw[idx])
    print(f"\n{'='*90}\n[test idx {idx}] PROMPT:\n{prompt}\n")
    print(f"--- gpt2 DPO helpful-only ---\n{generate(gpt2_helpful, gpt2_tok, prompt)}\n")
    print(f"--- Qwen2.5-1.5B-Instruct (base, no DPO) ---\n{generate(qwen_base, qwen_tok, prompt)}\n")
    print(f"--- Qwen2.5-1.5B + LoRA DPO (helpful-only) ---\n{generate(qwen_lora, qwen_tok, prompt)}\n")
