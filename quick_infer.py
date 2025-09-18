import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 또는 Qwen/Qwen2-1.5B-Instruct
ADAPTER = "outputs/docent-tinyllama-lora"    # 학습 결과 경로

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)

system = (
  "너는 미술관 도슨트다. 반드시 JSON으로 답하라.\n"
  '{ "intro": string, "facts": string[], "interpretation": string, '
  '"tips": string[], "followup": string, "sources": string[] }'
)

prompt = "색채와 분위기 중심으로 알려줘"  # 테스트 질문

if hasattr(tok, "apply_chat_template"):
    text = tok.apply_chat_template(
        [{"role":"system","content":system},
         {"role":"user","content":f"질문: {prompt}\nJSON만 출력."}],
        tokenize=False, add_generation_prompt=True
    )
else:
    text = f"[SYSTEM]\n{system}\n[USER]\n질문: {prompt}\n[ASSISTANT]\n"

inputs = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False, temperature=0.0)
gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(gen)
