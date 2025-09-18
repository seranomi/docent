import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"       # 학습 때 쓴 베이스
ADAPTER = "outputs/docent-tinyllama-lora"         # 학습 결과 경로

# ★ 학습/평가/추론에서 "동일하게" 사용할 SYSTEM
SYSTEM = (
  "너는 미술관 도슨트다. 반드시 아래 JSON 스키마 **그 자체만** 출력한다.\n"
  '{ "intro": string, "facts": string[], "interpretation": string, '
  '"tips": string[], "followup": string, "sources": string[] }\n'
  "규칙: 1) 설명문 금지 2) 코드블록/백틱 금지 3) JSON 외 문자 금지 4) 모르면 빈 값으로.\n"
  '예: {"intro":"","facts":[],"interpretation":"","tips":[],"followup":"","sources":[]}'
  '추가 규칙:\n - "facts"에는 최소 3개 항목 작성 (작가/연도/재료/크기/분류 등 메타 기반)\n'
  '- "tips"에는 최소 2개 항목 작성 (감상 포인트)\n'
  '- "sources"에는 최소 1개(가능하면 Met objectURL)'
)

QUESTION = "색채와 분위기 중심으로 알려줘"

def to_chat(tok, system, user):
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            [{"role":"system","content":system},
             {"role":"user","content":f"{user}\nJSON만 출력."}],
            tokenize=False, add_generation_prompt=True
        )
    # fallback 포맷
    return f"[SYSTEM]\n{system}\n[USER]\n{user}\n[ASSISTANT]\n"

def extract_json(text: str) -> str:
    # 코드블럭/백틱 제거
    text = text.replace("```json", "").replace("```", "")
    # 첫 { ~ 마지막 } 범위만 취함
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}")+1]
    return text.strip()

class StopOnClosingBrace(StoppingCriteria):
    """생성 텍스트가 '}'로 끝나면 멈춤 (불필요한 꼬리 방지)"""
    def __init__(self, tokenizer, prompt_len):
        self.tok = tokenizer
        self.prompt_len = prompt_len
    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0][self.prompt_len:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        return text.rstrip().endswith("}")

def main():
    from json_repair import repair_json  # pip install json-repair

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    prompt = to_chat(tok, SYSTEM, f"질문: {QUESTION}")
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    stop = StoppingCriteriaList([StopOnClosingBrace(tok, inputs["input_ids"].shape[1])])

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=450,    # 충분히 크게
            do_sample=False,       # 결정적 디코딩(temperature/top_p 무시)
            stopping_criteria=stop
        )

    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    raw = extract_json(gen)

    print("\n=== RAW ===\n", raw)
    try:
        j = json.loads(raw)
    except Exception:
        try:
            j = json.loads(repair_json(raw))   # 자동 복구 시도
        except Exception as e:
            print("\n[WARN] JSON parse failed even after repair:", e)
            return

    print("\n=== Parsed keys ===", list(j.keys()))
    # 필요하면 여기서 대화체로 변환해 보기:
    # print(to_conversation(j))

if __name__ == "__main__":
    main()
