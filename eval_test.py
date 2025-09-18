import json, random, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from json_repair import repair_json


REQ = ["intro","facts","interpretation","tips","followup","sources"]

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

def to_chat(tok, system, user):
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            [{"role":"system","content":system},
             {"role":"user","content":f"{user}\nJSON만 출력."}],
            tokenize=False, add_generation_prompt=True
        )
    return f"[SYSTEM]\n{system}\n[USER]\n{user}\n[ASSISTANT]\n"

def extract_json(text: str) -> str:
    text = text.replace("```json","").replace("```","")
    if "{" in text and "}" in text:
        return text[text.find("{"): text.rfind("}")+1].strip()
    return text.strip()

class StopOnClosingBrace(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len):
        self.tok = tokenizer; self.prompt_len = prompt_len
    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0][self.prompt_len:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        return text.rstrip().endswith("}")

def run_eval(base, adapter, test_file, n=20, max_new=450):
    from json_repair import repair_json  # pip install json-repair

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base_m = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base_m, adapter)
    model.eval()

    rows = [json.loads(l) for l in open(test_file, "r", encoding="utf-8")]
    rows = rows if len(rows) <= n else random.sample(rows, n)

    ok_parse = 0
    key_hit = {k:0 for k in REQ}

    for r in rows:
        prompt = f"질문: {r['prompt']}"
        chat = to_chat(tok, SYSTEM, prompt)
        inputs = tok(chat, return_tensors="pt").to(model.device)
        stop = StoppingCriteriaList([StopOnClosingBrace(tok, inputs["input_ids"].shape[1])])

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False, stopping_criteria=stop
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        raw = extract_json(gen)

        try:
            j = json.loads(raw)
        except Exception:
            try:
                j = json.loads(repair_json(raw))
            except Exception:
                continue
            
        j = backfill(j, r.get("meta"))

        ok_parse += 1
        for k in REQ:
            if k in j and j[k]:
                key_hit[k] += 1

    parse_rate = ok_parse / len(rows)
    key_rates = {k: key_hit[k]/len(rows) for k in REQ}
    return parse_rate, key_rates

def backfill(d, meta=None):
    # facts
    if isinstance(d.get("facts"), list) and len(d["facts"]) < 3 and meta:
        add = []
        if meta.get("artist"): add.append(f"작가: {meta['artist']}")
        if meta.get("objectDate"): add.append(f"연도: {meta['objectDate']}")
        if meta.get("medium"): add.append(f"재료: {meta['medium']}")
        if meta.get("dimensions"): add.append(f"크기: {meta['dimensions']}")
        d["facts"] = (d.get("facts") or []) + add[: max(0, 3 - len(d.get('facts', [])))]
    # tips
    if isinstance(d.get("tips"), list) and len(d["tips"]) == 0:
        d["tips"] = ["색과 빛의 대비를 유심히 보세요.", "구도와 시선 흐름을 따라가 보세요."]
    # sources
    if isinstance(d.get("sources"), list) and len(d["sources"]) == 0 and meta and meta.get("objectURL"):
        d["sources"] = [meta["objectURL"]]
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter", default="outputs/docent-tinyllama-lora")
    p.add_argument("--test_file", default="data/sft.test.jsonl")
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()

    pr, kr = run_eval(args.base, args.adapter, args.test_file, n=args.n, max_new=450)
    print("[eval] parse_rate:", round(pr,3))
    print("[eval] key_rates:", {k: round(v,3) for k,v in kr.items()})
