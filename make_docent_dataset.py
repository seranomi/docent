import os, json, time, random, argparse
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm

try:
    from json_repair import repair_json
except Exception:
    repair_json = None

MET_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
DEFAULT_QUERIES = ["a","e","i","o","u","art","woman","man","flower","portrait","landscape","paint","sculpture"]

SYSTEM_PROMPT = (
    "너는 미술관 도슨트다. 아래 JSON 스키마 '그 자체만' 출력하라.\n"
    '{ "intro": string, "facts": string[], "interpretation": string, '
    '"tips": string[], "followup": string, "sources": string[] }\n'
    "규칙: 1) 설명문 금지 2) 코드블록/백틱 금지 3) JSON 외 문자 금지 "
    "4) 모르면 '정보 없음' 대신 일반 미술사·큐레이터 지식으로 보강해 채워라.\n"
    "추가 규칙: facts≥3, tips≥2, sources≥1(objectURL 우선)."
)

USER_TMPL = (
    "다음 '작품 메타데이터'와 네가 알고 있는 일반 미술사·큐레이터 지식을 근거로 "
    "위 스키마를 작성하라. 한국어로.\n"
    "작품 메타데이터:\n{meta}\n\n"
    "주의: JSON만 출력."
)

def openai_chat(model: str, system: str, user: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY 없음")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages":[{"role":"system","content":system},{"role":"user","content":user}],
               "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(5):
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        time.sleep(1.5*(attempt+1))
    r.raise_for_status()
    return ""

def met_search_ids(q: str, has_images: Optional[bool], public_domain: Optional[bool]) -> List[int]:
    params = {"q": q}
    if has_images is not None: params["hasImages"] = "true" if has_images else "false"
    if public_domain is not None: params["isPublicDomain"] = "true" if public_domain else "false"
    try:
        js = requests.get(f"{MET_BASE}/search", params=params, timeout=30).json()
        return js.get("objectIDs") or []
    except Exception:
        return []

def met_get_object(oid: int) -> Optional[Dict[str, Any]]:
    try:
        return requests.get(f"{MET_BASE}/objects/{oid}", timeout=30).json()
    except Exception:
        return None

def extract_meta(obj: Dict[str, Any]) -> Dict[str, Any]:
    tags = []
    if isinstance(obj.get("tags"), list):
        for t in obj["tags"]:
            term = t.get("term")
            if term: tags.append(term)
    dims = obj.get("dimensions")
    if not dims and isinstance(obj.get("measurements"), list):
        for m in obj["measurements"]:
            if m.get("elementName") in (None, "Overall"):
                em = m.get("elementMeasurements") or {}
                h, w = em.get("Height"), em.get("Width")
                if h and w: dims = f"Height {h} × Width {w}"
    return {
        "objectID": obj.get("objectID"),
        "title": obj.get("title"),
        "artistDisplayName": obj.get("artistDisplayName"),
        "artistBeginDate": obj.get("artistBeginDate"),
        "artistEndDate": obj.get("artistEndDate"),
        "artistDisplayBio": obj.get("artistDisplayBio"),
        "objectDate": obj.get("objectDate") or obj.get("objectBeginDate"),
        "medium": obj.get("medium"),
        "dimensions": dims,
        "classification": obj.get("classification"),
        "department": obj.get("department"),
        "repository": obj.get("repository"),
        "isPublicDomain": obj.get("isPublicDomain"),
        "primaryImage": obj.get("primaryImage"),
        "objectURL": obj.get("objectURL"),
        "tags": tags,
        "gallery": obj.get("GalleryNumber"),
    }

def try_extract_json(text: str) -> str:
    t = text.strip().replace("```json","```").strip()
    if "```" in t:
        t = t.split("```",1)[-1]
    l, r = t.find("{"), t.rfind("}")
    return t[l:r+1] if l>=0 and r>l else t

def parse_json_strict(text: str) -> Optional[Dict[str, Any]]:
    raw = try_extract_json(text)
    try:
        return json.loads(raw)
    except Exception:
        if repair_json:
            try: return json.loads(repair_json(raw))
            except Exception: return None
        return None

def backfill(d: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(d or {})
    for k in ["facts","tips","sources"]:
        if not isinstance(d.get(k), list): d[k] = [] if d.get(k) is None else [str(d[k])]
    if len(d["facts"]) < 3:
        cand = []
        if meta.get("artistDisplayName"): cand.append(f"작가: {meta['artistDisplayName']}")
        if meta.get("objectDate"): cand.append(f"연도: {meta['objectDate']}")
        if meta.get("medium"): cand.append(f"재료: {meta['medium']}")
        if meta.get("dimensions"): cand.append(f"크기: {meta['dimensions']}")
        if meta.get("classification"): cand.append(f"분류: {meta['classification']}")
        for c in cand:
            if len(d["facts"]) >= 3: break
            if c not in d["facts"]: d["facts"].append(c)
        while len(d["facts"]) < 3: d["facts"].append("정보 없음")
    if len(d["tips"]) < 2:
        base = ["색과 빛의 대비를 살펴보세요.", "구도와 시선의 흐름을 따라가 보세요."]
        for c in base:
            if len(d["tips"]) >= 2: break
            if c not in d["tips"]: d["tips"].append(c)
    if len(d["sources"]) == 0:
        d["sources"].append(meta.get("objectURL") or "정보 없음")
    d["intro"] = d.get("intro") or "작품의 기본 정보를 간략히 소개합니다."
    d["interpretation"] = d.get("interpretation") or "색채와 구도를 중심으로 작품의 분위기를 해석할 수 있습니다."
    d["followup"] = d.get("followup") or "구도와 색채 중 어떤 부분을 더 들어볼까요?"
    return d

def build_user(meta: Dict[str, Any]) -> str:
    compact = {
        "title": meta.get("title"),
        "artist": meta.get("artistDisplayName"),
        "date": meta.get("objectDate"),
        "medium": meta.get("medium"),
        "dimensions": meta.get("dimensions"),
        "classification": meta.get("classification"),
        "department": meta.get("department"),
        "repository": meta.get("repository"),
        "objectURL": meta.get("objectURL"),
        "tags": meta.get("tags"),
        "gallery": meta.get("gallery"),
        "hasImage": bool(meta.get("primaryImage")),
        "isPublicDomain": meta.get("isPublicDomain"),
    }
    return USER_TMPL.format(meta=json.dumps(compact, ensure_ascii=False, indent=2))

def generate_completion(meta: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    raw = openai_chat(model, SYSTEM_PROMPT, build_user(meta), temperature=0.2, max_tokens=700)
    j = parse_json_strict(raw)
    if not j:
        raw2 = openai_chat(model, SYSTEM_PROMPT, build_user(meta), temperature=0.0, max_tokens=600)
        j = parse_json_strict(raw2)
        if not j: return None
    return backfill(j, meta)

def pick_ids(target: int, with_images_first=True, public_domain_only=True) -> List[int]:
    ids, seen = [], set()
    queries = list(DEFAULT_QUERIES); random.shuffle(queries)
    # 1) 이미지+퍼블릭 우선
    for q in queries:
        if len(ids) >= target: break
        got = met_search_ids(q, has_images=True if with_images_first else None,
                             public_domain=True if public_domain_only else None)
        for oid in (got or []):
            if oid not in seen:
                ids.append(oid); seen.add(oid)
                if len(ids) >= target: break
    # 2) 부족하면 완화
    if len(ids) < target:
        for q in queries:
            if len(ids) >= target: break
            got = met_search_ids(q, has_images=None, public_domain=True if public_domain_only else None)
            for oid in (got or []):
                if oid not in seen:
                    ids.append(oid); seen.add(oid)
                    if len(ids) >= target: break
    return ids[:target]

def to_input_text(prompt: str, meta: Dict[str, Any]) -> str:
    meta_text = "\n".join([
        f"제목: {meta.get('title','')}",
        f"작가: {meta.get('artistDisplayName','')} ({meta.get('artistBeginDate','')}–{meta.get('artistEndDate','')})",
        f"연도: {meta.get('objectDate','')}",
        f"재료: {meta.get('medium','')}",
        f"크기: {meta.get('dimensions','')}",
        f"분류: {meta.get('classification','')}",
        f"URL: {meta.get('objectURL','')}"
    ])
    return f"{prompt}\n\n[메타데이터]\n{meta_text}"

def write_jsonl(path: str, items: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--out_prefix", type=str, default="data/sft")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--with_images_first", action="store_true")
    ap.add_argument("--public_domain_only", action="store_true")
    args = ap.parse_args()

    if os.path.exists(".env"):
        try:
            from dotenv import load_dotenv; load_dotenv()
        except Exception: pass

    random.seed(args.seed)

    print("[1/5] objectIDs 수집…")
    ids = pick_ids(args.count * 2, args.with_images_first, args.public_domain_only)
    ids = ids[:args.count] if len(ids) >= args.count else ids
    if not ids: raise SystemExit("수집 실패")

    print("[2/5] 메타 수집…")
    metas = []
    for oid in tqdm(ids):
        obj = met_get_object(oid)
        if not obj: continue
        meta = extract_meta(obj)
        if not meta.get("title") or not meta.get("objectURL"): continue
        metas.append(meta)

    if not metas: raise SystemExit("메타 없음")

    print("[3/5] 도슨트 JSON 생성…")
    samples = []
    for meta in tqdm(metas):
        comp = generate_completion(meta, args.model)
        if not comp: continue
        samples.append({"prompt":"이 작품을 설명해줘", "completion":comp, "meta":{
            "objectID": meta.get("objectID"),
            "title": meta.get("title"),
            "artistDisplayName": meta.get("artistDisplayName"),
            "artistBeginDate": meta.get("artistBeginDate"),
            "artistEndDate": meta.get("artistEndDate"),
            "objectDate": meta.get("objectDate"),
            "medium": meta.get("medium"),
            "dimensions": meta.get("dimensions"),
            "classification": meta.get("classification"),
            "objectURL": meta.get("objectURL"),
            "primaryImage": meta.get("primaryImage"),
            "tags": meta.get("tags"),
        }})

    if not samples: raise SystemExit("생성 실패")

    random.shuffle(samples)
    n = len(samples); n_train = int(n*0.8); n_val = int(n*0.1)
    train, val, test = samples[:n_train], samples[n_train:n_train+n_val], samples[n_train+n_val:]

    # 3종 저장: raw (prompt+completion+meta)
    out_train = f"{args.out_prefix}.train.jsonl"
    out_val   = f"{args.out_prefix}.val.jsonl"
    out_test  = f"{args.out_prefix}.test.jsonl"
    print("[4/5] raw 저장…")
    write_jsonl(out_train, train); write_jsonl(out_val, val); write_jsonl(out_test, test)

    # converted (input/output) 저장
    def convert_split(rows, out_path):
        converted = []
        for r in rows:
            input_text = to_input_text(r["prompt"], r["meta"])
            converted.append({"input": input_text, "output": r["completion"]})
        write_jsonl(out_path, converted)

    print("[5/5] converted(input/output) 저장…")
    convert_split(train, f"{args.out_prefix}.train.converted.jsonl")
    convert_split(val,   f"{args.out_prefix}.val.converted.jsonl")
    convert_split(test,  f"{args.out_prefix}.test.converted.jsonl")

    print(f"완료! 총 {n}개")
    print("raw:       ", out_train, out_val, out_test)
    print("converted: ", f"{args.out_prefix}.train.converted.jsonl",
                        f"{args.out_prefix}.val.converted.jsonl",
                        f"{args.out_prefix}.test.converted.jsonl")

if __name__ == "__main__":
    main()
