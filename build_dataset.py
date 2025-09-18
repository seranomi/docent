import os, json, time, random, re, argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm

MET_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

# -----------------------------
# Met API helpers
# -----------------------------
def met_search(query: str, has_images: bool = True) -> List[int]:
    params = {"q": query, "hasImages": str(has_images).lower()}
    r = requests.get(f"{MET_BASE}/search", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("objectIDs", []) or []

def met_get_object(object_id: int) -> Optional[Dict[str, Any]]:
    r = requests.get(f"{MET_BASE}/objects/{object_id}", timeout=30)
    if r.status_code != 200:
        return None
    return r.json()

def clean(s: Optional[str]) -> str:
    if not s: return ""
    return re.sub(r"\s+", " ", s).strip()

def build_meta(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 필수
    title = clean(obj.get("title"))
    artist = clean(obj.get("artistDisplayName"))
    objectDate = clean(obj.get("objectDate"))
    medium = clean(obj.get("medium"))
    if not title or not objectDate or not medium:
        return None

    # measurements(Overall) 우선, 없으면 dimensions 문자열 사용
    height = width = None
    for m in obj.get("measurements", []):
        if m.get("elementName") == "Overall":
            em = m.get("elementMeasurements") or {}
            height, width = em.get("Height"), em.get("Width")
            break

    meta = {
        "id": obj.get("objectID"),
        "title": title,
        "artist": artist,
        "artistDisplayBio": clean(obj.get("artistDisplayBio")),
        "objectDate": objectDate,
        "medium": medium,
        "dimensions": clean(obj.get("dimensions")),
        "height": height,
        "width": width,
        "classification": clean(obj.get("classification")),
        "culture": clean(obj.get("culture")),
        "tags": [{"term": t.get("term")} for t in (obj.get("tags") or [])],
        "objectURL": obj.get("objectURL"),
        "primaryImage": obj.get("primaryImage") or obj.get("primaryImageSmall"),
        "isPublicDomain": bool(obj.get("isPublicDomain")),
        "repository": clean(obj.get("repository")),
    }
    return meta

# -----------------------------
# (옵션) OpenAI로 큐레이터 초안 생성
# -----------------------------
def gen_curator_note_with_openai(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    api_key = "test"  # 실제 키로 교체 예정
    if not api_key:
        return None

    system = (
        "너는 미술관 도슨트다. 반드시 아래 JSON 스키마를 채워라.\n"
        '{ "intro": string, "facts": string[], "interpretation": string, '
        '"tips": string[], "followup": string, "sources": string[] }'
    )
    user = (
        "다음 작품 메타데이터를 바탕으로 과장 없이 한국어로 작성. JSON만 출력.\n"
        + json.dumps(meta, ensure_ascii=False, indent=2)
    )
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role":"system","content":system},{"role":"user","content":user}],
                "temperature": 0.4,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # JSON만 나오도록 강제했지만, 방어적 파싱
        return json.loads(content)
    except Exception as e:
        print(f"[warn] OpenAI 실패: {e}")
        return None

def fallback_curator_note(meta: Dict[str, Any]) -> Dict[str, Any]:
    facts = []
    if meta.get("artist"): facts.append(f"작가: {meta['artist']}")
    if meta.get("objectDate"): facts.append(f"연도: {meta['objectDate']}")
    if meta.get("medium"): facts.append(f"재료: {meta['medium']}")
    if meta.get("dimensions"): facts.append(f"크기: {meta['dimensions']}")

    return {
        "intro": f"이 작품은 {meta.get('artist') or '작자 미상'}의 작품으로, {meta.get('title')}를 보여줍니다.",
        "facts": facts,
        "interpretation": "색채와 구도의 균형이 주는 분위기를 중심으로 감상해 보세요.",
        "tips": ["인물/정물의 윤곽과 반사광 관찰", "배경 톤과 주요 대상의 대비 보기"],
        "followup": "색채와 구도 중 어느 부분을 더 들어볼까요?",
        "sources": [f"Met API (objectID {meta['id']})"]
    }

# -----------------------------
# 샘플(학습 항목) 생성
# -----------------------------
def make_samples_for_artwork(meta: Dict[str, Any], completion: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    한 작품에서 prompt를 2~3개 변형해 소량 증강
    """
    prompts = [
        "이 작품을 설명해줘",
        "색채와 분위기 중심으로 알려줘",
        "구도와 감상 포인트를 알려줘"
    ]
    samples = []
    for p in prompts[:2]:  # 기본 2개만 사용(원하면 3개)
        samples.append({"prompt": p, "completion": completion, "meta": meta})
    return samples

# -----------------------------
# 분할 저장
# -----------------------------
def train_val_test_split(items: List[Dict[str, Any]], ratios=(0.8, 0.1, 0.1), seed=42):
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    train = items[:n_train]
    val   = items[n_train:n_train+n_val]
    test  = items[n_train+n_val:]
    return train, val, test

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            # 모델 학습용으로는 prompt/completion만 필요하지만,
            # 초기 검수 편의를 위해 meta도 같이 저장(원하면 제거 가능)
            out = {"prompt": r["prompt"], "completion": r["completion"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

# -----------------------------
# 메인 파이프라인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default="painting", help="Met API 검색어")
    ap.add_argument("--limit", type=int, default=200, help="가져올 작품 수")
    ap.add_argument("--outdir", type=str, default="data", help="결과 저장 폴더")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--public-domain-only", action="store_true", help="퍼블릭도메인 작품만 사용")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"[info] query='{args.query}', limit={args.limit}, outdir={outdir}")

    # 1) ID 검색
    print("[1/4] Met IDs 검색…")
    ids = met_search(args.query, has_images=True)
    if not ids:
        print("검색 결과가 없습니다."); return
    random.Random(args.seed).shuffle(ids)

    # 2) 오브젝트 수집/정제
    print(f"[2/4] 오브젝트 수집(최대 {args.limit})…")
    metas: List[Dict[str, Any]] = []
    bar = tqdm(ids, total=min(len(ids), args.limit*3))  # 넉넉히 돌아보다가 한도 충족 시 중단
    for oid in bar:
        try:
            obj = met_get_object(oid)
            if not obj: 
                continue
            if args.public_domain_only and not obj.get("isPublicDomain"):
                continue
            meta = build_meta(obj)
            if not meta:
                continue
            metas.append(meta)
            if len(metas) >= args.limit:
                break
            time.sleep(0.05)  # 예의상 소폭 슬립
        except Exception:
            continue

    if not metas:
        print("사용 가능한 메타가 없습니다."); return

    # 3) 큐레이터 completion 생성 (OpenAI 우선, 실패시 fallback)
    print("[3/4] 큐레이터 completion 생성…")
    all_samples: List[Dict[str, Any]] = []
    for m in tqdm(metas):
        comp = gen_curator_note_with_openai(m) or fallback_curator_note(m)
        # 사실 필드 품질 보강: facts 안에 artist/year/medium 누락 시 채우기
        facts_txt = " ".join(comp.get("facts", []))
        need = []
        if m.get("artist") and "작가:" not in facts_txt: need.append(f"작가: {m['artist']}")
        if m.get("objectDate") and "연도:" not in facts_txt: need.append(f"연도: {m['objectDate']}")
        if m.get("medium") and "재료:" not in facts_txt: need.append(f"재료: {m['medium']}")
        if need: comp["facts"] = (comp.get("facts") or []) + need

        samples = make_samples_for_artwork(m, comp)
        all_samples.extend(samples)

    # 4) 분할/저장
    print("[4/4] 분할 및 저장…")
    train, val, test = train_val_test_split(all_samples, ratios=(0.8,0.1,0.1), seed=args.seed)
    write_jsonl(outdir / "sft.train.jsonl", train)
    write_jsonl(outdir / "sft.val.jsonl", val)
    write_jsonl(outdir / "sft.test.jsonl", test)

    print(f"완료 ✅  전체 샘플: {len(all_samples)}")
    print(f" - train: {len(train)}  -> {outdir/'sft.train.jsonl'}")
    print(f" - val:   {len(val)}    -> {outdir/'sft.val.jsonl'}")
    print(f" - test:  {len(test)}   -> {outdir/'sft.test.jsonl'}")

if __name__ == "__main__":
    main()
