import os, json, time, random, argparse, threading
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from json_repair import repair_json
except Exception:
    repair_json = None

MET_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

# =========================
# 도슨트 프롬프트
# =========================
SYSTEM_PROMPT = (
    "너는 미술관 도슨트다. 아래 JSON 스키마 '그 자체만' 출력하라.\n"
    '{ "intro": string, "facts": string[], "interpretation": string, '
    '"tips": string[], "followup": string, "sources": string[] }\n'
    "규칙: 1) 설명문 금지 2) 코드블록/백틱 금지 3) JSON 외 문자 금지 "
    "4) 추가적인 부분은 일반 미술사·큐레이터 지식으로 보강해 채워라.\n"
    "추가 규칙: facts≥3, tips≥2, sources≥1(objectURL 우선)."
)

USER_TMPL = (
    "다음 '작품 메타데이터'와 네가 알고 있는 일반 미술사·큐레이터 지식을 근거로 "
    "위 스키마를 작성하라. 한국어로.\n"
    "작품 메타데이터:\n{meta}\n\n"
    "주의: JSON만 출력."
)

# =========================
# OpenAI 호출
# =========================
def openai_chat(model: str, system: str, user: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY 없음")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    base_payload = {
        "model": model,
        "messages":[{"role":"system","content":system},{"role":"user","content":user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    payloads = [
        {**base_payload, "response_format": {"type": "json_object"}},  # 1차: JSON 강제
        base_payload,  # 2차: 일반 모드
    ]

    last_err = None
    backoff = 1.5
    for attempt in range(8):
        try:
            p = payloads[0] if attempt == 0 else payloads[min(1, attempt)]
            r = requests.post(url, headers=headers, json=p, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last_err = f"status={r.status_code} body={r.text[:300]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff)
        backoff = min(backoff * 1.8, 15)
    raise RuntimeError(f"OpenAI 실패: {last_err}")

# =========================
# 글로벌 속도제어 (RPS + 백오프)
# =========================
class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: Optional[int] = None):
        self.rate = float(rate_per_sec)
        self.capacity = int(capacity or max(1, int(rate_per_sec * 2)))
        self.tokens = self.capacity
        self.lock = threading.Lock()
        self.timestamp = time.time()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                delta = now - self.timestamp
                refill = delta * self.rate
                if refill > 0:
                    self.tokens = min(self.capacity, self.tokens + refill)
                    self.timestamp = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.01)

GLOBAL_BUCKET = TokenBucket(rate_per_sec=3.0, capacity=8)  # 안전선: 초당 3요청
_global_backoff_lock = threading.Lock()
_global_backoff_until = 0.0

def global_backoff_sleep():
    t = _global_backoff_until
    if t > 0:
        now = time.time()
        if now < t:
            time.sleep(t - now)

def set_global_backoff(seconds: float):
    global _global_backoff_until
    with _global_backoff_lock:
        _global_backoff_until = max(_global_backoff_until, time.time() + seconds)

# =========================
# MET API & 메타 가공
# =========================
def _session_with_retries(total_retry=8, backoff=0.6, pool=100):
    s = requests.Session()
    retries = Retry(
        total=total_retry, connect=total_retry, read=total_retry,
        backoff_factor=backoff, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=pool, pool_maxsize=pool)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "docent-dataset-builder/1.0 (+https://metmuseum.org)"})
    return s

def met_get_object_retry_with_session(session: requests.Session, oid: int, tries=3, base_sleep=0.5):
    for k in range(tries):
        # 전역 백오프 존중 + 전역 RPS 제한
        global_backoff_sleep()
        GLOBAL_BUCKET.acquire()
        try:
            r = session.get(f"{MET_BASE}/objects/{oid}", timeout=30)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if (ra and ra.isdigit()) else (3.0 * (k+1))
                set_global_backoff(wait)
                time.sleep(wait)
                continue
            if 500 <= r.status_code < 600:
                set_global_backoff(2.0 * (k+1))
                time.sleep(0.5 * (k+1))
                continue
            if r.status_code != 200:
                time.sleep(base_sleep * (k+1))
                continue
            try:
                return r.json()
            except Exception:
                time.sleep(base_sleep * (k+1))
                continue
        except Exception:
            time.sleep(base_sleep * (k+1))
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

# =========================
# JSON 파싱/보정
# =========================
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

# =========================
# Backfill (누락값 채우기)
# =========================
def backfill(d: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(d or {})
    for k in ["facts","tips","sources"]:
        if not isinstance(d.get(k), list):
            d[k] = [] if d.get(k) is None else [str(d[k])]

    if len(d["facts"]) < 3:
        cand = []
        if meta.get("title"): cand.append(f"작품 제목: {meta['title']}")
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

# =========================
# 프롬프트 빌드 / 컴플리션
# =========================
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
        "isPublicDomain": bool(meta.get("isPublicDomain")),
    }
    return USER_TMPL.format(meta=json.dumps(compact, ensure_ascii=False, indent=2))

def generate_completion(meta: Dict[str, Any], model: str) -> Dict[str, Any]:
    try:
        raw = openai_chat(model, SYSTEM_PROMPT, build_user(meta), temperature=0.2, max_tokens=600)
        j = parse_json_strict(raw)
        if not j:
            raw2 = openai_chat(model, SYSTEM_PROMPT, build_user(meta), temperature=0.0, max_tokens=500)
            j = parse_json_strict(raw2)
        if not j:
            return backfill({}, meta)
        return backfill(j, meta)
    except Exception:
        return backfill({}, meta)

# =========================
# 전체 ID 캐시 → 랜덤 샘플 → (필요시) 필터
# =========================
def fetch_all_object_ids(cache_path: str = "met_all_ids.json") -> List[int]:
    if os.path.exists(cache_path):
        try:
            return json.load(open(cache_path, "r", encoding="utf-8"))
        except Exception:
            pass
    try:
        js = requests.get(f"{MET_BASE}/objects", timeout=60).json()
        ids = js.get("objectIDs") or []
    except Exception:
        ids = []
    if ids:
        try:
            json.dump(ids, open(cache_path, "w", encoding="utf-8"))
        except Exception:
            pass
    return ids

def fetch_meta_batch(ids: List[int],
                     max_workers: int = 8,
                     chunk: int = 1000,
                     verbose: bool = True) -> List[Dict[str, Any]]:
    """
    - 전역 RPS 제한 + 전역 백오프 적용
    - 큰 chunk로 배치 수 절감, 배치별 통계 출력(valid/no-title/errors/cum)
    """
    metas = []
    session = _session_with_retries()
    total_batches = (len(ids) + chunk - 1) // chunk

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for bi in range(0, len(ids), chunk):
            batch = ids[bi:bi+chunk]
            bidx = bi // chunk + 1

            batch_total = len(batch)
            batch_valid = 0
            batch_no_title = 0
            batch_errors = 0

            desc = f"메타 수집 {bidx}/{total_batches}"
            futs = {
                ex.submit(met_get_object_retry_with_session, session, oid): oid
                for oid in batch
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                try:
                    obj = fut.result()
                except Exception:
                    obj = None

                if not isinstance(obj, dict) or "objectID" not in obj:
                    batch_errors += 1
                    continue

                meta = extract_meta(obj)
                # 최소 요건: title 또는 objectURL 중 하나는 있어야 살림
                if not meta.get("title") and not meta.get("objectURL"):
                    batch_no_title += 1
                    continue

                metas.append(meta)
                batch_valid += 1

            if verbose:
                tqdm.write(
                    f"  - [{bidx}/{total_batches}] "
                    f"valid {batch_valid}/{batch_total} | "
                    f"no-title {batch_no_title} | "
                    f"errors {batch_errors} | "
                    f"cum {len(metas)}"
                )

            # 배치 간 쿨다운 + 전역 백오프 존중
            global_backoff_sleep()
            time.sleep(30.0)

    return metas

def pick_ids(target: int,
             with_images_first: bool=True,
             public_domain_only: bool=True,
             oversample_factor: float=1.5,
             cache_all_ids: str="met_all_ids.json",
             meta_workers: int=8,
             chunk: int=1000,
             verbose: bool=True) -> List[int]:
    all_ids = fetch_all_object_ids(cache_all_ids)
    if not all_ids:
        return []
    random.shuffle(all_ids)

    take = min(len(all_ids), int(target * oversample_factor))
    pool = all_ids[:take]

    # 조건이 없으면 바로 반환(최속)
    if not with_images_first and not public_domain_only:
        return pool[:target]

    def ok(meta: Dict[str, Any]) -> bool:
        if public_domain_only and not meta.get("isPublicDomain"):
            return False
        if with_images_first and not (meta.get("primaryImage") or ""):
            return False
        return True

    selected = []
    metas = fetch_meta_batch(pool, max_workers=meta_workers, chunk=chunk, verbose=verbose)
    for m in metas:
        if ok(m):
            selected.append(m["objectID"])
            if len(selected) >= target:
                if verbose: print(f"  - 대상 ID: {len(selected)}개")
                return selected[:target]

    i = take
    while len(selected) < target and i < len(all_ids):
        extra_take = min(len(all_ids) - i, int(max(target - len(selected), target * 0.5)))
        extra_pool = all_ids[i:i+extra_take]
        i += extra_take
        metas = fetch_meta_batch(extra_pool, max_workers=meta_workers, chunk=chunk, verbose=verbose)
        for m in metas:
            if ok(m):
                selected.append(m["objectID"])
                if len(selected) >= target:
                    break

    if verbose: print(f"  - 대상 ID: {len(selected)}개")
    return selected[:target]

# =========================
# 변환/저장 유틸
# =========================
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
    return f"이 작품을 설명해줘\n\n[메타데이터]\n{meta_text}"

def write_jsonl(path: str, items: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def safe_write_progress(path: str, items: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    
def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

# =========================
# 메인
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--out_prefix", type=str, default="data04/sft")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--with_images_first", action="store_true")
    ap.add_argument("--public_domain_only", action="store_true")
    ap.add_argument("--meta_workers", type=int, default=8, help="MET 메타 수집 병렬 워커 수")
    ap.add_argument("--chunk", type=int, default=1000, help="메타 수집 배치 크기")
    ap.add_argument("--save_every", type=int, default=50, help="샘플 N개마다 진행 상황 임시 저장")
    ap.add_argument("--oversample_factor", type=float, default=1.5, help="초기 랜덤 샘플 과샘플링 배수")
    ap.add_argument("--cache_all_ids", type=str, default="met_all_ids.json", help="전체 objectIDs 캐시 파일")
    ap.add_argument("--auto_yes", action="store_true", help="메타 개수 확인 프롬프트 건너뛰기")
    ap.add_argument("--verbose", action="store_true", help="배치별 유효 메타 요약 출력")
    args = ap.parse_args()

    if os.path.exists(".env"):
        try:
            from dotenv import load_dotenv; load_dotenv()
        except Exception:
            pass

    random.seed(args.seed)

    print("[1/6] objectIDs 수집(전체→캐시→랜덤 샘플)…")
    ids = pick_ids(
        target=args.count,
        with_images_first=args.with_images_first,
        public_domain_only=args.public_domain_only,
        oversample_factor=args.oversample_factor,
        cache_all_ids=args.cache_all_ids,
        meta_workers=args.meta_workers,
        chunk=args.chunk,
        verbose=args.verbose,
    )
    if not ids:
        raise SystemExit("수집 실패")
    print(f"  - 대상 ID: {len(ids)}개")

    print("[2/6] 메타 수집…")
    metas = fetch_meta_batch(ids, max_workers=args.meta_workers, chunk=args.chunk, verbose=args.verbose)
    if not metas:
        raise SystemExit("메타 없음")
    print(f"  - 유효 메타: {len(metas)}개")

    if not args.auto_yes:
        proceed = input(f"총 {len(metas)}개의 메타데이터를 수집했습니다. 계속 진행하시겠습니까? (y/n): ").strip().lower()
        if proceed not in ("y", "yes"):
            print("사용자에 의해 중단되었습니다.")
            return

    print("[3/6] 도슨트 JSON 생성…")
    samples = []
    for i, meta in enumerate(tqdm(metas, desc="OpenAI 생성")):
        comp = generate_completion(meta, args.model)
        samples.append({
            "prompt": "이 작품을 설명해줘",
            "completion": comp,
            "meta": {
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
            }
        })
        if (i+1) % args.save_every == 0:
            safe_write_progress(f"{args.out_prefix}.progress.jsonl", samples)

    if not samples:
        raise SystemExit("생성 실패")
    
    # samples = load_jsonl(f"{args.out_prefix}.progress.jsonl")

    random.shuffle(samples)
    n = len(samples); n_train = int(n*0.8); n_val = int(n*0.1)
    train, val, test = samples[:n_train], samples[n_train:n_train+n_val], samples[n_train+n_val:]

    out_train = f"{args.out_prefix}.train.jsonl"
    out_val   = f"{args.out_prefix}.val.jsonl"
    out_test  = f"{args.out_prefix}.test.jsonl"
    print("[4/6] raw 저장…")
    write_jsonl(out_train, train); write_jsonl(out_val, val); write_jsonl(out_test, test)

    def convert_split(rows, out_path):
        converted = []
        for r in rows:
            input_text = to_input_text(r["prompt"], r["meta"])
            converted.append({"input": input_text, "output": r["completion"]})
        write_jsonl(out_path, converted)

    print("[5/6] converted(input/output) 저장…")
    convert_split(train, f"{args.out_prefix}.train.converted.jsonl")
    convert_split(val,   f"{args.out_prefix}.val.converted.jsonl")
    convert_split(test,  f"{args.out_prefix}.test.converted.jsonl")

    print(f"[6/6] 완료! 총 {n}개")
    print("raw:       ", out_train, out_val, out_test)
    print("converted: ", f"{args.out_prefix}.train.converted.jsonl",
                        f"{args.out_prefix}.val.converted.jsonl",
                        f"{args.out_prefix}.test.converted.jsonl")

if __name__ == "__main__":
    main()
