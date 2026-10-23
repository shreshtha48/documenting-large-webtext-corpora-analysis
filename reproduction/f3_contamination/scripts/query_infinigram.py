"""Dodge et al. 2021 §4.2: is each benchmark text in C4.EN? Exact-match counts from the infini-gram API.

The infini-gram index "v4_c4train_llama" covers all of C4.EN train (allenai/c4 "en": 364,868,892 documents),
the corpus the paper measured. A text counts as contaminated if its count is at least 1.

Difference from the paper, which matched "exact matches, normalized for capitalization and punctuation":
infini-gram matches token sequences exactly, so case and punctuation must agree. Our rates are therefore LOWER
BOUNDS on the paper's. Two cheap steps narrow the gap:
- trailing punctuation and quotes are removed before querying (a text that ends "hopes." is then also found where
  C4 has "hopes!" or "hopes"; anything matching the full text still matches);
- if a text has no capital letter at all (a lowercased source), a sentence-case variant is also queried.
Texts whose query fails (e.g. too long for the API) are recorded with their error and left out of the denominator.

Each field is a random sample of --sample texts (seed 0; all texts if fewer), so every rate gets a Wilson CI.
Results append to <out_dir>/counts_<field>.jsonl, so an interrupted run resumes.

Politeness: the API is a free shared service behind an AWS gateway that answers HTTP 403 when one IP sends too
many requests (we hit it at ~6 requests/s). So requests go one at a time at --rate per second (default 1), with an
identifying User-Agent; a 403 / 429 pauses the run for --block_pause seconds and retries the same text, instead of
recording an error.

Usage:
    python scripts/query_infinigram.py --out_dir results/C1_infinigram_2000 --sample 2000
    python scripts/query_infinigram.py --fields xsum cola --sample 0 ...     # 0 = every text
"""
import argparse, glob, gzip, json, os, random, re, threading, time
import urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "data", "benchmarks")
API = "https://api.infini-gram.io/"
INDEX = "v4_c4train_llama"
TRAIL = re.compile(r"[\s.!?,;:'\"”’)\]]+$")
UA = "c4-documentation-reproduction (Dodge et al. 2021 Table 2; research use)"
PACE = {"interval": 1.0, "next": 0.0, "block_pause": 600, "lock": threading.Lock()}


def wait_turn():
    with PACE["lock"]:
        now = time.time()
        if now < PACE["next"]:
            time.sleep(PACE["next"] - now)
        PACE["next"] = max(now, PACE["next"]) + PACE["interval"]


def variants(text):
    v = TRAIL.sub("", text.strip())
    out = [v] if v else []
    if v and not re.search(r"[A-Z]", v):
        out.append(v[0].upper() + v[1:])
    return out


def count(query, retries=6, blocks=12):
    body = json.dumps({"index": INDEX, "query_type": "count", "query": query}).encode()
    attempt = 0
    while attempt < retries:
        wait_turn()
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            if "error" in d:
                return None, d["error"]
            return int(d["count"]), None
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and blocks > 0:     # rate-limited: pause, then retry the same query
                blocks -= 1
                print(f"  HTTP {e.code}: pausing {PACE['block_pause']} s", flush=True)
                time.sleep(PACE["block_pause"])
                continue
            if e.code not in (500, 502, 503, 504):
                return None, f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            pass
        attempt += 1
        time.sleep(min(60, 2 ** attempt))
    return None, "gave up after retries"


def score(row):
    counts, err = [], None
    for q in variants(row["text"]):
        c, e = count(q)
        if e:
            err = e
            break
        counts.append(c)
        if c:
            break                                   # found: no need to try the other variant
    return {"id": row["id"], "counts": counts, "matched": bool(any(counts)) if err is None else None,
            "error": err, "text": row["text"]}


def load(field):
    with gzip.open(os.path.join(BENCH, f"{field}.jsonl.gz"), "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fields", nargs="*", help="default: every built field")
    ap.add_argument("--sample", type=int, default=2000, help="texts per field (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=1.0, help="requests per second (one at a time)")
    ap.add_argument("--block_pause", type=int, default=600, help="seconds to pause after HTTP 403 / 429")
    args = ap.parse_args()
    PACE["interval"], PACE["block_pause"] = 1.0 / args.rate, args.block_pause
    os.makedirs(args.out_dir, exist_ok=True)
    fields = args.fields or sorted(os.path.basename(p)[:-9] for p in glob.glob(os.path.join(BENCH, "*.jsonl.gz")))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"index": INDEX, "api": API, "sample": args.sample, "seed": args.seed, "fields": fields}, f, indent=1)
    for field in fields:
        rows = load(field)
        if args.sample and len(rows) > args.sample:
            rows = random.Random(f"{args.seed}-{field}").sample(rows, args.sample)
        path = os.path.join(args.out_dir, f"counts_{field}.jsonl")
        done = set()
        if os.path.exists(path):
            done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")}
        todo = [r for r in rows if r["id"] not in done]
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as out:
            for i, res in enumerate(map(score, todo), 1):
                if res["error"] and res["error"].startswith("HTTP 403"):
                    raise SystemExit("still blocked after repeated pauses; re-run later (finished texts are kept)")
                out.write(json.dumps(res, ensure_ascii=False) + "\n")
                if i % 500 == 0:
                    out.flush()
                    print(f"  {field}: {i}/{len(todo)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        res = [json.loads(l) for l in open(path, encoding="utf-8")]
        ok = [r for r in res if r["error"] is None]
        print(f"{field}: {sum(r['matched'] for r in ok)}/{len(ok)} matched, {len(res) - len(ok)} errors, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
