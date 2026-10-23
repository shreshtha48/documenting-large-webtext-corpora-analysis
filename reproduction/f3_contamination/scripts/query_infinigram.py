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
many requests (in practice after ~50-65 requests at one per second). Requests go one at a time at --rate per second
(default 0.2), with an identifying User-Agent; a 403 / 429 pauses the run for --block_pause seconds, slows the pace
by 1.5x, and retries the same query instead of recording an error.

Group testing (default; --no_group turns it off): the count query accepts "A OR B OR ...", and pure OR counts are
exact (the API only subsamples AND queries). Most benchmark texts are absent, so several texts are tested in one
request; a group with count 0 resolves all its texts at once, and a group with a hit is split in halves until single
texts remain. The result per text is identical to querying it alone. The group size comes from the paper's rate
for the field (1/sqrt(rate), at most 8; no grouping above 30%). A text's case variants go into the same query.
Texts containing the words AND / OR in capitals cannot be sent as plain strings (the API reads them as operators)
and are recorded as errors.

Usage:
    python scripts/query_infinigram.py --out_dir results/C1_infinigram_2000 --sample 2000
    python scripts/query_infinigram.py --fields xsum cola --sample 0 ...     # 0 = every text
"""
import argparse, glob, gzip, json, os, random, re, sys, threading, time
import urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "data", "benchmarks")
sys.path.insert(0, HERE)
API = "https://api.infini-gram.io/"
INDEX = "v4_c4train_llama"
TRAIL = re.compile(r"[\s.!?,;:'\"”’)\]]+$")
UA = "c4-documentation-reproduction (Dodge et al. 2021 Table 2; research use)"
PACE = {"interval": 5.0, "next": 0.0, "block_pause": 600, "lock": threading.Lock(), "requests": 0}
OPERATOR = re.compile(r"\b(?:AND|OR)\b")


def wait_turn():
    with PACE["lock"]:
        now = time.time()
        if now < PACE["next"]:
            time.sleep(PACE["next"] - now)
        PACE["next"] = max(now, PACE["next"]) + PACE["interval"]
        PACE["requests"] += 1


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
            if "count" not in d:                        # e.g. a gateway error page: retry
                raise ValueError(f"no count in response: {str(d)[:120]}")
            return int(d["count"]), None
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and blocks > 0:     # rate-limited: pause, then retry the same query
                blocks -= 1
                PACE["interval"] = min(30.0, PACE["interval"] * 1.5)
                print(f"  HTTP {e.code}: pausing {PACE['block_pause']} s, then one request per "
                      f"{PACE['interval']:.1f} s", flush=True)
                time.sleep(PACE["block_pause"])
                continue
            if e.code not in (500, 502, 503, 504):
                return None, f"HTTP {e.code}"
        except Exception as e:                          # network, SSL, truncated or odd responses: retry
            print(f"  retry after {type(e).__name__}: {str(e)[:100]}", flush=True)
        attempt += 1
        time.sleep(min(60, 2 ** attempt))
    return None, "gave up after retries"


def result(row, counts, err, via="single"):
    return {"id": row["id"], "counts": counts, "matched": bool(any(counts)) if err is None else None,
            "error": err, "text": row["text"], "via": via}


def unqueryable(row):
    v = variants(row["text"])
    if not v:
        return "empty text"
    if any(OPERATOR.search(x) for x in v):
        return "contains AND/OR in capitals (read as query operators)"
    return None


def group_test(rows):
    """Results for `rows`, testing them together first (see the module docstring)."""
    c, e = count(" OR ".join(v for r in rows for v in variants(r["text"])))
    if e is None and c == 0:
        return [result(r, [0], None, "group" if len(rows) > 1 else "single") for r in rows]
    if len(rows) == 1:
        return [result(rows[0], [c] if e is None else [], e)]
    half = len(rows) // 2
    return group_test(rows[:half]) + group_test(rows[half:])


def group_size(field):
    from benchmarks import PAPER
    rate = PAPER[field][2] / PAPER[field][3] if field in PAPER else 0.1
    return 1 if rate >= 0.3 else max(1, min(8, round(rate ** -0.5)))


def score(row):
    bad = unqueryable(row)
    if bad:
        return result(row, [], bad)
    counts, err = [], None
    for q in variants(row["text"]):
        c, e = count(q)
        if e:
            err = e
            break
        counts.append(c)
        if c:
            break                                   # found: no need to try the other variant
    return result(row, counts, err)


def load(field):
    with gzip.open(os.path.join(BENCH, f"{field}.jsonl.gz"), "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fields", nargs="*", help="default: every built field")
    ap.add_argument("--sample", type=int, default=2000, help="texts per field (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=0.2, help="requests per second (one at a time)")
    ap.add_argument("--no_group", action="store_true", help="query every text on its own")
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
        g = 1 if args.no_group else group_size(field)
        t0, r0, n = time.time(), PACE["requests"], 0
        print(f"{field}: {len(todo)} texts to query, group size {g}", flush=True)
        with open(path, "a", encoding="utf-8") as out:
            for b in range(0, len(todo), g):
                chunk = todo[b:b + g]
                res = [score(r) for r in chunk if unqueryable(r)]
                ok = [r for r in chunk if not unqueryable(r)]
                if ok:
                    res += group_test(ok) if g > 1 else [score(r) for r in ok]
                for x in res:
                    if x["error"] and x["error"].startswith("HTTP 403"):
                        raise SystemExit("still blocked after repeated pauses; re-run later (finished texts are kept)")
                    out.write(json.dumps(x, ensure_ascii=False) + "\n")
                out.flush()                             # every finished group is kept if the run stops
                n_prev, n = n, n + len(chunk)
                if n // 50 > n_prev // 50:
                    print(f"  {field}: {n}/{len(todo)} texts, {PACE['requests'] - r0} requests "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        res = [json.loads(l) for l in open(path, encoding="utf-8")]
        ok = [r for r in res if r["error"] is None]
        print(f"{field}: {sum(r['matched'] for r in ok)}/{len(ok)} matched, {len(res) - len(ok)} errors, "
              f"{PACE['requests'] - r0} requests, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
