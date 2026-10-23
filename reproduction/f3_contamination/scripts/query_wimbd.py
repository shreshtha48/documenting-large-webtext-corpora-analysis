"""Dodge et al. 2021 §4.2 via WIMBD's Elasticsearch index of C4 (Elazar et al. 2023, "What's In My Big Data?").

WIMBD indexes C4 with a `subset` field, and its count_documents_for_each_phrase() restricts the `c4` index to
subset "en", i.e. C4.EN, the corpus the paper measured. A match is an Elasticsearch phrase match: the standard
analyzer lowercases and drops punctuation, which is the paper's "exact matches, normalized for capitalization and
punctuation". So, unlike query_infinigram.py, this is not a lower bound. Phrases are sent in batches, so even full
fields take minutes.

Access needs WIMBD keys (form: https://forms.gle/yMz7uTFhd1dKNYTk7). Put them in reproduction/f3_contamination/
es_config.yml (git-ignored), as WIMBD expects:
    cloud_id: "<from the WIMBD reply>"
    api_key: "<from the WIMBD reply>"

Output: <out_dir>/counts_<field>.jsonl in the same format as query_infinigram.py, so analyze_contamination.py works
unchanged. Resumable: texts already in the file are skipped.

Usage (pip install wimbd):
    python scripts/query_wimbd.py --out_dir results/W1_wimbd_full --sample 0          # every text of every field
    python scripts/query_wimbd.py --out_dir results/W1_wimbd_key6 --fields xsum tifu_short --sample 300
"""
import argparse, glob, json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from query_infinigram import BENCH, load  # noqa: E402

CONFIG = os.path.join(ROOT, "es_config.yml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fields", nargs="*", help="default: every built field")
    ap.add_argument("--sample", type=int, default=0, help="texts per field (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", default="c4")
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()
    if not os.path.exists(args.config):
        raise SystemExit(f"no WIMBD keys: create {args.config} (see this script's docstring)")
    from wimbd.es import count_documents_for_each_phrase, es_init
    es = es_init(args.config, timeout=180)

    os.makedirs(args.out_dir, exist_ok=True)
    fields = args.fields or sorted(os.path.basename(p)[:-9] for p in glob.glob(os.path.join(BENCH, "*.jsonl.gz")))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"source": "WIMBD Elasticsearch", "index": args.index, "subset": "en", "sample": args.sample,
                   "seed": args.seed, "fields": fields}, f, indent=1)
    for field in fields:
        rows = load(field)
        if args.sample and len(rows) > args.sample:
            rows = random.Random(f"{args.seed}-{field}").sample(rows, args.sample)   # same sample as infini-gram
        path = os.path.join(args.out_dir, f"counts_{field}.jsonl")
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
        todo = [r for r in rows if r["id"] not in done and r["text"].strip()]
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as out:
            for b in range(0, len(todo), args.batch_size):
                batch = todo[b:b + args.batch_size]
                counts = list(count_documents_for_each_phrase(args.index, [r["text"] for r in batch],
                                                              batch_size=args.batch_size, es=es))
                for r, c in zip(batch, counts):
                    out.write(json.dumps({"id": r["id"], "counts": [int(c)], "matched": c > 0, "error": None,
                                          "text": r["text"]}, ensure_ascii=False) + "\n")
                out.flush()
                print(f"  {field}: {min(b + args.batch_size, len(todo))}/{len(todo)}", flush=True)
        res = [json.loads(l) for l in open(path, encoding="utf-8")]
        print(f"{field}: {sum(r['matched'] for r in res)}/{len(res)} matched, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
