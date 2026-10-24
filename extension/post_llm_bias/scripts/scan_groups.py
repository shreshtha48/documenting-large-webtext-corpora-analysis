"""Collect the Arm B paragraphs (all 15 Table 7 groups) plus a random baseline, for 2019 C4 and post-LLM C4.

Group matching is Arm B's (reproduction/demographic_bias/scripts/scan_c4_paragraphs.py): a paragraph is a
newline-separated line of a C4.EN document, and it matches a group if it contains one of the group's spellings
with a space on each side, case-sensitive.

New here: target "Baseline", a deterministic 1-in-N sample of all paragraphs (md5 of url + paragraph index), with
the same N in every corpus. Its positivity is the general tone of that corpus, so a change in a group's positivity
between 2019 and 2024 can be read against the change in the web's tone overall.

Corpora:
  --c4_2019 N        the N seeded C4.EN shards from allenai/c4 (seed 0, 12 = Arm B's B1 shards), downloaded and
                     deleted one by one
  --c4_files GLOB    our post-LLM C4.EN outputs (c4_en_<crawl>_shard<i>.jsonl.gz) from run_wet_shards.py

Writes <out_dir>/paragraphs_<shard>.jsonl.gz (shard, url, targets, text: the format count_sentiment.py reads) and
stats_<shard>.json. Finished shards are skipped.

Usage:
    python scan_groups.py --c4_2019 12 --out_dir paragraphs/2019 --workers 3
    python scan_groups.py --c4_files "post_llm_c4/CC-MAIN-2024-10/c4_en_*.jsonl.gz" --out_dir paragraphs/CC-MAIN-2024-10
"""
import argparse, glob, gzip, hashlib, json, os, random, re, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "..", "..", "reproduction", "demographic_bias", "scripts")):
    sys.path.insert(0, os.path.abspath(p))
from scan_c4_paragraphs import TABLE7_GROUPS, N_SHARDS, shard_name

BASELINE = "Baseline"
BASELINE_EVERY = 200


def is_baseline(url, i, every):
    return int(hashlib.md5(f"{url}\t{i}".encode("utf-8")).hexdigest()[:8], 16) % every == 0


def scan(path, shard, out_dir, every=BASELINE_EVERY):
    pats = {t: [f" {v} " for v in vs] for t, vs in TABLE7_GROUPS.items()}
    stats = {"shard": shard, "source": os.path.basename(path), "baseline_every": every, "documents": 0,
             "paragraphs": 0, "matched_paragraphs": {t: 0 for t in list(pats) + [BASELINE]}}
    tmp = os.path.join(out_dir, f"paragraphs_{shard}.jsonl.gz.tmp")
    with gzip.open(path, "rt", encoding="utf-8") as f, gzip.open(tmp, "wt", encoding="utf-8") as g:
        for line in f:
            d = json.loads(line)
            stats["documents"] += 1
            for i, p in enumerate(d["text"].split("\n")):
                stats["paragraphs"] += 1
                hit = [t for t, ps in pats.items() if any(x in p for x in ps)]
                if is_baseline(d["url"], i, every):
                    hit.append(BASELINE)
                if hit:
                    for t in hit:
                        stats["matched_paragraphs"][t] += 1
                    g.write(json.dumps({"shard": shard, "url": d["url"], "targets": hit, "text": p},
                                       ensure_ascii=False) + "\n")
    os.replace(tmp, os.path.join(out_dir, f"paragraphs_{shard}.jsonl.gz"))
    with open(os.path.join(out_dir, f"stats_{shard}.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def process(shard, path, out_dir, cache_dir):
    t0 = time.time()
    downloaded = path is None
    if downloaded:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("allenai/c4", f"en/{shard}.json.gz", repo_type="dataset", cache_dir=cache_dir)
    s = scan(path, shard, out_dir)
    if downloaded:
        os.remove(os.path.realpath(path))
    m = s["matched_paragraphs"]
    return f"{shard}: {s['documents']} docs, Jewish {m['Jewish']}, Arab {m['Arab']}, baseline {m[BASELINE]} ({time.time() - t0:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--c4_2019", type=int, metavar="N", help="N seeded allenai/c4 en shards (12 = Arm B's B1)")
    src.add_argument("--c4_files", help="glob of post-LLM c4_en_*.jsonl.gz files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.c4_2019:
        jobs = [(shard_name(i), None) for i in sorted(random.Random(args.seed).sample(range(N_SHARDS), args.c4_2019))]
    else:
        files = sorted(glob.glob(args.c4_files))
        if not files:
            raise SystemExit(f"no files match {args.c4_files}")
        jobs = [(re.sub(r"^c4_en_|\.jsonl\.gz$", "", os.path.basename(p)), p) for p in files]
    todo = [(s, p) for s, p in jobs if not os.path.exists(os.path.join(args.out_dir, f"stats_{s}.json"))]
    print(f"{len(jobs)} shards, {len(jobs) - len(todo)} done already, {len(todo)} to scan", flush=True)

    t_all = time.time()
    with ProcessPoolExecutor(max(1, args.workers)) as ex:
        futs = [ex.submit(process, s, p, args.out_dir, args.cache_dir) for s, p in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            print(f"[{k}/{len(todo)}, {(time.time() - t_all) / 60:.0f} min] {fu.result()}", flush=True)


if __name__ == "__main__":
    main()
