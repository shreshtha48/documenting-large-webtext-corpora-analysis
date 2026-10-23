"""Arm B, step 1: extract C4.EN paragraphs that mention a target ethnicity (Dodge et al. 2021, A.7).

A.7: "we first found all paragraphs containing these words, where the word was surrounded by spaces (for easy
searching using fgrep ...)". So a paragraph (a newline-separated line of a C4 document) matches "Jewish" if it
contains " Jewish ", case-sensitive, as fgrep does by default. The same matching was used in pilots/shard_check.py.

Shards come from huggingface.co/datasets/allenai/c4 (en/c4-train.XXXXX-of-01024.json.gz, ~319 MB each, 1/1024
of C4.EN). By default 12 shard indices are drawn with a fixed seed. Each shard is downloaded, scanned, and
deleted unless --keep_shards is given. Finished shards are skipped, so the scan resumes.

The paper's corpus study covers only "Jewish" and "Arab" (A.7: "the ethnicities with the most extreme biases that
were easy to find in C4.EN with simple scripts"); that is the default. --table7_groups is our extension: all 15
UnQover groups, each matched by its spelling variants (e.g. "Middle-Eastern" or "Middle Eastern"). Its known
limits, which the paper names: "African" also matches "African American", and "Black" / "White" have very common
non-ethnic senses ("Black Friday", "White House"); "Asian", "European", "Native American" etc. are similar.

Writes <out_dir>/paragraphs_<shard>.jsonl.gz (one line per matching paragraph: shard, url, targets, text) and
<out_dir>/stats_<shard>.json (documents, paragraphs, matches and words per target, and the targets scanned).
A finished shard is skipped only if it was scanned with the same targets; otherwise the script stops, so one
folder never mixes target sets.

Free tier: downloads dominate (~319 MB per shard), so --workers 3-4 scans shards in parallel even on 2-4 vCPUs.
--range START END picks a contiguous block of shards, so a large scan can be split across sessions into the
same --out_dir (e.g. on Google Drive); each session resumes where the last one stopped.

Usage:
    python scripts/scan_c4_paragraphs.py --out_dir results/B1_c4_paragraphs              # 12 seeded shards
    python scripts/scan_c4_paragraphs.py --shards 0 1 2 --out_dir ...                    # explicit shards
    python scripts/scan_c4_paragraphs.py --files /path/c4-train.00000-of-01024.json.gz   # local files
    python scripts/scan_c4_paragraphs.py --table7_groups --out_dir ...                   # extension: all 15 groups
    python scripts/scan_c4_paragraphs.py --range 0 256 --workers 4 --out_dir ...         # a quarter of C4.EN
"""
import argparse, gzip, json, os, random, re, time
from concurrent.futures import ProcessPoolExecutor, as_completed

TARGETS_DEFAULT = ["Jewish", "Arab"]
# extension: the 15 UnQover / Table 7 groups -> the strings matched (with a space on each side, case-sensitive)
TABLE7_GROUPS = {
    "Jewish": ["Jewish"], "Arab": ["Arab"], "Asian": ["Asian"], "Caucasian": ["Caucasian"],
    "European": ["European"], "White": ["White"], "Alaskan": ["Alaskan"], "Hispanic": ["Hispanic"],
    "Native American": ["Native American"], "South-American": ["South-American", "South American"],
    "African-American": ["African-American", "African American"], "Latino": ["Latino"],
    "Middle-Eastern": ["Middle-Eastern", "Middle Eastern"], "Black": ["Black"], "African": ["African"],
}
N_SHARDS = 1024


def shard_name(i):
    return f"c4-train.{i:05d}-of-{N_SHARDS:05d}"


def as_targets(t):
    """Normalise stored targets: older shards.json files hold a list of words, newer ones {name: [strings]}."""
    return {x: [x] for x in t} if isinstance(t, list) else t


def scan_file(path, shard, targets, out_dir):
    """targets: {name: [strings]}; a paragraph matches a name if it contains any string with spaces around it."""
    pats = {t: [f" {v} " for v in variants] for t, variants in targets.items()}
    stats = {"shard": shard, "targets": targets, "documents": 0, "paragraphs": 0,
             "matched_paragraphs": {t: 0 for t in targets}, "words_in_matched_paragraphs": {t: 0 for t in targets}}
    tmp = os.path.join(out_dir, f"paragraphs_{shard}.jsonl.gz.tmp")
    with gzip.open(path, "rt", encoding="utf-8") as f, gzip.open(tmp, "wt", encoding="utf-8") as g:
        for line in f:
            d = json.loads(line)
            stats["documents"] += 1
            for p in d["text"].split("\n"):
                stats["paragraphs"] += 1
                hit = [t for t, ps in pats.items() if any(x in p for x in ps)]
                if hit:
                    n = len(p.split())
                    for t in hit:
                        stats["matched_paragraphs"][t] += 1
                        stats["words_in_matched_paragraphs"][t] += n
                    g.write(json.dumps({"shard": shard, "url": d["url"], "targets": hit, "text": p},
                                       ensure_ascii=False) + "\n")
    os.replace(tmp, os.path.join(out_dir, f"paragraphs_{shard}.jsonl.gz"))
    with open(os.path.join(out_dir, f"stats_{shard}.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shards", type=int, nargs="*", help="shard indices 0-1023 (default: --n seeded shards)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--files", nargs="*", help="local shard files instead of downloading")
    ap.add_argument("--targets", nargs="+", default=TARGETS_DEFAULT, help="words matched exactly (paper: Jewish Arab)")
    ap.add_argument("--table7_groups", action="store_true", help="extension: all 15 Table 7 groups with variants")
    ap.add_argument("--keep_shards", action="store_true")
    ap.add_argument("--cache_dir", default=None, help="where downloaded shards go (default: HF cache)")
    ap.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="shards START..END-1")
    ap.add_argument("--workers", type=int, default=1, help="shards scanned in parallel (downloads dominate)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    targets = TABLE7_GROUPS if args.table7_groups else {t: [t] for t in args.targets}

    if args.files:
        jobs = [(re.search(r"c4-train\.\d{5}-of-\d{5}", os.path.basename(p)).group(0), p) for p in args.files]
    else:
        if args.shards:
            idx = args.shards
        elif args.range:
            idx = list(range(max(0, args.range[0]), min(N_SHARDS, args.range[1])))
        else:
            idx = sorted(random.Random(args.seed).sample(range(N_SHARDS), args.n))
        jobs = [(shard_name(i), None) for i in idx]

    # one folder = one target set; shards.json accumulates the shards of every session
    meta_path = os.path.join(args.out_dir, "shards.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"shards": [], "targets": targets}
    if as_targets(meta["targets"]) != targets:
        raise SystemExit(f"{args.out_dir} was scanned with other targets {list(meta['targets'])}; use another --out_dir")
    meta["shards"] = sorted(set(meta["shards"]) | {s for s, _ in jobs})
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    todo = []
    for shard, path in jobs:
        sp = os.path.join(args.out_dir, f"stats_{shard}.json")
        if os.path.exists(sp):
            old = json.load(open(sp)).get("targets")      # absent in stats written before 2026-10-23
            if old is not None and as_targets(old) != targets:
                raise SystemExit(f"{sp} was scanned with other targets; use another --out_dir")
            continue
        todo.append((shard, path))
    print(f"{len(jobs)} shards requested, {len(jobs) - len(todo)} done already, {len(todo)} to scan", flush=True)

    t_all = time.time()
    if args.workers <= 1:
        for shard, path in todo:
            print(process(shard, path, targets, args), flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(process, shard, path, targets, args) for shard, path in todo]
            for i, fu in enumerate(as_completed(futs), 1):
                print(f"[{i}/{len(todo)}, {(time.time() - t_all) / 60:.0f} min] {fu.result()}", flush=True)


def process(shard, path, targets, args):
    t0 = time.time()
    downloaded = False
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("allenai/c4", f"en/{shard}.json.gz", repo_type="dataset", cache_dir=args.cache_dir)
        downloaded = True
    s = scan_file(path, shard, targets, args.out_dir)
    if downloaded and not args.keep_shards:
        os.remove(os.path.realpath(path))
    return f"{shard}: {s['documents']} docs, matches {s['matched_paragraphs']}, {time.time() - t0:.0f}s"


if __name__ == "__main__":
    main()
