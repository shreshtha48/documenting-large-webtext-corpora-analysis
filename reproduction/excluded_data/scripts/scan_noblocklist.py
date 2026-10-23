"""Dodge et al. 2021 §5: one pass over C4.EN.NOBLOCKLIST shards for every §5 analysis.

The excluded set: §5 compares C4.EN.NOBLOCKLIST with C4.EN. The documents the blocklist removed are the
noblocklist documents that match the TFDS v3.0.1 bad-words regex (the filter is the last step of C4's pipeline;
see research-empirical-methods/README.md §1; the one-shard pilot matched the released C4.EN for 99.4% of
documents). So each document is classified with that regex, exactly as TFDS does: on the lowercased text,
LDNOOBW list at commit 25e679f, joined without escaping.

Per document, the scan records:
- excluded or kept (the blocklist regex);
- which of the 22 identity regexes of Table 6 it matches (§5.2; case-insensitive, with word boundaries);
- its dialect: the argmax of the Blodgett et al. (2016) model (§5.3; see dialect.py), or "unassigned".

These are summed per web domain (URL host), so the analysis can bootstrap over domains, which are not
independent. Also saved per shard:
- a random sample of excluded documents for the clustering (§5.1, Fig. 6);
- excluded documents that mention "lesbian" or "gay", for the paper's manual check (§5.2);
- the agreement between the fast and the exact dialect inference on the first documents.

Outputs per shard, in --out_dir:
- counts_<shard>.npz: host list and a hosts x columns count matrix (columns in COLUMNS);
- excluded_sample_<shard>.jsonl.gz, inspect_<shard>.jsonl.gz;
- stats_<shard>.json, written last, which marks the shard as done.

Free tier: about 400 MB download and a few minutes of CPU per shard; --workers scans shards in parallel;
--range splits a large scan across sessions into the same --out_dir (e.g. on Google Drive).

Usage:
    python scripts/scan_noblocklist.py --n 12 --seed 0 --workers 2 --out_dir results/S1_noblocklist_12_shards
    python scripts/scan_noblocklist.py --files /path/c4-train.00000-of-01024.json.gz --out_dir ...
"""
import argparse, gzip, json, os, random, re, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.parse import urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BADWORDS = os.path.join(ROOT, "data", "ldnoobw_en_25e679f.txt")
MODEL_DIR = os.path.join(ROOT, "data", "twitteraae_model")
N_SHARDS = 1024

# Table 6, in the order printed; the label is the one Fig. 5 uses
IDENTITY = [
    ("homosexual", r"homosexuals?"), ("gay", r"gays?"), ("non-binary", r"non[ -]?binary"),
    ("trans", r"trans(?:|\+|gender)"), ("lesbian", r"lesbians?"), ("black", r"blacks?"),
    ("african american", r"african[ -]americans?"), ("latina / latino", r"latin[oax]s?"),
    ("asian", r"asian(?:[ -]american)?s?"), ("muslim", r"muslims?"), ("jewish", r"jew(?:|s|ish)"),
    ("women", r"wom[ae]n"), ("female", r"females?"), ("man", r"m[ae]n"), ("male", r"males?"),
    ("straight", r"straights?"), ("heterosexual", r"heterosexuals?"), ("bisexual", r"bi-?sexuals?"),
    ("white", r"whites?"), ("caucasian", r"caucasians?"), ("european", r"european(?:[ -]american)?s?"),
    ("christian", r"christians?"),
]
DIALECTS = ["AAE", "Hispanic", "Other", "WAE", "unassigned"]
# per host: docs, excluded | per identity term: docs, excluded | per dialect: docs, excluded
COLUMNS = (["docs", "excluded"] + [f"{t}|{k}" for t, _ in IDENTITY for k in ("docs", "excluded")]
           + [f"dialect:{d}|{k}" for d in DIALECTS for k in ("docs", "excluded")])
INSPECT = {"lesbian": re.compile(r"\blesbians?\b", re.I), "gay": re.compile(r"\bgays?\b", re.I)}


def badwords_regex():
    words = [w.strip() for w in open(BADWORDS, encoding="utf-8") if w.strip()]
    return re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))   # TFDS v3.0.1: not escaped


def identity_regex():
    """All 22 regexes in one pass; the named group that matched tells which term."""
    return re.compile("|".join(rf"\b(?P<g{i}>{p})\b" for i, (_, p) in enumerate(IDENTITY)), re.I)


def shard_name(i):
    return f"c4-train.{i:05d}-of-{N_SHARDS:05d}"


def scan_file(path, shard, args):
    from dialect import DialectModel
    bad, ident = badwords_regex(), identity_regex()
    model = DialectModel(MODEL_DIR)
    rng = random.Random(f"{args.seed}-{shard}")
    T = len(IDENTITY)
    col = {c: i for i, c in enumerate(COLUMNS)}
    per_host = defaultdict(lambda: np.zeros(len(COLUMNS), dtype=np.int64))
    excluded_pool, inspect = [], {k: [] for k in INSPECT}
    agree = n_val = 0
    n = n_ex = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text, url = d["text"], d["url"]
            n += 1
            ex = bad.search(text.lower()) is not None
            n_ex += ex
            row = per_host[urlparse(url).netloc]
            row[0] += 1
            row[1] += ex
            hit = {int(m.lastgroup[1:]) for m in ident.finditer(text)}
            for i in hit:
                row[2 + 2 * i] += 1
                row[3 + 2 * i] += ex
            toks = model.tokens(text)
            post = model.fast(toks)
            dl = DIALECTS[int(np.argmax(post))] if post is not None else "unassigned"
            if n_val < args.validate_dialect:
                ref = model.exact(toks)
                agree += (DIALECTS[int(np.argmax(ref))] if ref is not None else "unassigned") == dl
                n_val += 1
            row[col[f"dialect:{dl}|docs"]] += 1
            row[col[f"dialect:{dl}|excluded"]] += ex
            if ex:
                excluded_pool.append({"url": url, "text": text, "dialect": dl})
                for k, r in INSPECT.items():
                    if len(inspect[k]) < args.inspect_cap and r.search(text):
                        inspect[k].append({"term": k, "url": url, "text": text,
                                           "blocklist_words": sorted({m.strip() for m in bad.findall(text.lower())})})
    sample = rng.sample(excluded_pool, min(args.sample_excluded, len(excluded_pool)))
    for name, rows in (("excluded_sample", sample), ("inspect", [r for v in inspect.values() for r in v])):
        tmp = os.path.join(args.out_dir, f"{name}_{shard}.jsonl.gz.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as g:
            for r in rows:
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, tmp[:-4])
    hosts = list(per_host)
    np.savez_compressed(os.path.join(args.out_dir, f"counts_{shard}.npz"), hosts=np.array(hosts, dtype=str),
                        counts=np.stack([per_host[h] for h in hosts]), columns=np.array(COLUMNS))
    stats = {"shard": shard, "documents": n, "excluded": n_ex, "excluded_rate": n_ex / max(1, n),
             "hosts": len(hosts), "excluded_sampled": len(sample),
             "inspect_docs": {k: len(v) for k, v in inspect.items()},
             "dialect_fast_vs_exact_agreement": agree / max(1, n_val), "dialect_validated_docs": n_val}
    with open(os.path.join(args.out_dir, f"stats_{shard}.json"), "w") as f:   # written last = shard done
        json.dump(stats, f, indent=1)
    return stats


def process(shard, path, args):
    import sys
    sys.path.insert(0, HERE)
    t0 = time.time()
    downloaded = False
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("allenai/c4", f"en.noblocklist/{shard}.json.gz", repo_type="dataset",
                               cache_dir=args.cache_dir)
        downloaded = True
    s = scan_file(path, shard, args)
    if downloaded and not args.keep_shards:
        os.remove(os.path.realpath(path))
    return (f"{shard}: {s['documents']} docs, {100 * s['excluded_rate']:.2f}% excluded, dialect fast/exact "
            f"agreement {100 * s['dialect_fast_vs_exact_agreement']:.1f}%, {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--range", type=int, nargs=2, metavar=("START", "END"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0, help="shard choice (seed 0 = Arm B's B1 shard indices)")
    ap.add_argument("--files", nargs="*", help="local noblocklist shard files instead of downloading")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sample_excluded", type=int, default=10000, help="excluded docs kept per shard for Fig. 6")
    ap.add_argument("--inspect_cap", type=int, default=500, help="excluded lesbian / gay docs kept per shard and term")
    ap.add_argument("--validate_dialect", type=int, default=1000, help="docs per shard also scored with exact CVB0")
    ap.add_argument("--keep_shards", action="store_true")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    import sys
    sys.path.insert(0, HERE)
    from dialect import ensure_files
    ensure_files(MODEL_DIR)                      # download once, before workers start

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
    meta_path = os.path.join(args.out_dir, "shards.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"shards": []}
    meta["shards"] = sorted(set(meta["shards"]) | {s for s, _ in jobs})
    meta["config"] = "en.noblocklist"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    todo = [(s, p) for s, p in jobs if not os.path.exists(os.path.join(args.out_dir, f"stats_{s}.json"))]
    print(f"{len(jobs)} shards requested, {len(jobs) - len(todo)} done already, {len(todo)} to scan", flush=True)
    t_all = time.time()
    if args.workers <= 1:
        for s, p in todo:
            print(process(s, p, args), flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(process, s, p, args) for s, p in todo]
            for i, fu in enumerate(as_completed(futs), 1):
                print(f"[{i}/{len(todo)}, {(time.time() - t_all) / 60:.0f} min] {fu.result()}", flush=True)


if __name__ == "__main__":
    main()
