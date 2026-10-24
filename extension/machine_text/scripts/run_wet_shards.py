"""Run filter_c4_pipeline.py over many Common Crawl WET shards, several at a time, resumably.

Shards are drawn per crawl with a fixed seed from the crawl's full wet.paths.gz, so they spread over many crawl
segments (consecutive indices sit in the same segment). --include adds indices that must be in the sample (e.g.
the 0 and 3 already processed), and the rest are drawn at random up to --n_per_crawl.

Each shard is one run_pipeline() call, so URL and line dedupe happen within a shard, as in the 4-shard pilot.
Its log goes to <out_dir>/<crawl>/logs/, the WET file is deleted afterwards (unless --keep_wet), and a shard
whose stats_*.json exists is skipped, so re-running the same command resumes.

Writes per crawl: <out_dir>/<crawl>/{c4_en,c4_blocklist,stats}_<crawl>_shard<idx>.* and shards.json
(the chosen indices and WET URLs).

Usage:
    python run_wet_shards.py --crawls CC-MAIN-2024-10 CC-MAIN-2024-51 --n_per_crawl 24 --include 0 3 \
        --workers 4 --out_dir post_llm_c4
"""
import argparse, contextlib, gzip, io, json, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filter_c4_pipeline as c4


def wet_paths(crawl_id):
    with c4._open_url(f"{c4.CC_BASE_URL}crawl-data/{crawl_id}/wet.paths.gz", timeout=60) as resp:
        data = resp.read()
    return [l.strip() for l in gzip.decompress(data).decode("utf-8").splitlines() if l.strip()]


def choose(crawl_id, n, include, seed):
    paths = wet_paths(crawl_id)
    rest = [i for i in range(len(paths)) if i not in set(include)]
    picked = sorted(set(include) | set(random.Random(f"{seed}-{crawl_id}").sample(rest, max(0, n - len(include)))))
    return {i: c4.CC_BASE_URL + paths[i] for i in picked}, len(paths)


def one_shard(crawl_id, idx, url, out_dir, cache_dir, keep_wet):
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    t0 = time.time()
    log = os.path.join(out_dir, "logs", f"shard{idx}.log")
    with open(log, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        s = c4.run_pipeline(source="commoncrawl", crawl_id=crawl_id, wet_url=url, shard_idx=idx,
                            out_dir=out_dir, cache_dir=cache_dir)
    if not keep_wet:
        p = os.path.join(cache_dir, url.rsplit("/", 1)[-1])
        if os.path.exists(p):
            os.remove(p)
    return (f"{crawl_id} shard {idx}: {s['total_documents_scanned']} pages -> C4.EN {s['c4_en_kept']}, "
            f"blocklist {s['blocklist_excluded']} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawls", nargs="+", default=["CC-MAIN-2024-10", "CC-MAIN-2024-51"])
    ap.add_argument("--n_per_crawl", type=int, default=24)
    ap.add_argument("--include", type=int, nargs="*", default=[0, 3], help="indices always in the sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", default="post_llm_c4")
    ap.add_argument("--cache_dir", default="wet_cache")
    ap.add_argument("--keep_wet", action="store_true")
    args = ap.parse_args()

    jobs = []
    for crawl in args.crawls:
        d = os.path.join(args.out_dir, crawl)
        os.makedirs(d, exist_ok=True)
        meta_path = os.path.join(d, "shards.json")
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            urls = {int(k): v for k, v in meta["urls"].items()}
            if meta["n_per_crawl"] != args.n_per_crawl or meta["seed"] != args.seed:
                raise SystemExit(f"{meta_path} was made with n_per_crawl={meta['n_per_crawl']}, seed={meta['seed']}; "
                                 "use the same values or another --out_dir")
        else:
            urls, total = choose(crawl, args.n_per_crawl, args.include, args.seed)
            with open(meta_path, "w") as f:
                json.dump({"crawl": crawl, "n_per_crawl": args.n_per_crawl, "seed": args.seed,
                           "include": args.include, "wet_files_in_crawl": total, "urls": urls}, f, indent=1)
        for idx, url in sorted(urls.items()):
            if not os.path.exists(os.path.join(d, f"stats_{crawl}_shard{idx}.json")):
                jobs.append((crawl, idx, url, d))
    n_all = len(args.crawls) * args.n_per_crawl
    print(f"{n_all} shards requested, {n_all - len(jobs)} done already, {len(jobs)} to run", flush=True)

    t_all, failed = time.time(), []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one_shard, c, i, u, d, args.cache_dir, args.keep_wet): (c, i) for c, i, u, d in jobs}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                msg = fu.result()
            except Exception as e:      # one bad download must not stop the rest; re-running retries it
                failed.append(futs[fu])
                msg = f"{futs[fu]} FAILED: {type(e).__name__}: {e}"
            print(f"[{k}/{len(jobs)}, {(time.time() - t_all) / 60:.0f} min] {msg}", flush=True)
    if failed:
        print(f"{len(failed)} shards failed: {failed}. Re-run the same command to retry them.")


if __name__ == "__main__":
    main()
