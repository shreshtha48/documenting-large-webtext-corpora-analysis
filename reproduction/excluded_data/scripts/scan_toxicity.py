"""Arm B of Dodge et al. 2021 §5: audit C4 filtering using a neural toxicity classifier.

Instead of a crude keyword blocklist (LDNOOBW), this script evaluates web text with an actual
context-aware toxicity classification model (e.g., unitary/toxic-bert) side-by-side with the
original LDNOOBW regex filter on the exact same documents.

For every document in C4.EN.NOBLOCKLIST, it records:
1. Excluded by Blocklist: matches LDNOOBW regex.
2. Excluded by Toxicity Classifier: model toxicity probability >= threshold (default 0.5).
3. Identity terms: matches any of the 22 identity regexes (Table 6).
4. Dialect: predicted by the Blodgett et al. TwitterAAE model (AAE, Hispanic, Other, WAE).
5. Filter agreement: both kept, both excluded, blocklist-only excluded, toxicity-only excluded.

Counts are aggregated per web domain (URL host) and saved as counts_<shard>.npz for domain bootstrapping.

Usage:
    # On Kaggle with GPU (T4 x2 or single T4):
    python scripts/scan_toxicity.py --n 4 --seed 0 --batch_size 128 --out_dir results/arm_b_toxicity_4_shards
    # On CPU:
    python scripts/scan_toxicity.py --n 1 --seed 0 --batch_size 32 --limit 50000 --out_dir results/test_toxicity
"""
import argparse, gzip, json, os, random, re, time
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BADWORDS = os.path.join(ROOT, "data", "ldnoobw_en_25e679f.txt")
MODEL_DIR = os.path.join(ROOT, "data", "twitteraae_model")
N_SHARDS = 1024

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

# Columns stored per host in the count matrix
COLUMNS = (
    ["docs", "ex_blocklist", "ex_toxic", "both_ex", "blocklist_only", "toxic_only", "both_kept"]
    + [f"bl:{t}|{k}" for t, _ in IDENTITY for k in ("docs", "excluded")]
    + [f"tx:{t}|{k}" for t, _ in IDENTITY for k in ("docs", "excluded")]
    + [f"bl:dialect:{d}|{k}" for d in DIALECTS for k in ("docs", "excluded")]
    + [f"tx:dialect:{d}|{k}" for d in DIALECTS for k in ("docs", "excluded")]
)


def badwords_regex():
    words = [w.strip() for w in open(BADWORDS, encoding="utf-8") if w.strip()]
    return re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))


def identity_regex():
    return re.compile("|".join(rf"\b(?P<g{i}>{p})\b" for i, (_, p) in enumerate(IDENTITY)), re.I)


def shard_name(i):
    return f"c4-train.{i:05d}-of-{N_SHARDS:05d}"


class BatchedToxicityClassifier:
    def __init__(self, model_name, device, max_length=256):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.is_multilabel = getattr(self.model.config, "problem_type", None) == "multi_label_classification" or \
                             self.model.config.num_labels > 2

    @torch.no_grad()
    def score_batch(self, texts):
        if not texts:
            return []
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        use_amp = self.device.type == "cuda"
        with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
            logits = self.model(**inputs).logits

        if self.is_multilabel:
            # Multi-label classification (e.g., toxic, severe_toxic, obscene, threat, insult, identity_hate)
            probs = torch.sigmoid(logits)
            # Max toxicity score across any category (or specifically primary 'toxic' column index 0)
            scores = probs.max(dim=-1).values.cpu().numpy()
        else:
            # Binary classification (label 1 = toxic)
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, 1].cpu().numpy()

        return scores.tolist()


def scan_file(path, shard, tox_clf, args):
    from dialect import DialectModel
    bad, ident = badwords_regex(), identity_regex()
    dialect_model = DialectModel(MODEL_DIR)
    col = {c: i for i, c in enumerate(COLUMNS)}
    per_host = defaultdict(lambda: np.zeros(len(COLUMNS), dtype=np.int64))

    discrepancies = []
    n = n_ex_bl = n_ex_tx = 0
    batch_docs = []

    def flush_batch(batch):
        nonlocal n, n_ex_bl, n_ex_tx
        texts = [d["text"] for d in batch]
        scores = tox_clf.score_batch(texts)

        for d, tx_score in zip(batch, scores):
            text, url = d["text"], d["url"]
            n += 1
            ex_bl = bool(bad.search(text.lower()))
            ex_tx = bool(tx_score >= args.threshold)

            n_ex_bl += ex_bl
            n_ex_tx += ex_tx

            host = urlparse(url).netloc
            row = per_host[host]

            row[col["docs"]] += 1
            row[col["ex_blocklist"]] += ex_bl
            row[col["ex_toxic"]] += ex_tx

            if ex_bl and ex_tx:
                row[col["both_ex"]] += 1
            elif ex_bl and not ex_tx:
                row[col["blocklist_only"]] += 1
            elif not ex_bl and ex_tx:
                row[col["toxic_only"]] += 1
            else:
                row[col["both_kept"]] += 1

            # Identity term hits
            hit = {int(m.lastgroup[1:]) for m in ident.finditer(text)}
            for i in hit:
                term_name = IDENTITY[i][0]
                row[col[f"bl:{term_name}|docs"]] += 1
                row[col[f"bl:{term_name}|excluded"]] += ex_bl
                row[col[f"tx:{term_name}|docs"]] += 1
                row[col[f"tx:{term_name}|excluded"]] += ex_tx

            # Dialect
            toks = dialect_model.tokens(text)
            post = dialect_model.fast(toks)
            dl = DIALECTS[int(np.argmax(post))] if post is not None else "unassigned"

            row[col[f"bl:dialect:{dl}|docs"]] += 1
            row[col[f"bl:dialect:{dl}|excluded"]] += ex_bl
            row[col[f"tx:dialect:{dl}|docs"]] += 1
            row[col[f"tx:dialect:{dl}|excluded"]] += ex_tx

            # Collect informative discrepancies for qualitative audit
            if ex_bl != ex_tx and len(discrepancies) < args.sample_discrepancies:
                discrepancies.append({
                    "url": url,
                    "ex_blocklist": ex_bl,
                    "ex_toxic": ex_tx,
                    "toxic_score": round(float(tx_score), 4),
                    "dialect": dl,
                    "identity_hits": [IDENTITY[i][0] for i in hit],
                    "excerpt": text[:500]
                })

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            batch_docs.append(d)
            if len(batch_docs) >= args.batch_size:
                flush_batch(batch_docs)
                batch_docs = []
            if args.limit and n >= args.limit:
                break
        if batch_docs:
            flush_batch(batch_docs)

    hosts = list(per_host)
    counts_arr = np.stack([per_host[h] for h in hosts]) if hosts else np.zeros((0, len(COLUMNS)), dtype=np.int64)

    np.savez_compressed(
        os.path.join(args.out_dir, f"counts_{shard}.npz"),
        hosts=np.array(hosts, dtype=str),
        counts=counts_arr,
        columns=np.array(COLUMNS)
    )

    if discrepancies:
        disc_path = os.path.join(args.out_dir, f"discrepancies_{shard}.jsonl.gz")
        with gzip.open(disc_path, "wt", encoding="utf-8") as g:
            for r in discrepancies:
                g.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "shard": shard,
        "documents": n,
        "ex_blocklist": n_ex_bl,
        "ex_blocklist_rate": n_ex_bl / max(1, n),
        "ex_toxic": n_ex_tx,
        "ex_toxic_rate": n_ex_tx / max(1, n),
        "hosts": len(hosts),
        "discrepancies_sampled": len(discrepancies)
    }

    with open(os.path.join(args.out_dir, f"stats_{shard}.json"), "w") as f:
        json.dump(stats, f, indent=1)

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="unitary/toxic-bert", help="HuggingFace toxicity classification model")
    ap.add_argument("--threshold", type=float, default=0.5, help="Toxicity score threshold for exclusion")
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--range", type=int, nargs=2, metavar=("START", "END"))
    ap.add_argument("--n", type=int, default=4, help="Number of seeded shards to scan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--files", nargs="*", help="Local noblocklist shard files")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="Max documents per shard (useful for quick tests)")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--sample_discrepancies", type=int, default=1000)
    ap.add_argument("--keep_shards", action="store_true")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading toxicity model: {args.model} on {device} ...", flush=True)
    tox_clf = BatchedToxicityClassifier(args.model, device, max_length=args.max_length)

    from dialect import ensure_files
    ensure_files(MODEL_DIR)

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
    meta["model"] = args.model
    meta["threshold"] = args.threshold
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    print(f"Processing {len(jobs)} shards with {args.model} (threshold={args.threshold}) ...", flush=True)
    for shard, path in jobs:
        done_stat = os.path.join(args.out_dir, f"stats_{shard}.json")
        if os.path.exists(done_stat):
            s = json.load(open(done_stat))
            print(f"Skipping completed shard {shard} ({s['documents']:,} docs)", flush=True)
            continue

        t0 = time.time()
        downloaded = False
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("allenai/c4", f"en.noblocklist/{shard}.json.gz", repo_type="dataset",
                                   cache_dir=args.cache_dir)
            downloaded = True

        s = scan_file(path, shard, tox_clf, args)
        if downloaded and not args.keep_shards:
            os.remove(os.path.realpath(path))

        elapsed = time.time() - t0
        print(f"[{shard}] {s['documents']:,} docs in {elapsed:.0f}s | "
              f"Blocklist Excluded: {100 * s['ex_blocklist_rate']:.2f}% | "
              f"Toxicity Excluded: {100 * s['ex_toxic_rate']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
