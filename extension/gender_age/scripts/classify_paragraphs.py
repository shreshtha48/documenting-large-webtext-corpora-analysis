"""Extension, corpus: score the sampled paragraphs with a sentiment classifier (GPU if available).

The reproduction's corpus sentiment (A.7) counts lexicon words. Our scoping note asked whether a trained classifier
changes the picture, so the paragraphs sampled by scan_corpus.py (para_sample_*.jsonl.gz, a random sample per term,
from kept and from excluded documents) are scored with a 3-class sentiment model. Like the lexicon count, this is the
sentiment of the whole paragraph, not of the attitude toward the person mentioned.

Default model: cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa-base, negative / neutral / positive).
Inputs are truncated to 256 tokens.

Writes <scan_dir>/para_scores.csv.gz: term, kept, year, url, p_neg, p_neu, p_pos. Resumable: paragraphs already in
the output (by term, url, text hash) are skipped.

Usage:
    python scripts/classify_paragraphs.py --scan results/C1_noblocklist_12_shards
"""
import argparse, csv, glob, gzip, hashlib, json, os, time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def key(r):
    return f"{r['term']}|{r['url']}|{hashlib.md5(r['text'].encode()).hexdigest()[:12]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="only the first N paragraphs (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.scan, "para_sample_*.jsonl.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rows += [json.loads(l) for l in f]
    if args.limit:
        rows = rows[:args.limit]
    out = os.path.join(args.scan, "para_scores.csv.gz")
    done = set()
    if os.path.exists(out):
        with gzip.open(out, "rt", newline="", encoding="utf-8") as f:
            done = {r["key"] for r in csv.DictReader(f)}
    todo = [r for r in rows if key(r) not in done]
    print(f"{len(rows)} sampled paragraphs, {len(todo)} to score on {args.device}", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    idx = {l: labels.index(l) for l in ("negative", "neutral", "positive")}
    new = not os.path.exists(out)
    t0 = time.time()
    with gzip.open(out, "at", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["key", "term", "kept", "year", "url", "p_neg", "p_neu", "p_pos"])
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b:b + args.batch_size]
            enc = tok([r["text"] for r in batch], return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len).to(args.device)
            with torch.no_grad():
                p = model(**enc).logits.float().softmax(-1).cpu().numpy()
            for r, q in zip(batch, p):
                w.writerow([key(r), r["term"], r["kept"], r["year"], r["url"], round(float(q[idx["negative"]]), 5),
                            round(float(q[idx["neutral"]]), 5), round(float(q[idx["positive"]]), 5)])
            if (b // args.batch_size) % 50 == 0:
                n = b + len(batch)
                print(f"  {n}/{len(todo)}, {n / (time.time() - t0):.0f}/s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
