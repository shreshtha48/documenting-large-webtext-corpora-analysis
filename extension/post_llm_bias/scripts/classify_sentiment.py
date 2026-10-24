"""Score every paragraph with a sentiment classifier (GPU if available), as a check on the word lexicons.

The scoping note asks whether a classifier changes the lexicon picture. Model: cardiffnlp/twitter-roberta-base-
sentiment-latest (negative / neutral / positive). It reads the whole paragraph (first --max_length tokens), so,
like the lexicons, it scores the paragraph's tone, not the tone towards the group.

For each paragraphs_<shard>.jsonl.gz it writes scores_<shard>.npy (float16, one row per line: p_neg, p_neu, p_pos),
in the same order as the lines. Finished files are skipped. Paragraphs are sorted by length into batches, which
roughly halves the time on a T4.

Usage: python classify_sentiment.py --paragraphs paragraphs/2019 paragraphs/CC-MAIN-2024-10 ...
"""
import argparse, glob, gzip, json, os, time

import numpy as np

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paragraphs", nargs="+", required=True, help="folders with paragraphs_*.jsonl.gz")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()
    if dev == "cuda":
        model.half()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    order = [labels["negative"], labels["neutral"], labels["positive"]]
    print(f"{args.model} on {dev}; label order {order}", flush=True)

    for folder in args.paragraphs:
        for path in sorted(glob.glob(os.path.join(folder, "paragraphs_*.jsonl.gz"))):
            out = path.replace("paragraphs_", "scores_").replace(".jsonl.gz", ".npy")
            if os.path.exists(out):
                continue
            t0 = time.time()
            texts = [json.loads(l)["text"] for l in gzip.open(path, "rt", encoding="utf-8")]
            idx = np.argsort([len(t) for t in texts])
            probs = np.zeros((len(texts), 3), dtype=np.float32)
            with torch.inference_mode():
                for b in range(0, len(texts), args.batch):
                    sel = idx[b:b + args.batch]
                    enc = tok([texts[i] for i in sel], truncation=True, max_length=args.max_length,
                              padding=True, return_tensors="pt").to(dev)
                    p = torch.softmax(model(**enc).logits.float(), -1)[:, order]
                    probs[sel] = p.cpu().numpy()
            np.save(out + ".tmp.npy", probs.astype(np.float16))
            os.replace(out + ".tmp.npy", out)
            print(f"{path}: {len(texts)} paragraphs, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
