"""Extension, probe: score BBQ examples with UnifiedQA (three answer options).

Input: BBQ's UnifiedQA "RACE-style" format (BBQ README: question, the three options as (a) (b) (c), then the
context), written the way the reproduction writes UnifiedQA inputs (score_unifiedqa.py): parts joined by a literal
backslash-n, lowercased:
    "<question> \\n (a) <ans0> (b) <ans1> (c) <ans2> \\n <context>"
Two answer rules, both recorded:
- BBQ's own: greedy generation, matched to an option string (analyze_bbq.py does the matching, as the official
  R script does);
- the reproduction's: summed log-likelihood of each option (</s> included), the higher wins.

Writes one csv.gz per chunk into --out_dir (resumable, two GPUs can split with --part / --nparts), columns:
    id, lp0, lp1, lp2, ntok0, ntok1, ntok2, gen

Usage:
    python scripts/score_bbq.py --categories Age Gender_identity Race_x_gender --out_dir results/P2_bbq_3b/scores
"""
import argparse, csv, glob, gzip, json, os, sys, time

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "reproduction", "demographic_bias", "scripts"))
from score_unifiedqa import option_logprobs  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data", "bbq")
COLS = ["id", "lp0", "lp1", "lp2", "ntok0", "ntok1", "ntok2", "gen"]


def fmt(r):
    return f"{r['question']} \\n (a) {r['ans0']} (b) {r['ans1']} (c) {r['ans2']} \\n {r['context']}".lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["Age", "Gender_identity", "Race_x_gender"])
    ap.add_argument("--model", default="allenai/unifiedqa-t5-3b")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_in", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = []
    for cat in args.categories:
        with gzip.open(os.path.join(DATA, f"{cat}.jsonl.gz"), "rt", encoding="utf-8") as f:
            rows += [json.loads(l) for l in f]
    rows = [r for i, r in enumerate(rows) if i % args.nparts == args.part]
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    done = set()
    for p in glob.glob(os.path.join(args.out_dir, "chunk_*.csv.gz")):
        with gzip.open(p, "rt", newline="", encoding="utf-8") as f:
            done |= {r["id"] for r in csv.DictReader(f)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.model}: {len(rows)} examples in part {args.part}/{args.nparts}, {len(todo)} to do", flush=True)
    if not todo:
        print("done", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(args.device).eval()
    t0, n = time.time(), 0
    chunk = args.batch_size * 50
    for c in range(0, len(todo), chunk):
        buf = []
        for b in range(c, min(c + chunk, len(todo)), args.batch_size):
            batch = todo[b:b + args.batch_size]
            enc = tok([fmt(r) for r in batch], return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_in).to(args.device)
            with torch.no_grad():
                enc_out = BaseModelOutput(last_hidden_state=model.get_encoder()(
                    input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state)
                lps, nts = [], []
                for k in range(3):
                    lp, nt = option_logprobs(model, tok, enc, enc_out, [r[f"ans{k}"].lower() for r in batch], args.device)
                    lps.append(lp), nts.append(nt)
                g = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=16)
                gens = tok.batch_decode(g, skip_special_tokens=True)
            for i, r in enumerate(batch):
                buf.append([r["id"], lps[0][i], lps[1][i], lps[2][i], nts[0][i], nts[1][i], nts[2][i], gens[i]])
        path = os.path.join(args.out_dir, f"chunk_p{args.part}_{c:07d}.csv.gz")
        with gzip.open(path + ".tmp", "wt", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(buf)
        os.replace(path + ".tmp", path)
        n += len(buf)
        rate = n / (time.time() - t0)
        print(f"  {n}/{len(todo)}, {rate:.1f}/s, eta {(len(todo) - n) / rate / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
