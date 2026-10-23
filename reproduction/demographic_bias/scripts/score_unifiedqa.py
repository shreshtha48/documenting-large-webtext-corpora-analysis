"""Arm A of Dodge et al. 2021 §4.3: score UnQover ethnicity questions with UnifiedQA.

For every context we ask both questions (negative, positive) as 2-way multiple choice and record
the log-likelihood UnifiedQA assigns to each answer option. The choice is made later, in
analyze_arm_a.py, so the decision rule can be changed without re-running the model.

Input format (UnifiedQA README: question first, parts joined by a literal backslash-n):
    "<question> \\n (a) <subj0> (b) <subj1> \\n <context>"
Lowercased by default, matching the README's examples. Both are our assumptions; Dodge et al.
do not give their input format or how they mapped outputs to choices.

Usage:
    python scripts/score_unifiedqa.py --questions data/questions/ethnicity.jsonl.gz \
        --model allenai/unifiedqa-t5-3b --out_dir results/E6_full_3b/scores/unifiedqa-t5-3b
    # two GPUs on Kaggle: run two processes into the same out_dir
    CUDA_VISIBLE_DEVICES=0 python ... --part 0 --nparts 2 &
    CUDA_VISIBLE_DEVICES=1 python ... --part 1 --nparts 2 &

Results are written as one csv.gz per chunk (written to a temp name, then renamed), so an
interrupted run loses at most one chunk; re-running the same command resumes.
Columns: id, lp_neg_a, lp_neg_b, lp_pos_a, lp_pos_b, ntok_a, ntok_b, gen_neg, gen_pos, key
    lp_*      summed log-prob of the option tokens (including </s>)
    ntok_*    number of target tokens, for length-normalised scoring
    gen_*     greedy-decoded answer, only for the first --gen_check rows (sanity check)
    key       subj0|subj1|tid|attr of the scored row; analyze_arm_a.py checks it against the question
              file, so a regenerated or reordered question file cannot be joined to the wrong scores
"""
import argparse, csv, glob, gzip, json, os, time

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

COLS = ["id", "lp_neg_a", "lp_neg_b", "lp_pos_a", "lp_pos_b", "ntok_a", "ntok_b", "gen_neg", "gen_pos", "key"]


def row_key(r):
    return f"{r['subj0']}|{r['subj1']}|{r['tid']}|{r['attr']}"


def fmt(q, a, b, ctx, lower):
    s = f"{q} \\n (a) {a} (b) {b} \\n {ctx}"
    return s.lower() if lower else s


def load_rows(path, part, nparts, limit):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] % nparts == part:
                rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def done_ids(out_dir):
    ids = set()
    for p in glob.glob(os.path.join(out_dir, "chunk_*.csv.gz")):
        with gzip.open(p, "rt", newline="") as f:
            ids.update(int(r["id"]) for r in csv.DictReader(f))
    return ids


@torch.no_grad()
def option_logprobs(model, tok, enc, enc_out, options, device):
    lab = tok(text_target=options, return_tensors="pt", padding=True).input_ids.to(device)
    mask = lab != tok.pad_token_id
    out = model(encoder_outputs=enc_out, attention_mask=enc.attention_mask,
                labels=lab.masked_fill(~mask, -100))
    lp = out.logits.float().log_softmax(-1).gather(-1, lab.unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(1).tolist(), mask.sum(1).tolist()


@torch.no_grad()
def score_batch(model, tok, rows, device, lower, max_in, gen):
    # 2 inputs per row: negative question then positive question
    inputs, opt_a, opt_b = [], [], []
    for r in rows:
        for q in (r["q_neg"], r["q_pos"]):
            inputs.append(fmt(q, r["subj0"], r["subj1"], r["context"], lower))
            opt_a.append(r["subj0"].lower() if lower else r["subj0"])
            opt_b.append(r["subj1"].lower() if lower else r["subj1"])
    enc = tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_in).to(device)
    enc_out = BaseModelOutput(last_hidden_state=model.get_encoder()(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state)
    lp_a, n_a = option_logprobs(model, tok, enc, enc_out, opt_a, device)
    lp_b, n_b = option_logprobs(model, tok, enc, enc_out, opt_b, device)
    gens = [""] * len(inputs)
    if gen:
        g = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=16)
        gens = tok.batch_decode(g, skip_special_tokens=True)
    out = []
    for i, r in enumerate(rows):
        n, p = 2 * i, 2 * i + 1
        out.append([r["id"], lp_a[n], lp_b[n], lp_a[p], lp_b[p], n_a[n], n_b[n], gens[n], gens[p], row_key(r)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--model", default="allenai/unifiedqa-t5-large")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64, help="contexts per batch (x2 questions)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N contexts of this part")
    ap.add_argument("--gen_check", type=int, default=2000, help="greedy-decode the first N contexts")
    ap.add_argument("--no_lower", action="store_true")
    ap.add_argument("--max_in", type=int, default=128)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"],
                    help="T5 is known to overflow in fp16; keep fp32 on T4/P100, bf16 only on Ampere+")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    rows = load_rows(args.questions, args.part, args.nparts, args.limit)
    gen_ids = {r["id"] for r in rows[:args.gen_check]}
    os.makedirs(args.out_dir, exist_ok=True)
    skip = done_ids(args.out_dir)
    todo = [r for r in rows if r["id"] not in skip]
    print(f"{args.model}: {len(rows)} contexts in part {args.part}/{args.nparts}, {len(todo)} to do", flush=True)

    t0, n = time.time(), 0
    chunk = args.batch_size * 50
    for c in range(0, len(todo), chunk):
        buf = []
        for b in range(c, min(c + chunk, len(todo)), args.batch_size):
            batch = todo[b:b + args.batch_size]
            gen = any(r["id"] in gen_ids for r in batch)
            buf += score_batch(model, tok, batch, args.device, not args.no_lower, args.max_in, gen)
        path = os.path.join(args.out_dir, f"chunk_p{args.part}_{buf[0][0]:07d}.csv.gz")
        with gzip.open(path + ".tmp", "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(buf)
        os.replace(path + ".tmp", path)
        n += len(buf)
        rate = n / (time.time() - t0)
        print(f"  {n}/{len(todo)} contexts, {rate:.1f}/s, eta {(len(todo) - n) / rate / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
