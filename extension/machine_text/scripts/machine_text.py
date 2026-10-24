"""How much of post-LLM C4 is machine-generated, did it grow from Feb to Dec 2024, and is that text's tone different?

Corpora: the post-LLM C4 shards (C4.EN and C4.BLOCKLIST from filter_c4_pipeline.py, one corpus per crawl) and a
2019 C4.EN shard from allenai/c4 as the negative control: it predates ChatGPT, so whatever a detector flags there
is its false-positive level on this kind of web text.

Two independent measures of machine text (if they agree, neither is just an artefact):
  1. Marker words (Kobak et al. 2024 style "excess vocabulary"): share of documents whose first 300 words contain
     any pre-registered LLM style word, against control words that LLMs did not change. Lower bound on the share
     of LLM-processed documents: f >= (p - q) / (1 - q), p = share now, q = share in the comparison corpus.
  2. Binoculars (Hans et al. 2024): zero-shot score = log-perplexity under a performer model / cross-perplexity
     between an observer and that performer; low = machine-like. The threshold is set on 2019 documents (1% and
     5% false-positive rates), so "flagged" means "more machine-like than 99% of 2019 C4".
     The same 2019 check is repeated per ethnicity group: detectors flag non-native English more often (Liang et
     al. 2023), which would bias any group comparison.

Tone: lexicon positivity (Arm B's lexicons) and RoBERTa sentiment of flagged vs unflagged documents, and of
group-mentioning paragraphs inside them. 4 shards give wide CIs; they are reported, not hidden.

All CIs: 95% percentile, web-domain bootstrap (B=1000).

Steps (each resumable; run in order):
    python machine_text.py docs --c4_files "post-llm-crawl/*/c4_*.jsonl.gz" --out work      # + 2019 shard 00000
    python machine_text.py detect --out work            # GPU: Binoculars
    python machine_text.py classify --out work          # GPU: RoBERTa sentiment (optional)
    python machine_text.py analyze --out work
"""
import argparse, glob, gzip, hashlib, json, os, re, sys, time
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "..", "..", "reproduction", "demographic_bias", "scripts")):
    sys.path.insert(0, os.path.abspath(p))

WINDOW = 300          # words read per document for markers and sentiment
MIN_WORDS = 100       # shorter documents are left out of every analysis (detectors are unreliable on them)
KEEP_WORDS = 600      # words stored per document (the detector reads up to --max_tokens of them)
# Pre-registered on 2026-10-24, before looking at their 2019 rates. LLM style words from Kobak et al. 2024 /
# Liang et al. 2024 that are not topic words; matched as whole lower-cased tokens.
MARKERS = ["delve", "delves", "delving", "underscore", "underscores", "underscoring", "showcase", "showcases",
           "showcasing", "pivotal", "intricate", "intricacies", "meticulous", "meticulously", "realm", "tapestry",
           "testament", "unwavering", "embark", "embarking", "elevate", "elevating", "seamless", "seamlessly",
           "bustling", "commendable", "noteworthy", "invaluable", "holistic", "furthermore"]
CONTROLS = ["however", "important", "people", "example", "information", "including", "between", "because",
            "different", "during", "without", "although"]
CORPUS_2019 = "2019 C4"
TOKEN = re.compile(r"[a-z]+")


def crawl_of(path):
    m = re.search(r"(CC-MAIN-\d{4}-\d{2})", os.path.basename(path))
    return m.group(1) if m else os.path.basename(os.path.dirname(path))


def _groups():
    from scan_c4_paragraphs import TABLE7_GROUPS
    return {t: [f" {v} " for v in vs] for t, vs in TABLE7_GROUPS.items()}


def doc_record(corpus, split, sample, url, text, pats):
    words = text.split()
    gp = []
    for p in text.split("\n"):
        hit = [t for t, ps in pats.items() if any(x in p for x in ps)]
        if hit:
            gp.append([hit, p])
    return {"corpus": corpus, "split": split, "sample": sample, "url": url, "host": urlparse(url).netloc,
            "n_words": len(words), "text": " ".join(words[:KEEP_WORDS]), "group_paragraphs": gp}


# ----------------------------------------------------------------------------------------------- docs
def cmd_docs(args):
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "docs.jsonl.gz")
    if os.path.exists(path):
        print(f"{path} exists; delete it to rebuild")
        return
    pats = _groups()
    n = defaultdict(int)
    with gzip.open(path + ".tmp", "wt", encoding="utf-8") as g:
        files = sorted(glob.glob(args.c4_files, recursive=True))
        if not files:
            raise SystemExit(f"no files match {args.c4_files}")
        for f in files:
            corpus, split = crawl_of(f), ("blocklist" if "c4_blocklist_" in os.path.basename(f) else "en")
            for line in gzip.open(f, "rt", encoding="utf-8"):
                d = json.loads(line)
                if len(d["text"].split()) < MIN_WORDS:
                    n[(corpus, "too_short")] += 1
                    continue
                g.write(json.dumps(doc_record(corpus, split, "all", d["url"], d["text"], pats), ensure_ascii=False) + "\n")
                n[(corpus, split)] += 1

        # 2019: a seeded random sample, plus up to --per_group documents mentioning each group (per-group FPR)
        from huggingface_hub import hf_hub_download
        shard = f"en/c4-train.{args.shard_2019:05d}-of-01024.json.gz"
        local = hf_hub_download("allenai/c4", shard, repo_type="dataset", cache_dir=args.cache_dir)
        rank = lambda u: hashlib.md5(f"{args.seed}\t{u}".encode()).hexdigest()
        docs = [json.loads(l) for l in gzip.open(local, "rt", encoding="utf-8")]
        docs = [d for d in docs if len(d["text"].split()) >= MIN_WORDS]
        docs.sort(key=lambda d: rank(d["url"]))
        chosen, per_group = {}, defaultdict(int)
        for d in docs[:args.n_2019]:
            chosen[d["url"]] = "random"
        for d in docs:
            if d["url"] in chosen:
                continue
            hit = [t for t, ps in pats.items() if any(x in d["text"] for x in ps)]
            need = [t for t in hit if per_group[t] < args.per_group]
            if need:
                chosen[d["url"]] = "group"
                for t in hit:
                    per_group[t] += 1
        for d in docs:
            if d["url"] in chosen:
                g.write(json.dumps(doc_record(CORPUS_2019, "en", chosen[d["url"]], d["url"], d["text"], pats),
                                   ensure_ascii=False) + "\n")
                n[(CORPUS_2019, chosen[d["url"]])] += 1
        if not args.keep_2019:
            os.remove(os.path.realpath(local))
    os.replace(path + ".tmp", path)
    print({f"{a} / {b}": v for (a, b), v in sorted(n.items())})


def load_docs(out):
    return [json.loads(l) for l in gzip.open(os.path.join(out, "docs.jsonl.gz"), "rt", encoding="utf-8")]


# ----------------------------------------------------------------------------------------------- detect
def cmd_detect(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    docs = load_docs(args.out)
    path = os.path.join(args.out, "binoculars.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["i"] for l in open(path)}
    wanted = range(len(docs))
    if args.max_per_corpus:      # a seeded random subset of each post-LLM corpus; 2019 documents are all kept
        by = defaultdict(list)
        for i, d in enumerate(docs):
            by[d["corpus"]].append(i)
        rank = lambda i: hashlib.md5(f"{args.seed}	{docs[i]['url']}".encode()).hexdigest()
        wanted = sorted(i for c, ids in by.items()
                        for i in (ids if c == CORPUS_2019 else sorted(ids, key=rank)[:args.max_per_corpus]))
    todo = [i for i in wanted if i not in done]
    print(f"{len(wanted)} documents selected, {len(done)} scored already, {len(todo)} to score", flush=True)
    if not todo:
        return

    ngpu = torch.cuda.device_count()
    d_obs = "cuda:0" if ngpu else "cpu"
    d_perf = "cuda:1" if ngpu > 1 else d_obs
    dtype = torch.float16 if ngpu else torch.float32
    tok = AutoTokenizer.from_pretrained(args.observer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    obs = AutoModelForCausalLM.from_pretrained(args.observer, torch_dtype=dtype).to(d_obs).eval()
    perf = AutoModelForCausalLM.from_pretrained(args.performer, torch_dtype=dtype).to(d_perf).eval()
    print(f"observer {args.observer} on {d_obs}, performer {args.performer} on {d_perf}", flush=True)

    todo.sort(key=lambda i: len(docs[i]["text"]))     # similar lengths per batch: less padding
    t0 = time.time()
    with open(path, "a") as out, torch.inference_mode():
        for k in range(0, len(todo), args.batch):
            ids = todo[k:k + args.batch]
            enc = tok([docs[i]["text"] for i in ids], return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_tokens)
            lo = obs(**enc.to(d_obs)).logits[:, :-1].float()
            lp = perf(**enc.to(d_perf)).logits[:, :-1].to(d_obs).float()
            labels = enc["input_ids"][:, 1:].to(d_obs)
            mask = enc["attention_mask"][:, 1:].to(d_obs).float()
            log_ppl = (F.cross_entropy(lp.transpose(1, 2), labels, reduction="none") * mask).sum(1) / mask.sum(1)
            x_ent = -(F.softmax(lo, -1) * F.log_softmax(lp, -1)).sum(-1)
            x_ppl = (x_ent * mask).sum(1) / mask.sum(1)
            score = (log_ppl / x_ppl).cpu().numpy()
            for i, s, nt in zip(ids, score, mask.sum(1).cpu().numpy()):
                out.write(json.dumps({"i": i, "score": float(s), "tokens": int(nt) + 1}) + "\n")
            out.flush()
            if (k // args.batch) % 50 == 0:
                rate = (k + len(ids)) / (time.time() - t0)
                print(f"{k + len(ids)}/{len(todo)} ({rate:.1f} docs/s, ~{(len(todo) - k) / rate / 60:.0f} min left)", flush=True)


# ----------------------------------------------------------------------------------------------- classify
def cmd_classify(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    docs = load_docs(args.out)
    path = os.path.join(args.out, "roberta.npy")
    if os.path.exists(path):
        print(f"{path} exists")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()
    if dev == "cuda":
        model.half()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    order = [lab["negative"], lab["neutral"], lab["positive"]]
    texts = [" ".join(d["text"].split()[:WINDOW]) for d in docs]
    idx = np.argsort([len(t) for t in texts])
    probs = np.zeros((len(texts), 3), dtype=np.float32)
    with torch.inference_mode():
        for b in range(0, len(texts), args.batch):
            sel = idx[b:b + args.batch]
            enc = tok([texts[i] for i in sel], truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
            probs[sel] = torch.softmax(model(**enc).logits.float(), -1)[:, order].cpu().numpy()
    np.save(path, probs)
    print(f"scored {len(texts)} documents on {dev}")


# ----------------------------------------------------------------------------------------------- analyze
def domain_boot(hosts, values, B, seed):
    """values: (docs, ..., k) array of sums. Returns point ratio-free sums per replicate: (B, ..., k)."""
    uh, inv = np.unique(hosts, return_inverse=True)
    X = np.zeros((len(uh),) + values.shape[1:])
    np.add.at(X, inv, values)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(uh), np.full(len(uh), 1 / len(uh)), size=B)
    return X.sum(0), np.tensordot(w, X, axes=(1, 0))


def rate_ci(hosts, num, den, B, seed):
    """Ratio sum(num)/sum(den) in %, with a domain-bootstrap CI, and the replicates."""
    tot, rep = domain_boot(hosts, np.stack([num, den], -1).astype(float), B, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 100 * rep[..., 0] / rep[..., 1]
        pt = 100 * tot[..., 0] / tot[..., 1]
    lo, hi = np.nanpercentile(r, [2.5, 97.5], axis=0)
    return pt, lo, hi, r


def cmd_analyze(args):
    import pandas as pd
    import count_sentiment as cs
    docs = load_docs(args.out)
    os.makedirs(os.path.join(args.out, "analysis"), exist_ok=True)
    A = lambda name: os.path.join(args.out, "analysis", name)
    corpus = np.array([d["corpus"] for d in docs])
    hosts = np.array([d["host"] for d in docs])
    sample = np.array([d["sample"] for d in docs])
    corpora = [CORPUS_2019] + sorted(c for c in set(corpus) if c != CORPUS_2019)
    base = (sample != "group")         # random 2019 docs + every post-LLM doc; 2019 "group" docs only for FPR checks
    B, seed = args.B, args.seed

    # 1. marker words
    vocab = MARKERS + CONTROLS
    has = np.zeros((len(docs), len(vocab)), dtype=bool)
    for i, d in enumerate(docs):
        toks = set(TOKEN.findall(" ".join(d["text"].split()[:WINDOW]).lower()))
        has[i] = [w in toks for w in vocab]
    any_m = has[:, :len(MARKERS)].any(1)
    any_c = has[:, len(MARKERS):].any(1)
    rows, reps = [], {}
    for c in corpora:
        s = base & (corpus == c)
        for name, col in [("ANY MARKER", any_m), ("ANY CONTROL", any_c)] + [(w, has[:, j]) for j, w in enumerate(vocab)]:
            pt, lo, hi, r = rate_ci(hosts[s], col[s], np.ones(s.sum()), B, seed)
            reps[(c, name)] = r
            rows.append({"corpus": c, "word": name, "kind": "marker" if name in MARKERS or name == "ANY MARKER" else "control",
                         "docs": int(s.sum()), "doc_rate_pct": pt, "ci_lo": lo, "ci_hi": hi})
    mk = pd.DataFrame(rows)
    mk.round(3).to_csv(A("marker_rates.csv"), index=False)

    # lower bound on the LLM-processed share, vs 2019 and vs the previous crawl
    rows = []
    for k, c in enumerate(corpora[1:], 1):
        for q_c in [CORPUS_2019] + ([corpora[k - 1]] if k > 1 else []):
            for name in ("ANY MARKER", "ANY CONTROL"):
                p, q = reps[(c, name)] / 100, reps[(q_c, name)] / 100
                f = 100 * (p - q) / (1 - q)
                pt_p = mk[(mk.corpus == c) & (mk.word == name)].doc_rate_pct.iloc[0] / 100
                pt_q = mk[(mk.corpus == q_c) & (mk.word == name)].doc_rate_pct.iloc[0] / 100
                lo, hi = np.nanpercentile(f, [2.5, 97.5])
                rows.append({"corpus": c, "compared_with": q_c, "words": name,
                             "rate_now_pct": 100 * pt_p, "rate_then_pct": 100 * pt_q,
                             "lower_bound_llm_share_pct": 100 * (pt_p - pt_q) / (1 - pt_q), "ci_lo": lo, "ci_hi": hi})
    lb = pd.DataFrame(rows)
    lb.round(3).to_csv(A("llm_share_lower_bound.csv"), index=False)

    # 2. Binoculars
    det_path = os.path.join(args.out, "binoculars.jsonl")
    score = np.full(len(docs), np.nan)
    if os.path.exists(det_path):
        for l in open(det_path):
            r = json.loads(l)
            score[r["i"]] = r["score"]
    flagged = {}
    det = gp_rate = agree = pd.DataFrame()
    if np.isfinite(score).any():
        ref = score[(corpus == CORPUS_2019) & (sample == "random") & np.isfinite(score)]
        thr = {fpr: np.percentile(ref, fpr) for fpr in (1, 5)}
        rows = []
        for fpr, t in thr.items():
            flagged[fpr] = score < t
            for c in corpora:
                s = base & (corpus == c) & np.isfinite(score)
                pt, lo, hi, _ = rate_ci(hosts[s], flagged[fpr][s], np.ones(s.sum()), B, seed)
                rows.append({"fpr_on_2019_pct": fpr, "threshold": t, "corpus": c, "docs_scored": int(s.sum()),
                             "flagged_pct": pt, "ci_lo": lo, "ci_hi": hi, "excess_over_2019_pct": pt - fpr})
        det = pd.DataFrame(rows)
        det.round(4).to_csv(A("binoculars_flagged.csv"), index=False)

        # per-group flag rates: 2019 (false positives) and post-LLM, at the 1% threshold
        groups = list(_groups())
        rows = []
        for c in corpora:
            for gname in groups + ["ANY GROUP"]:
                sel = np.array([bool(d["group_paragraphs"]) and (gname == "ANY GROUP" or
                                any(gname in h for h, _ in d["group_paragraphs"])) for d in docs])
                s = sel & (corpus == c) & np.isfinite(score)
                if s.sum() == 0:
                    continue
                pt, lo, hi, _ = rate_ci(hosts[s], flagged[1][s], np.ones(s.sum()), B, seed)
                rows.append({"corpus": c, "group": gname, "docs": int(s.sum()), "flagged_pct": pt, "ci_lo": lo, "ci_hi": hi})
        gp_rate = pd.DataFrame(rows)
        gp_rate.round(3).to_csv(A("binoculars_by_group.csv"), index=False)

        # do the two measures agree? marker share among flagged vs unflagged documents
        rows = []
        for c in corpora:
            s = base & (corpus == c) & np.isfinite(score)
            for lab, m in (("flagged", flagged[1]), ("not flagged", ~flagged[1])):
                ss = s & m
                if ss.sum():
                    pt, lo, hi, _ = rate_ci(hosts[ss], any_m[ss], np.ones(ss.sum()), B, seed)
                    rows.append({"corpus": c, "binoculars": lab, "docs": int(ss.sum()), "any_marker_pct": pt,
                                 "ci_lo": lo, "ci_hi": hi})
        agree = pd.DataFrame(rows)
        agree.round(3).to_csv(A("markers_vs_binoculars.csv"), index=False)

    # 3. tone of flagged vs unflagged text (documents, and group paragraphs inside them)
    lex = cs.load_lexicons()
    rob_path = os.path.join(args.out, "roberta.npy")
    rob = np.load(rob_path) if os.path.exists(rob_path) else None
    names = list(lex) + (["RoBERTa"] if rob is not None else [])
    doc_pn = np.zeros((len(docs), len(names), 2))
    par_pn = np.zeros((len(docs), len(lex), 2))
    for i, d in enumerate(docs):
        toks = cs.tokens(" ".join(d["text"].split()[:WINDOW]))
        ptoks = cs.tokens(" ".join(p for _, p in d["group_paragraphs"]))
        for li, (ln, words) in enumerate(lex.items()):
            h = [words[w] for w in toks if w in words]
            doc_pn[i, li] = (h.count("pos"), h.count("neg"))
            h = [words[w] for w in ptoks if w in words]
            par_pn[i, li] = (h.count("pos"), h.count("neg"))
        if rob is not None:
            doc_pn[i, -1] = (rob[i, 2], rob[i, 0])
    rows = []
    split_by = {"all documents": None}
    if flagged:
        split_by = {"Binoculars flagged (1% FPR)": flagged[1], "Binoculars not flagged": ~flagged[1]}
    split_by.update({"has marker word": any_m, "no marker word": ~any_m})
    for c in corpora:
        for lab, m in split_by.items():
            s = base & (corpus == c) & (np.ones(len(docs), bool) if m is None else m)
            if s.sum() == 0:
                continue
            for unit, arr, nm in (("document", doc_pn, names), ("group paragraphs", par_pn, list(lex))):
                for li, ln in enumerate(nm):
                    pt, lo, hi, _ = rate_ci(hosts[s], arr[s, li, 0], arr[s, li].sum(-1), B, seed)
                    rows.append({"corpus": c, "subset": lab, "unit": unit, "measure": ln, "docs": int(s.sum()),
                                 "docs_with_group_paragraphs": int(sum(bool(docs[i]["group_paragraphs"]) for i in np.where(s)[0])),
                                 "positivity": pt, "ci_lo": lo, "ci_hi": hi})
    tone = pd.DataFrame(rows)
    tone.round(3).to_csv(A("tone.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures(mk, det, score, corpus, sample, corpora, tone, A, args.dpi, plt, cs)

    with open(A("summary.md"), "w", encoding="utf-8") as f:
        f.write("# Machine-generated text in post-LLM C4\n\n")
        f.write(f"Documents (>= {MIN_WORDS} words): " + ", ".join(
            f"{c}: {int((base & (corpus == c)).sum())}" for c in corpora) + "\n\n")
        f.write("## Lower bound on the LLM-processed share (marker words)\n\n" + cs.md_table(lb.round(2)) + "\n")
        f.write("## Marker / control document rates\n\n" + cs.md_table(mk[mk.word.str.startswith("ANY")].round(2)) + "\n")
        if len(det):
            f.write("## Binoculars: share flagged (threshold set on 2019 C4)\n\n" + cs.md_table(det.round(2)) + "\n")
            f.write("## Do the two measures agree?\n\n" + cs.md_table(agree.round(2)) + "\n")
            f.write("## Flag rate by group (1% threshold; 2019 rows are false positives)\n\n" + cs.md_table(gp_rate.round(2)) + "\n")
        f.write("## Tone\n\n" + cs.md_table(tone.round(2)) + "\n")
    print(lb.round(2).to_string(index=False))
    if len(det):
        print(det.round(2).to_string(index=False))
    print(f"wrote {A('')}")


def figures(mk, det, score, corpus, sample, corpora, tone, A, dpi, plt, cs):
    colors = dict(zip(corpora, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    from matplotlib.lines import Line2D
    legend = [Line2D([], [], color=colors[c], marker="o", linestyle="none", label=c) for c in corpora]

    # marker words and control words: document rate per word, every corpus (linear, so 0% stays visible)
    off = np.linspace(0.25, -0.25, len(corpora))
    panels = [("LLM marker words", ["ANY MARKER"] + MARKERS), ("Control words", ["ANY CONTROL"] + CONTROLS)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.28 * len(panels[0][1]) + 2), facecolor=cs.SURFACE,
                             gridspec_kw={"height_ratios": [1]})
    for ax, (title, words) in zip(axes, panels):
        cs.style(ax)
        y = np.arange(len(words))[::-1]
        for c, dy in zip(corpora, off):
            d = mk[mk.corpus == c].set_index("word").reindex(words)
            v = d["doc_rate_pct"].to_numpy()
            ax.errorbar(v, y + dy, xerr=[v - d["ci_lo"], d["ci_hi"] - v], fmt="o", color=colors[c], ms=4.5,
                        mec=cs.SURFACE, mew=1, ecolor=colors[c], elinewidth=1, capsize=1.5, zorder=3)
        ax.set_yticks(y, words, color=cs.INK, fontsize=8)
        ax.set_ylim(-0.7, len(words) - 0.3)
        ax.set_xlim(left=0)
        ax.set_title(title, color=cs.INK, loc="left", fontsize=10)
        ax.set_xlabel(f"Documents with the word in their first {WINDOW} words (%)", color=cs.INK2)
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.01, 0), ncol=len(corpora), frameon=False,
               labelcolor=cs.INK2, fontsize=9)
    fig.suptitle("LLM marker words vs control words, by corpus (95% CIs, domain bootstrap)", color=cs.INK,
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(A("fig_marker_words.png"), dpi=dpi, facecolor=cs.SURFACE)
    plt.close(fig)

    if len(det):
        # Binoculars score distributions (ECDF) with the 2019-calibrated thresholds
        fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=cs.SURFACE)
        cs.style(ax)
        ax.grid(axis="y", color=cs.GRID, linewidth=0.8)
        for c in corpora:
            s = np.sort(score[(corpus == c) & (sample != "group") & np.isfinite(score)])
            ax.plot(s, np.arange(1, len(s) + 1) / len(s) * 100, color=colors[c], linewidth=2, label=c)
        for fpr in (1, 5):
            t = det[det.fpr_on_2019_pct == fpr].threshold.iloc[0]
            ax.axvline(t, color=cs.INK2, linewidth=1, linestyle=(0, (3, 3)))
            ax.text(t, 97, f"{fpr}% of 2019 below ", color=cs.INK2, fontsize=8, rotation=90, ha="right", va="top")
        ax.set_xlabel("Binoculars score (lower = more machine-like)", color=cs.INK2)
        ax.set_ylabel("Documents at or below this score (%)", color=cs.INK2)
        ax.set_ylim(0, 108)
        lo = np.nanpercentile(score, 0.5)
        ax.set_xlim(lo, np.nanpercentile(score, 99.5))
        ax.legend(loc="lower right", frameon=False, labelcolor=cs.INK2, fontsize=9)
        ax.set_title("Binoculars scores by corpus; curves left of 2019 = more machine-like text", color=cs.INK,
                     loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(A("fig_binoculars_ecdf.png"), dpi=dpi, facecolor=cs.SURFACE)
        plt.close(fig)

    # tone: flagged vs not flagged, per corpus and measure (documents)
    subs = [s for s in tone.subset.unique()]
    t = tone[tone.unit == "document"]
    measures = list(t.measure.unique())
    fig, axes = plt.subplots(1, len(measures), figsize=(4 * len(measures), 0.5 * len(subs) * len(corpora) / 2 + 2.2),
                             sharey=True, facecolor=cs.SURFACE)
    yy = np.arange(len(subs))[::-1]
    for ax, m in zip(np.atleast_1d(axes), measures):
        cs.style(ax)
        for c, dy in zip(corpora, off):
            d = t[(t.corpus == c) & (t.measure == m)].set_index("subset").reindex(subs)
            v = d["positivity"].to_numpy()
            ax.errorbar(v, yy + dy, xerr=[v - d["ci_lo"], d["ci_hi"] - v], fmt="o", color=colors[c], ms=5.5,
                        mec=cs.SURFACE, mew=1.1, ecolor=colors[c], elinewidth=1.2, capsize=2, zorder=3)
        ax.set_title(m, color=cs.INK, loc="left", fontsize=10)
        ax.set_xlabel("Positivity (%)", color=cs.INK2)
    np.atleast_1d(axes)[0].set_yticks(yy, subs, color=cs.INK)
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.01, 0), ncol=len(corpora), frameon=False, labelcolor=cs.INK2, fontsize=9)
    fig.suptitle("Tone of machine-like vs other documents (95% CIs, domain bootstrap)", color=cs.INK, x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(A("fig_tone.png"), dpi=dpi, facecolor=cs.SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("docs")
    p.add_argument("--c4_files", required=True, help="glob of post-LLM c4_en_* and c4_blocklist_* files")
    p.add_argument("--shard_2019", type=int, default=0)
    p.add_argument("--n_2019", type=int, default=5000, help="random 2019 documents (FPR calibration, marker rates)")
    p.add_argument("--per_group", type=int, default=200, help="extra 2019 documents per group (per-group FPR)")
    p.add_argument("--keep_2019", action="store_true")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("detect")
    p.add_argument("--observer", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--performer", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_per_corpus", type=int, default=None, help="score a random subset of each post-LLM crawl")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("classify")
    p.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("analyze")
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=200)
    for p in sub.choices.values():
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"docs": cmd_docs, "detect": cmd_detect, "classify": cmd_classify, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
