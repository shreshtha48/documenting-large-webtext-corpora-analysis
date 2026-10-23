"""Arm B, step 2: sentiment of C4 paragraphs that mention an ethnicity (Dodge et al. 2021, §4.3 / A.7).

A.7: paragraphs are "tokenized ... by whitespace, removed all punctuation"; for each lexicon "we counted the number
of positive and negative word occurrences in paragraphs containing the ethnicity word, then found the proportion of
these occurrences that had positive sentiment." A paragraph mentioning two groups counts for both.
Tokens are lowercased before lookup (the lexicons are lowercase; the paper does not say) - our assumption.

The paper's corpus study is "Jewish" vs "Arab"; those numbers (positivity, gap, NYT / Al Jazeera, scale) are the
reproduction. If the paragraphs were scanned with --table7_groups (our extension), every group also gets a
positivity per lexicon, and the SocialSent positivity is compared with the paper's Table 7 (model) positivity.

CIs: percentile bootstrap resampling web domains (URL host), B=1000, because paragraphs from one site are not
independent. The NYT / Al Jazeera subsets use the paper's URL prefixes.

Usage:
    python scripts/count_sentiment.py --paragraphs results/B1_c4_12_shards --out_dir results/B1_c4_12_shards/analysis

Writes positivity.csv, gaps.csv, domains.csv, scale.csv, top_tokens.csv, summary.md, fig_positivity.png and
fig_gap.png (Jewish vs Arab, as in the paper); robustness.csv and fig_robustness.png (the gap with non-ethnic
"Arab" text such as "United Arab Emirates" removed, without noisy SocialSent words, without religious words);
with more groups also fig_positivity_all_groups.png and fig_corpus_vs_table7.png.
"""
import argparse, csv, glob, gzip, json, os, re
from collections import Counter, defaultdict
from urllib.parse import urlparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LEX_DIR = os.path.join(os.path.dirname(HERE), "data", "lexicons")
LEXICONS = {"SocialSent": "socialsent_250.tsv", "VADER": "vader.tsv", "UnQover": "unqover.tsv"}
PAIR = ["Jewish", "Arab"]   # the paper's comparison
DOMAINS = {"New York Times": "https://www.nytimes.com", "Al Jazeera": "https://www.aljazeera.com"}

# Dodge et al. 2021, §4.3 and A.7 (full C4.EN)
PAPER = {
    "positivity": {("SocialSent", "Jewish"): 73.2, ("SocialSent", "Arab"): 65.7},
    "gap": {"SocialSent": 7.5, "VADER": 2.0, "UnQover": 1.4},
    "sentiment_tokens": {"Jewish": 3.4e6, "Arab": 1.2e6},
    "words_in_paragraphs": {"Jewish": 249.8e6, "Arab": 134.8e6},
    "domain_positivity": {("New York Times", "Jewish"): 74.0, ("New York Times", "Arab"): 69.5,
                          ("Al Jazeera", "Jewish"): 42.5, ("Al Jazeera", "Arab"): 42.8},
}
# Dodge et al. 2021, Table 7 (UnifiedQA positivity), for the extension's corpus-vs-model comparison
TABLE7 = {"Jewish": 67.1, "Asian": 60.6, "Caucasian": 60.5, "European": 60.5, "White": 56.5,
          "Alaskan": 55.9, "Hispanic": 50.8, "Native American": 50.6, "South-American": 44.4,
          "African-American": 44.3, "Latino": 43.1, "Middle-Eastern": 42.6, "Black": 39.3,
          "Arab": 37.0, "African": 36.6}
PUNCT = re.compile(r"[^\w\s]")

# Robustness variants (audit 2026-10-23, C1 / C3). "as in paper" is the reproduction; each other variant changes
# one thing, to show how much of the Jewish-Arab gap comes from non-ethnic text or from noisy lexicon words.
ARAB_GEOPOLITICAL = ["United Arab Emirates", "Arab Emirates", "Arab League", "Arab Spring", "Arab world",
                     "Arab countries", "Arab states", "Arab nations"]
# SocialSent top-250 words that are Reddit artefacts rather than sentiment ("p" is ":p"; in C4 it is "p." page refs)
SS_NOISE = {"man", "mr", "series", "due", "saying", "calling", "thats", "damnit"}
RELIGION = {"god", "gods", "holy", "lord", "holiday", "holidays", "celebrate", "celebration", "christmas", "faith",
            "church", "prayer", "pray", "blessed", "bless", "blessing", "heaven", "jesus", "christ", "spiritual",
            "worship", "sacred", "divine"}
VARIANTS = {
    "as in paper": {},
    "Arab: no 'United Arab Emirates' paragraphs": {"drop_par": ["United Arab Emirates"]},
    "Arab: no geopolitical 'Arab' phrases": {"drop_par": ARAB_GEOPOLITICAL},
    "no 1-letter tokens or SocialSent noise words": {"drop_tok": SS_NOISE, "drop_short": True},
    "no religious words": {"drop_tok": RELIGION},
}

# reference palette (dataviz skill, validated): Jewish = blue, Arab = orange; paper = hollow ink diamond
C = {"Jewish": "#2a78d6", "Arab": "#eb6834"}
C_LEX = {"SocialSent": "#2a78d6", "VADER": "#1baf7a", "UnQover": "#6250d6"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def load_lexicons():
    lex = {}
    for name, fn in LEXICONS.items():
        with open(os.path.join(LEX_DIR, fn), encoding="utf-8") as f:
            lex[name] = {r["word"]: r["polarity"] for r in csv.DictReader(f, delimiter="\t")}
    return lex


def tokens(text):
    return [t for t in (PUNCT.sub("", w).lower() for w in text.split()) if t]


def scanned_targets(paragraph_dir, files):
    """Target names in scan order: from shards.json if present, else from the paragraphs themselves."""
    path = os.path.join(paragraph_dir, "shards.json")
    if os.path.exists(path):
        t = json.load(open(path))["targets"]
        names = list(t) if isinstance(t, dict) else list(t)
    else:
        names = []
        for p in files:
            for line in gzip.open(p, "rt", encoding="utf-8"):
                names += [x for x in json.loads(line)["targets"] if x not in names]
    return [x for x in PAIR if x in names] + [x for x in names if x not in PAIR]


def count(paragraph_files, lex, targets):
    """Per (domain, target, lexicon): positive and negative token counts; plus scale and token counters.

    Returns per_domain for the paper's setting and `robust`: {variant: per_domain} for every VARIANTS entry.
    """
    T, L = len(targets), len(LEXICONS)
    robust = {v: defaultdict(lambda: np.zeros((T, L, 2))) for v in VARIANTS}
    kept = Counter()                       # (variant, target) -> paragraphs counted
    per_domain = robust["as in paper"]
    subset = defaultdict(lambda: np.zeros((T, L, 2)))
    top = {(t, l): Counter() for t in targets for l in LEXICONS}
    scale = Counter()
    shards = set()
    for path in paragraph_files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                shards.add(r["shard"])
                toks = tokens(r["text"])
                host = urlparse(r["url"]).netloc
                hits = {}
                for lname, words in lex.items():
                    hits[lname] = [(w, words[w]) for w in toks if w in words]
                for t in r["targets"]:
                    if t not in targets:
                        continue
                    ti = targets.index(t)
                    scale[f"{t} paragraphs"] += 1
                    scale[f"{t} words"] += len(toks)
                    for li, lname in enumerate(lex):
                        pos = sum(1 for _, p in hits[lname] if p == "pos")
                        neg = len(hits[lname]) - pos
                        top[(t, lname)].update(w for w, _ in hits[lname])
                        for dname, prefix in DOMAINS.items():
                            if r["url"].startswith(prefix):
                                subset[dname][ti, li] += (pos, neg)
                    for v, spec in VARIANTS.items():
                        # paragraph filters apply to "Arab" only: they remove non-ethnic senses of that word
                        if t == "Arab" and any(x in r["text"] for x in spec.get("drop_par", ())):
                            continue
                        kept[(v, t)] += 1
                        drop, short = spec.get("drop_tok", set()), spec.get("drop_short", False)
                        for li, lname in enumerate(lex):
                            h = [p for w, p in hits[lname] if w not in drop and not (short and len(w) == 1)]
                            pos = sum(1 for p in h if p == "pos")
                            robust[v][host][ti, li] += (pos, len(h) - pos)
    return per_domain, robust, kept, subset, top, scale, sorted(shards)


def boot(per_domain, B, seed, batch=50):
    """Domain bootstrap. Replicates are drawn in batches so a full-corpus run (~10^5 domains) fits in free Colab RAM."""
    hosts = list(per_domain)
    X = np.stack([per_domain[h] for h in hosts])                  # domains x targets x lexicons x 2
    tot = X.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = 100 * tot[..., 0] / tot.sum(-1)
    rng = np.random.default_rng(seed)
    p = np.full(len(hosts), 1 / len(hosts))
    pb = []
    for b0 in range(0, B, batch):
        w = rng.multinomial(len(hosts), p, size=min(batch, B - b0))  # batch x domains
        S = np.einsum("bd,dtlk->btlk", w, X)
        with np.errstate(invalid="ignore", divide="ignore"):
            pb.append(100 * S[..., 0] / S.sum(-1))                  # batch x targets x lexicons
    return tot, point, np.concatenate(pb), len(hosts)


def robustness(robust, kept, targets, B, seed):
    """Jewish-Arab gap per lexicon under each variant, with domain-bootstrap CIs."""
    a, b = targets.index("Jewish"), targets.index("Arab")
    rows = []
    for v in VARIANTS:
        _, point, pb, _ = boot(robust[v], B, seed)
        for li, l in enumerate(LEXICONS):
            g = pb[:, a, li] - pb[:, b, li]
            lo, hi = np.nanpercentile(g, [2.5, 97.5])
            rows.append({"variant": v, "lexicon": l, "Jewish": point[a, li], "Arab": point[b, li],
                         "gap": point[a, li] - point[b, li], "ci_lo": lo, "ci_hi": hi,
                         "Jewish_paragraphs": kept[(v, "Jewish")], "Arab_paragraphs": kept[(v, "Arab")],
                         "paper_gap": PAPER["gap"][l]})
    return pd.DataFrame(rows)


def fig_robustness(rob, out, dpi, title, plt):
    """Gap per lexicon (panels) under each variant (rows), 95% CIs; the paper's gap as a dashed line."""
    vs = list(VARIANTS)[::-1]
    y = np.arange(len(vs))
    fig, axes = plt.subplots(1, len(LEXICONS), figsize=(4.4 * len(LEXICONS), 0.55 * len(vs) + 1.9),
                             sharey=True, facecolor=SURFACE)
    lim = np.nanmax(np.abs(rob[["ci_lo", "ci_hi"]].to_numpy())) + 1.5
    for ax, l in zip(axes, LEXICONS):
        style(ax)
        d = rob[rob["lexicon"] == l].set_index("variant").reindex(vs)
        v = d["gap"].to_numpy()
        ax.errorbar(v, y, xerr=[v - d["ci_lo"].to_numpy(), d["ci_hi"].to_numpy() - v], fmt="o", color=C_LEX[l],
                    ms=7, mec=SURFACE, mew=1.3, ecolor=C_LEX[l], elinewidth=1.4, capsize=3, zorder=3)
        for yi, vi, hi in zip(y, v, d["ci_hi"].to_numpy()):
            ax.text(hi + 0.4, yi, f"{vi:+.1f}", va="center", fontsize=8, color=INK2)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.axvline(PAPER["gap"][l], color=INK, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xlim(-lim, lim)
        ax.set_title(f"{l} (dashed = paper, {PAPER['gap'][l]:+.1f})", color=INK, loc="left", fontsize=10)
        ax.set_xlabel("Jewish minus Arab positivity (points)", color=INK2)
    axes[0].set_yticks(y, vs, color=INK, fontsize=8.5)
    fig.suptitle(f"{title}\nEach row changes one thing from the paper's setting; 95% CIs from a domain bootstrap",
                 color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_robustness.png"), dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_pair(pos_df, gap_df, out, dpi, title, plt):
    """The paper's comparison: Jewish vs Arab positivity per lexicon, and the gap."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    lexs = list(LEXICONS)[::-1]
    y = np.arange(len(lexs))
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    for t, dy in (("Jewish", 0.19), ("Arab", -0.19)):
        d = pos_df[pos_df["target"] == t].set_index("lexicon").reindex(lexs)
        v = d["positivity"].values
        ax.barh(y + dy, v, height=0.36, color=C[t], edgecolor=SURFACE, linewidth=1, zorder=2)
        ax.errorbar(v, y + dy, xerr=[v - d["ci_lo"].values, d["ci_hi"].values - v], fmt="none", ecolor=INK2,
                    elinewidth=1.2, capsize=3, zorder=3)
        for yi, vi, hi in zip(y + dy, v, d["ci_hi"].values):
            ax.text(hi + 0.8, yi, f"{vi:.1f}", va="center", fontsize=8, color=INK2)
        for li, l in enumerate(lexs):
            pv = PAPER["positivity"].get((l, t))
            if pv is not None:
                ax.scatter([pv], [li + dy], s=46, marker="D", facecolor=SURFACE, edgecolor=INK, linewidth=1.4, zorder=4)
    ax.set_yticks(y, lexs, color=INK)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Positivity (%) = positive / (positive + negative) lexicon tokens in matching paragraphs", color=INK2)
    ax.legend(handles=[Patch(color=C["Jewish"], label="Jewish"), Patch(color=C["Arab"], label="Arab"),
                       Line2D([], [], color=INK2, marker="|", markersize=8, label="95% CI (domain bootstrap)"),
                       Line2D([], [], marker="D", color=INK, markerfacecolor=SURFACE, linestyle="none",
                              label="Dodge et al. 2021 (SocialSent only)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=INK2, fontsize=8.5)
    ax.set_title(title, color=INK, loc="left", fontsize=12, pad=44)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_positivity.png"), dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    style(ax)
    d = gap_df.set_index("lexicon").reindex(lexs)
    v = d["gap"].values
    ax.barh(y, v, height=0.55, color=["#2a78d6" if x >= 0 else "#e34948" for x in v], edgecolor=SURFACE, zorder=2)
    ax.errorbar(v, y, xerr=[v - d["ci_lo"].values, d["ci_hi"].values - v], fmt="none", ecolor=INK2, elinewidth=1.2,
                capsize=3, zorder=3)
    ax.scatter(d["paper_gap"].values, y, s=46, marker="D", facecolor=SURFACE, edgecolor=INK, linewidth=1.4, zorder=4)
    ax.axvline(0, color=INK2, linewidth=1)
    for yi, vi, hi in zip(y, v, d["ci_hi"].values):
        ax.text(max(hi, vi) + 0.3, yi, f"{vi:+.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, lexs, color=INK)
    ax.set_xlabel("Positivity gap, Jewish minus Arab (percentage points)", color=INK2)
    ax.legend(handles=[Patch(color="#2a78d6", label="ours: Jewish more positive"),
                       Patch(color="#e34948", label="ours: Arab more positive"),
                       Line2D([], [], color=INK2, marker="|", markersize=8, label="95% CI"),
                       Line2D([], [], marker="D", color=INK, markerfacecolor=SURFACE, linestyle="none",
                              label="Dodge et al. 2021")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=INK2, fontsize=8.5)
    ax.set_title("Jewish-Arab positivity gap by lexicon", color=INK, loc="left", fontsize=12, pad=44)
    lim = np.nanmax(np.abs(np.r_[d["ci_lo"], d["ci_hi"], d["paper_gap"]])) + 2
    ax.set_xlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_gap.png"), dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_all_groups(pos_df, out, dpi, title, plt):
    """Extension: positivity of every scanned group, one panel per lexicon, sorted by SocialSent."""
    order = (pos_df[pos_df["lexicon"] == "SocialSent"].sort_values("positivity")["target"].tolist())
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, len(LEXICONS), figsize=(4.6 * len(LEXICONS), 0.42 * len(order) + 1.8),
                             sharey=True, facecolor=SURFACE)
    for ax, l in zip(axes, LEXICONS):
        style(ax)
        d = pos_df[pos_df["lexicon"] == l].set_index("target").reindex(order)
        v = d["positivity"].values
        ax.errorbar(v, y, xerr=[v - d["ci_lo"].values, d["ci_hi"].values - v], fmt="o", color=C_LEX[l], ms=6,
                    mec=SURFACE, mew=1.2, ecolor=C_LEX[l], elinewidth=1.3, capsize=2.5, zorder=3)
        pooled = 100 * d["pos"].sum() / d["sentiment_tokens"].sum()
        ax.axvline(pooled, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(f"{l} (dashed = all groups pooled)", color=INK, loc="left", fontsize=10)
        ax.set_xlabel("Positivity (%)", color=INK2)
    axes[0].set_yticks(y, [f"{t}  (n={int(n):,})" for t, n in
                           zip(order, pos_df[pos_df["lexicon"] == "SocialSent"].set_index("target").reindex(order)["sentiment_tokens"])],
                       color=INK)
    fig.suptitle(f"{title}\nn = SocialSent sentiment tokens; 95% CIs from a domain bootstrap. Black, White, African "
                 "and similar words also match non-ethnic uses", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_positivity_all_groups.png"), dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_corpus_vs_table7(pos_df, out, dpi, plt):
    """Extension: does the corpus explain the model? SocialSent corpus positivity vs the paper's Table 7."""
    d = pos_df[(pos_df["lexicon"] == "SocialSent") & pos_df["target"].isin(TABLE7)].copy()
    d["table7"] = d["target"].map(TABLE7)
    rho = d["positivity"].rank().corr(d["table7"].rank())
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    v = d["positivity"].values
    ax.errorbar(d["table7"], v, yerr=[v - d["ci_lo"].values, d["ci_hi"].values - v], fmt="o", color=C_LEX["SocialSent"],
                ms=7, mec=SURFACE, mew=1.5, ecolor=C_LEX["SocialSent"], elinewidth=1.3, capsize=3, zorder=3)
    for t, x, yv in zip(d["target"], d["table7"], v):
        key = t in PAIR
        ax.annotate(t, (x, yv), xytext=(6, 3), textcoords="offset points", fontsize=8.5 if key else 7.5,
                    color=INK if key else INK2, fontweight="bold" if key else "normal")
    ax.set_xlabel("Model: Dodge et al. 2021 Table 7 positivity (UnifiedQA, %)", color=INK2)
    ax.set_ylabel("Corpus: SocialSent positivity in C4 paragraphs (%), 95% CI", color=INK2)
    ax.set_title(f"Extension: corpus sentiment vs model bias, {len(d)} groups (Spearman {rho:.2f})\n"
                 "Word-sense noise (Black, White, African, ...) affects the corpus axis", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_corpus_vs_table7.png"), dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paragraphs", required=True, help="folder with paragraphs_*.jsonl.gz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.paragraphs, "paragraphs_*.jsonl.gz")))
    if not files:
        raise SystemExit(f"no paragraphs_*.jsonl.gz in {args.paragraphs}")
    targets = scanned_targets(args.paragraphs, files)
    lex = load_lexicons()
    per_domain, robust, kept, subset, top, scale, shards = count(files, lex, targets)
    n = len(shards)

    tot, point, pb, n_hosts = boot(per_domain, args.B, args.seed)
    rows = []
    for ti, t in enumerate(targets):
        for li, l in enumerate(LEXICONS):
            lo, hi = np.nanpercentile(pb[:, ti, li], [2.5, 97.5])
            rows.append({"target": t, "lexicon": l, "pos": int(tot[ti, li, 0]), "neg": int(tot[ti, li, 1]),
                         "sentiment_tokens": int(tot[ti, li].sum()), "positivity": point[ti, li],
                         "ci_lo": lo, "ci_hi": hi, "paper": PAPER["positivity"].get((l, t))})
    pos_df = pd.DataFrame(rows)

    has_pair = all(t in targets for t in PAIR)
    gap_df = pd.DataFrame()
    if has_pair:
        a, b = targets.index("Jewish"), targets.index("Arab")
        gaps = []
        for li, l in enumerate(LEXICONS):
            g = pb[:, a, li] - pb[:, b, li]
            lo, hi = np.nanpercentile(g, [2.5, 97.5])
            gaps.append({"lexicon": l, "gap": point[a, li] - point[b, li], "ci_lo": lo, "ci_hi": hi,
                         "share_of_bootstrap_gap_>0": float(np.mean(g > 0)), "paper_gap": PAPER["gap"][l]})
        gap_df = pd.DataFrame(gaps)
        rob_df = robustness(robust, kept, targets, args.B, args.seed)

    li = list(LEXICONS).index("SocialSent")
    drows = []
    for dname in DOMAINS:
        X = subset.get(dname, np.zeros((len(targets), len(LEXICONS), 2)))
        for ti, t in enumerate(targets):
            k = X[ti, li]
            drows.append({"domain": dname, "target": t, "lexicon": "SocialSent", "sentiment_tokens": int(k.sum()),
                          "positivity": 100 * k[0] / k.sum() if k.sum() else np.nan,
                          "paper": PAPER["domain_positivity"].get((dname, t))})
    dom_df = pd.DataFrame(drows)

    srows = []
    for t in targets:
        st = int(pos_df[(pos_df.target == t) & (pos_df.lexicon == "SocialSent")]["sentiment_tokens"].iloc[0])
        srows.append({"target": t, "paragraphs": scale[f"{t} paragraphs"], "words_in_paragraphs": scale[f"{t} words"],
                      "paper_words_scaled_to_sample": PAPER["words_in_paragraphs"].get(t, np.nan) * n / 1024,
                      "socialsent_tokens": st,
                      "paper_socialsent_tokens_scaled_to_sample": PAPER["sentiment_tokens"].get(t, np.nan) * n / 1024})
    scale_df = pd.DataFrame(srows)

    trows = [{"target": t, "lexicon": l, "token": w, "count": c, "polarity": lex[l][w]}
             for (t, l), cnt in top.items() for w, c in cnt.most_common(25)]

    pos_df.round(2).to_csv(os.path.join(args.out_dir, "positivity.csv"), index=False)
    if has_pair:
        gap_df.round(3).to_csv(os.path.join(args.out_dir, "gaps.csv"), index=False)
        rob_df.round(3).to_csv(os.path.join(args.out_dir, "robustness.csv"), index=False)
    dom_df.round(2).to_csv(os.path.join(args.out_dir, "domains.csv"), index=False)
    scale_df.round(0).to_csv(os.path.join(args.out_dir, "scale.csv"), index=False)
    pd.DataFrame(trows).to_csv(os.path.join(args.out_dir, "top_tokens.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rho = None
    if has_pair:
        fig_pair(pos_df[pos_df["target"].isin(PAIR)], gap_df, args.out_dir, args.dpi,
                 f"C4 paragraphs mentioning Jewish / Arab ({n} of 1,024 shards)", plt)
        fig_robustness(rob_df, args.out_dir, args.dpi, f"Robustness of the Jewish-Arab gap ({n} of 1,024 shards)", plt)
    if len(targets) > 2:
        fig_all_groups(pos_df, args.out_dir, args.dpi, f"Extension: C4 paragraphs per group ({n} of 1,024 shards)", plt)
        rho = fig_corpus_vs_table7(pos_df, args.out_dir, args.dpi, plt)

    sections = [("Positivity", pos_df)]
    if has_pair:
        sections.append(("Gap (Jewish - Arab)", gap_df))
        sections.append(("Robustness of the gap (each row changes one thing)", rob_df))
    sections += [("Domains (SocialSent)", dom_df), ("Scale vs the paper (paper numbers scaled by shards/1024)", scale_df)]
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# Arm B: C4 paragraph sentiment ({n} shards = {n}/1024 of C4.EN, {n_hosts} domains)\n\n")
        if rho is not None:
            f.write(f"Extension: Spearman between SocialSent corpus positivity and Table 7 positivity = {rho:.3f}\n\n")
        for name, d in sections:
            f.write(f"## {name}\n\n" + md_table(d.round(2)) + "\n")
    print(f"{n} shards, {n_hosts} domains, targets: {targets}")
    for name, d in sections:
        print(f"\n{name}\n" + d.round(2).to_string(index=False))
    if rho is not None:
        print(f"\nExtension: Spearman(corpus SocialSent positivity, Table 7) = {rho:.3f}")
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
