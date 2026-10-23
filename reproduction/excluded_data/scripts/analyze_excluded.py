"""Dodge et al. 2021 §5: what the blocklist excluded. Reads the output of scan_noblocklist.py.

§5.2 Figure 5 and Table 6: PMI between each of the 22 identity regexes and a document being excluded,
    PMI = ln[ P(term, excluded) / (P(term) P(excluded)) ] over en.noblocklist documents (natural log; the paper's
    axis runs 0-2.5, which fits ln). The paper prints no numbers, so its bars are read off the figure (±0.05) and
    compared by rank.
§5.2 manual check: 50 random excluded documents mentioning "lesbian" and 50 mentioning "gay", as a CSV to label.
    Paper: 22% (lesbian) and 36% (gay) are non-offensive or non-sexual.
§5.3 dialects: removal rate per dialect (paper: AAE 42%, Hispanic 32%, WAE 6.2%, other 7.2%) and the dialect mix
    of the kept documents, i.e. C4.EN (paper: WAE 97.8%, AAE 0.07%, Hispanic 0.09%).
§5.1 / A.6 Figure 6: 100,000 sampled excluded documents -> TF-IDF -> k-means (k=50) -> 5,000 shown in a t-SNE of a
    50-d PCA (truncated SVD) projection, legend = top 5 terms per cluster. The paper labelled 16 of 50 clusters
    (31% of documents) as "largely sexual" by hand. We label automatically (a cluster is sexual if at least 2 of
    its top 10 terms are blocklist words), repeat k-means over several seeds, and leave a column for hand labels.

Uncertainty: rates and PMIs get 95% CIs from a bootstrap over web domains (Poisson weights, B=200 by default),
because documents from one site are not independent.

TF-IDF settings, seeds and the labelling rule are not given by the paper: ours are max_features=50,000,
English stop words, min_df=5, sublinear tf; documents truncated to 20,000 characters.

Usage:
    python scripts/analyze_excluded.py --scan results/S1_noblocklist_12_shards --out_dir results/S1_noblocklist_12_shards/analysis
"""
import argparse, glob, gzip, json, os, random, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BADWORDS = os.path.join(ROOT, "data", "ldnoobw_en_25e679f.txt")

# Fig. 5 read off the paper's figure (no numbers are printed); label -> approximate PMI
PAPER_FIG5 = {"european": 0.84, "white": 1.12, "straight": 1.27, "christian": 1.37, "black": 1.43,
              "african american": 1.44, "jewish": 1.47, "muslim": 1.50, "man": 1.65, "caucasian": 1.68,
              "asian": 1.72, "women": 1.73, "trans": 1.75, "female": 1.82, "non-binary": 1.84, "male": 1.90,
              "latina / latino": 1.92, "bisexual": 2.24, "homosexual": 2.30, "heterosexual": 2.34, "gay": 2.34,
              "lesbian": 2.44}
CATEGORY = {"sexual orientation": ["lesbian", "gay", "homosexual", "bisexual", "heterosexual", "straight"],
            "gender identity": ["non-binary", "trans", "women", "man", "female", "male"],
            "race / ethnicity": ["black", "african american", "latina / latino", "asian", "white", "caucasian",
                                 "european"],
            "religion": ["jewish", "muslim", "christian"]}
PAPER_DIALECT = {"removal_%": {"AAE": 42.0, "Hispanic": 32.0, "WAE": 6.2, "Other": 7.2},
                 "share_of_C4.EN_%": {"WAE": 97.8, "AAE": 0.07, "Hispanic": 0.09}}
PAPER_CLUSTERS = {"k": 50, "sexual_clusters": 16, "sexual_share_%": 31.0}

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
C_CAT = {"sexual orientation": "#2a78d6", "gender identity": "#eb6834", "race / ethnicity": "#1baf7a",
         "religion": "#6250d6"}
C_DIAL = {"AAE": "#e34948", "Hispanic": "#eb6834", "Other": "#6250d6", "WAE": "#2a78d6", "unassigned": MUTED}


# ---------- loading and bootstrap ----------

def load_counts(scan_dir):
    files = sorted(glob.glob(os.path.join(scan_dir, "stats_*.json")))
    if not files:
        raise SystemExit(f"no finished shards (stats_*.json) in {scan_dir}")
    stats = [json.load(open(p)) for p in files]
    frames, columns = [], None
    for s in stats:
        z = np.load(os.path.join(scan_dir, f"counts_{s['shard']}.npz"))
        columns = list(z["columns"])
        frames.append(pd.DataFrame(z["counts"], index=z["hosts"], columns=columns))
    df = pd.concat(frames).groupby(level=0).sum()           # one row per web domain, over all shards
    return df, columns, stats


def bootstrap(X, B, seed, batch=10):
    """Domain bootstrap with Poisson(1) weights: returns B x columns resampled totals (sparse, so it fits free RAM)."""
    from scipy import sparse
    Xs = sparse.csr_matrix(X.astype(np.float64)).T.tocsr()      # columns x hosts
    rng = np.random.default_rng(seed)
    out = []
    for b0 in range(0, B, batch):
        W = rng.poisson(1.0, size=(X.shape[0], min(batch, B - b0))).astype(np.float32)
        out.append(np.asarray(Xs @ W).T)
    return np.concatenate(out)


def ci(a):
    return np.nanpercentile(a, [2.5, 97.5], axis=0)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


# ---------- §5.2: identity PMI (Fig. 5, Table 6) ----------

def identity_pmi(df, boot, columns):
    c = {k: i for i, k in enumerate(columns)}
    n, F = df["docs"].sum(), df["excluded"].sum()
    bn, bF = boot[:, c["docs"]], boot[:, c["excluded"]]
    rows = []
    from scan_noblocklist import IDENTITY
    for label, regex in IDENTITY:
        nt, ft = df[f"{label}|docs"].sum(), df[f"{label}|excluded"].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(ft * n / (nt * F)) if ft else np.nan
            b = np.log(boot[:, c[f"{label}|excluded"]] * bn / (boot[:, c[f"{label}|docs"]] * bF))
        lo, hi = ci(b[np.isfinite(b)]) if np.isfinite(b).sum() > 10 else (np.nan, np.nan)
        rows.append({"term": label, "regex (Table 6)": regex, "docs": int(nt), "excluded": int(ft),
                     "excluded_%": 100 * ft / nt if nt else np.nan, "pmi": pmi, "ci_lo": lo, "ci_hi": hi,
                     "paper_pmi_read_off_fig5": PAPER_FIG5.get(label),
                     "category": next(k for k, v in CATEGORY.items() if label in v)})
    return pd.DataFrame(rows).sort_values("pmi").reset_index(drop=True)


def fig5(t, n_docs, n_shards, out, dpi, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor=SURFACE)
    style(ax)
    ax.barh(y, t["pmi"], height=0.66, color=[C_CAT[c] for c in t["category"]], edgecolor=SURFACE, zorder=2)
    ax.errorbar(t["pmi"], y, xerr=[t["pmi"] - t["ci_lo"], t["ci_hi"] - t["pmi"]], fmt="none", ecolor=INK2,
                elinewidth=1.1, capsize=2.5, zorder=3)
    ax.scatter(t["paper_pmi_read_off_fig5"], y, s=40, marker="D", facecolor=SURFACE, edgecolor=INK, linewidth=1.3,
               zorder=4)
    for yi, r in zip(y, t.to_dict("records")):
        ax.text(max(r["ci_hi"], r["paper_pmi_read_off_fig5"] or 0) + 0.04, yi,
                f"{r['pmi']:.2f}  ({r['excluded_%']:.0f}% of {r['docs']:,} docs excluded)", va="center",
                fontsize=7.5, color=INK2)
    ax.set_yticks(y, t["term"], color=INK)
    ax.set_xlim(0, max(3.6, np.nanmax(t["ci_hi"]) + 1.2))
    ax.set_xlabel("PMI(identity term; excluded by blocklist), natural log", color=INK2)
    rho = t["pmi"].rank().corr(t["paper_pmi_read_off_fig5"].rank())
    ax.legend(handles=[Patch(color=c, label=k) for k, c in C_CAT.items()] +
              [Line2D([], [], marker="D", color=INK, markerfacecolor=SURFACE, linestyle="none",
                      label="Dodge et al. 2021 (read off Fig. 5)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title(f"Figure 5 reproduction: identity mentions vs blocklist exclusion (Spearman with paper {rho:.2f})\n"
                 f"C4 en.noblocklist, {n_shards} shards ({n_docs:,} docs); 95% CI: domain bootstrap",
                 color=INK, loc="left", fontsize=11, pad=48)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_pmi.png"), dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return rho


# ---------- §5.3: dialects ----------

def dialects(df, boot, columns):
    from scan_noblocklist import DIALECTS
    c = {k: i for i, k in enumerate(columns)}
    kept_all = df["docs"].sum() - df["excluded"].sum()
    b_kept_all = boot[:, c["docs"]] - boot[:, c["excluded"]]
    rows = []
    for d in DIALECTS:
        n, e = df[f"dialect:{d}|docs"].sum(), df[f"dialect:{d}|excluded"].sum()
        bn, be = boot[:, c[f"dialect:{d}|docs"]], boot[:, c[f"dialect:{d}|excluded"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            rem_lo, rem_hi = ci(100 * be / bn)
            sh_lo, sh_hi = ci(100 * (bn - be) / b_kept_all)
        rows.append({"dialect": d, "docs_noblocklist": int(n), "excluded": int(e),
                     "removal_%": 100 * e / n if n else np.nan, "removal_ci_lo": rem_lo, "removal_ci_hi": rem_hi,
                     "paper_removal_%": PAPER_DIALECT["removal_%"].get(d),
                     "share_of_kept_%": 100 * (n - e) / kept_all, "share_ci_lo": sh_lo, "share_ci_hi": sh_hi,
                     "paper_share_of_C4.EN_%": PAPER_DIALECT["share_of_C4.EN_%"].get(d)})
    return pd.DataFrame(rows)


def fig_dialects(t, out, dpi, plt):
    from matplotlib.lines import Line2D
    t = t[t["dialect"] != "unassigned"].iloc[::-1]
    y = np.arange(len(t))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=SURFACE)
    for ax, col, lo, hi, paper, xlabel, log in (
            (a1, "removal_%", "removal_ci_lo", "removal_ci_hi", "paper_removal_%",
             "Documents removed by the blocklist (%)", False),
            (a2, "share_of_kept_%", "share_ci_lo", "share_ci_hi", "paper_share_of_C4.EN_%",
             "Share of the kept documents, i.e. C4.EN (%, log scale)", True)):
        style(ax)
        v = t[col].to_numpy()
        ax.barh(y, v, height=0.6, color=[C_DIAL[d] for d in t["dialect"]], edgecolor=SURFACE, zorder=2)
        ax.errorbar(v, y, xerr=[v - t[lo], t[hi] - v], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=3)
        ax.scatter(t[paper], y, s=46, marker="D", facecolor=SURFACE, edgecolor=INK, linewidth=1.4, zorder=4)
        for yi, vi in zip(y, v):
            ax.annotate(f"{vi:.3g}%", (vi, yi), xytext=(6, -11), textcoords="offset points", fontsize=8, color=INK2)
        if log:
            ax.set_xscale("log")
        ax.set_yticks(y, t["dialect"], color=INK)
        ax.set_xlabel(xlabel, color=INK2)
    a1.legend(handles=[Line2D([], [], marker="D", color=INK, markerfacecolor=SURFACE, linestyle="none",
                              label="Dodge et al. 2021"),
                       Line2D([], [], color=INK2, marker="|", markersize=8, label="95% CI (domain bootstrap)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("§5.3 reproduction: which dialects the blocklist removes (argmax of Blodgett et al. 2016)",
                 color=INK, x=0.01, ha="left", fontsize=11, y=1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_dialects.png"), dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------- §5.2 manual check ----------

def inspection_sheet(scan_dir, out, seed, n=50):
    rows = []
    for p in sorted(glob.glob(os.path.join(scan_dir, "inspect_*.jsonl.gz"))):
        rows += [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8")]
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    rng = random.Random(seed)
    picked = []
    for term in ("lesbian", "gay"):
        d = df[df["term"] == term]
        picked.append(d.iloc[sorted(rng.sample(range(len(d)), min(n, len(d))))])
    s = pd.concat(picked)
    s = s.assign(blocklist_words=s["blocklist_words"].map(", ".join), text=s["text"].str.slice(0, 1500),
                 label="", notes="")
    s[["term", "url", "blocklist_words", "label", "notes", "text"]].to_csv(
        os.path.join(out, "inspect_lesbian_gay.csv"), index=False)
    return len(s)


# ---------- §5.1: clustering (Fig. 6) ----------

def fig6(X, labels, top5, sexual, summ_row, n_docs, seed, out, args, rng, plt):
    """Fig. 6: t-SNE of a 50-d SVD projection of --tsne_n docs, coloured by one k-means run's clusters."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    idx = np.array(sorted(rng.sample(range(X.shape[0]), min(args.tsne_n, X.shape[0]))))
    Z = TruncatedSVD(50, random_state=0).fit_transform(X[idx])
    E = TSNE(2, init="pca", random_state=0, perplexity=30).fit_transform(Z)
    cmap = [plt.get_cmap(m)(i) for m in ("tab20", "tab20b", "tab20c") for i in range(20)]
    markers = ["o", "^", "s", "v", "D", "P", "X", "*"]
    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for k in range(args.k):
        m = labels[idx] == k
        ax.scatter(E[m, 0], E[m, 1], s=7, color=cmap[k % 60], marker=markers[k % len(markers)], linewidths=0,
                   label=top5[k] + ("  [sexual]" if sexual[k] else ""))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.3, markerscale=2, frameon=False,
              labelcolor=INK2, ncol=1)
    ax.set_title(f"Figure 6 reproduction: k-means (k={args.k}) of {n_docs:,} excluded documents, TF-IDF; "
                 f"t-SNE of {len(idx):,} shown\nAuto-labelled sexual clusters (seed {seed}): "
                 f"{int(summ_row['sexual_clusters'])} of {args.k}, {summ_row['sexual_share_%']:.0f}% of documents "
                 f"(paper, by hand: 16 of 50, 31%)", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_clusters.png"), dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def clusters(scan_dir, out, args, plt):
    """k-means over the excluded sample, one run per --cluster_seeds. clusters.csv is rewritten after every seed
    and Fig. 6 is drawn right after the first, so a session that stops later keeps both."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = []
    for p in sorted(glob.glob(os.path.join(scan_dir, "excluded_sample_*.jsonl.gz"))):
        docs += [json.loads(l)["text"][:20000] for l in gzip.open(p, "rt", encoding="utf-8")]
    rng = random.Random(args.seed)
    docs = rng.sample(docs, min(args.n_cluster_docs, len(docs)))
    n_docs = len(docs)
    print(f"clustering {n_docs:,} excluded documents (paper: 100,000): TF-IDF ...", flush=True)
    vec = TfidfVectorizer(max_features=50000, stop_words="english", min_df=5, sublinear_tf=True)
    X = vec.fit_transform(docs)
    del docs                                                   # the raw text is no longer needed
    terms = np.array(vec.get_feature_names_out())
    bad = {w.strip() for w in open(BADWORDS, encoding="utf-8") if w.strip() and " " not in w.strip()}
    rows, summ = [], []
    for i, s in enumerate(args.cluster_seeds):
        t0 = time.time()
        print(f"  k-means seed {s} ({i + 1}/{len(args.cluster_seeds)}), about 10 min on 2 CPUs ...", flush=True)
        km = KMeans(n_clusters=args.k, n_init=1, random_state=s).fit(X)
        size = np.bincount(km.labels_, minlength=args.k)
        run = []
        for k in range(args.k):
            top = terms[np.argsort(-km.cluster_centers_[k])[:10]]
            run.append({"seed": s, "cluster": k, "docs": int(size[k]), "share_%": 100 * size[k] / n_docs,
                        "top10": " ".join(top), "blocklist_words_in_top10": int(sum(t in bad for t in top)),
                        "auto_sexual": bool(sum(t in bad for t in top) >= 2), "manual_label": ""})
        rows += run
        summ.append({"seed": s, "sexual_clusters": sum(r["auto_sexual"] for r in run),
                     "sexual_share_%": sum(r["share_%"] for r in run if r["auto_sexual"])})
        pd.DataFrame(rows).round(3).to_csv(os.path.join(out, "clusters.csv"), index=False)
        print(f"  seed {s}: {summ[-1]['sexual_clusters']} sexual clusters ({summ[-1]['sexual_share_%']:.1f}% of docs), "
              f"{(time.time() - t0) / 60:.0f} min; clusters.csv saved", flush=True)
        if i == 0:
            print("  drawing Fig. 6 (t-SNE of 5,000 docs) ...", flush=True)
            fig6(X, km.labels_, [" ".join(r["top10"].split()[:5]) for r in run], [r["auto_sexual"] for r in run],
                 summ[-1], n_docs, s, out, args, rng, plt)
            print("  fig6_clusters.png saved", flush=True)
    return pd.DataFrame(summ), n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="folder written by scan_noblocklist.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--cluster_seeds", type=int, nargs="+", default=[0],
                    help="one k-means run as in the paper; e.g. 0 1 2 adds a seed-robustness check")
    ap.add_argument("--n_cluster_docs", type=int, default=100000)
    ap.add_argument("--tsne_n", type=int, default=5000)
    ap.add_argument("--skip_clusters", action="store_true")
    ap.add_argument("--clusters_only", action="store_true",
                    help="only §5.1 (Fig. 6, clusters.csv, clusters_summary.md); keep the other outputs as they are")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, HERE)
    os.makedirs(args.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.clusters_only:
        clu, n_clu = clusters(args.scan, args.out_dir, args, plt)
        with open(os.path.join(args.out_dir, "clusters_summary.md"), "w", encoding="utf-8") as f:
            f.write(f"# §5.1 clusters ({n_clu:,} docs, k={args.k}; paper: 16 sexual clusters, 31%)\n\n"
                    + md_table(clu.round(1)))
        print(open(os.path.join(args.out_dir, "clusters_summary.md"), encoding="utf-8").read())
        return

    df, columns, stats = load_counts(args.scan)
    n_docs, n_ex = int(df["docs"].sum()), int(df["excluded"].sum())
    boot = bootstrap(df[columns].to_numpy(), args.B, args.seed)
    print(f"{len(stats)} shards, {n_docs:,} docs, {len(df):,} domains, {100 * n_ex / n_docs:.2f}% excluded", flush=True)

    pmi = identity_pmi(df, boot, columns)
    pmi.round(4).to_csv(os.path.join(args.out_dir, "fig5_pmi.csv"), index=False)
    rho = fig5(pmi, n_docs, len(stats), args.out_dir, args.dpi, plt)
    t6 = pmi.iloc[::-1][["term", "regex (Table 6)", "docs", "excluded", "excluded_%"]].round(1)
    with open(os.path.join(args.out_dir, "table6.md"), "w", encoding="utf-8") as f:
        f.write("# Table 6: identity regexes (case-insensitive, word boundaries), with counts\n\n" + md_table(t6))

    dia = dialects(df, boot, columns)
    dia.round(4).to_csv(os.path.join(args.out_dir, "dialects.csv"), index=False)
    fig_dialects(dia, args.out_dir, args.dpi, plt)

    n_inspect = inspection_sheet(args.scan, args.out_dir, args.seed)
    clu, n_clu = (None, 0) if args.skip_clusters else clusters(args.scan, args.out_dir, args, plt)

    agree = np.average([s["dialect_fast_vs_exact_agreement"] for s in stats],
                       weights=[s["dialect_validated_docs"] for s in stats])
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# §5 reproduction: {len(stats)} shards of en.noblocklist, {n_docs:,} documents, {len(df):,} domains\n\n"
                f"- Excluded by the blocklist: {100 * n_ex / n_docs:.2f}% (expected about 7.6% from 395M vs 365M docs)\n"
                f"- Dialect inference: fast vs exact CVB0 argmax agreement {100 * agree:.2f}% "
                f"on {sum(s['dialect_validated_docs'] for s in stats):,} documents\n"
                f"- Figure 5: Spearman with the paper's bars (read off the figure) = {rho:.3f}\n"
                f"- Manual-check sheet: {n_inspect} documents in inspect_lesbian_gay.csv (label them; paper: 22% / 36% "
                "non-offensive or non-sexual)\n\n")
        f.write("## §5.2 identity PMI (Fig. 5)\n\n" + md_table(pmi.iloc[::-1].drop(columns="regex (Table 6)").round(3)) + "\n")
        f.write("## §5.3 dialects\n\n" + md_table(dia.round(3)) + "\n")
        if clu is not None:
            f.write(f"## §5.1 clusters ({n_clu:,} docs, k={args.k}; paper: 16 sexual clusters, 31%)\n\n"
                    + md_table(clu.round(1)) + "\n")
    print(open(os.path.join(args.out_dir, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
