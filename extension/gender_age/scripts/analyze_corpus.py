"""Extension, corpus analysis: exclusion and sentiment by gender, age and gender x ethnicity. Reads scan_corpus.py.

1. Exclusion (Dodge §5.2). Per term: share of documents excluded by the blocklist, and document-level
   PMI = ln[P(term, excluded) / (P(term) P(excluded))] (the reproduction's definition, natural log). Also the same
   ratio counting only exclusions caused by a blocklist entry that does not contain the term's own words
   ("non-mechanical"; see scan_corpus.py). Women / men / female / male are Table 6 terms, so their PMIs can be checked
   against Fig. 5 (paper, read off the figure: women 1.73, man 1.65, female 1.82, male 1.90).
2. Dialect x gender (§5.3): removal rate of AAE / Hispanic / WAE documents, overall and among documents that
   mention a female or a male term.
3. Sentiment (A.7). Per term and lexicon: positivity = positive / (positive + negative) sentiment tokens in
   paragraphs mentioning the term, in kept documents (= C4.EN) and separately in excluded ones. Positive control: the
   paper-style " Jewish " / " Arab " gap, which Arm B measured as 7.1 (SocialSent) and 2.0 (VADER) on 12 C4.EN shards.
4. Gaps for matched pairs (women - men, girl - boy, Black women - Black men, ...), for both exclusion and sentiment.
5. Time: the same numbers per URL-year bucket (C4's timestamps are all April 2019; see scan_corpus.py).
6. Classifier (if para_scores.csv.gz exists): positivity = sum p_pos / (sum p_pos + sum p_neg) per term.

Uncertainty: 95% percentile intervals from a bootstrap over web domains (Poisson(1) weights per domain, B=200 by
default), because documents and paragraphs from one site are not independent. A gap's interval comes from the same
bootstrap replicates for both terms.

Usage:
    python scripts/analyze_corpus.py --scan results/C1_noblocklist_12_shards --out_dir results/C1_noblocklist_12_shards/analysis
"""
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = [("women", "men"), ("female", "male"), ("girl", "boy"), ("lady", "gentleman"), ("mother", "father"),
         ("wife", "husband"), ("daughter", "son"), ("she/her", "he/him"), ("teen girl", "teen boy"),
         ("little girl", "little boy"), ("young woman", "young man"), ("middle-aged woman", "middle-aged man"),
         ("old woman", "old man")]
ETH = ["Jewish", "Arab", "Black", "White", "Asian", "Hispanic", "Latina/o", "African American", "Indian",
       "European", "Chinese", "Muslim"]
PAIRS += [(f"{e} women", f"{e} men") for e in ETH]
AGES = ["child", "teen", "young adult", "adult", "middle-aged", "elderly", "person"]
LEXICONS = ["SocialSent", "VADER", "UnQover"]
YEARS = ["<=2009", "2010-12", "2013-15", "2016-17", "2018-19", "none"]
PAPER_FIG5 = {"women": 1.73, "men": 1.65, "female": 1.82, "male": 1.90}
ARM_B = {"SocialSent": 7.1, "VADER": 2.0}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
C_F, C_M = "#e87ba4", "#2a78d6"


def load(scan, name):
    files = sorted(glob.glob(os.path.join(scan, f"{name}_*.npz")))
    done = {os.path.basename(p)[6:-5] for p in glob.glob(os.path.join(scan, "stats_*.json"))}
    files = [p for p in files if p[len(os.path.join(scan, name)) + 1:-4] in done]
    if not files:
        raise SystemExit(f"no finished {name}_*.npz in {scan}")
    keys, mats, columns = [], [], None
    for p in files:
        z = np.load(p)
        columns = list(z["columns"])
        keys += list(z["keys"])
        mats.append(sparse.csr_matrix(z["counts"]))
    X = sparse.vstack(mats).tocsr()
    parts = pd.Series(keys).str.split("\t", expand=True)
    return X, columns, parts, len(files)


class Boot:
    """Totals and domain-bootstrap totals for row subsets of a sparse rows x columns count matrix."""

    def __init__(self, X, host, B, seed):
        self.X, self.B = X, B
        self.codes, uniq = pd.factorize(host)
        rng = np.random.default_rng(seed)
        self.W = rng.poisson(1.0, size=(len(uniq), B)).astype(np.float32)   # hosts x B

    def totals(self, mask=None):
        X = self.X if mask is None else self.X[np.flatnonzero(mask)]
        codes = self.codes if mask is None else self.codes[np.flatnonzero(mask)]
        point = np.asarray(X.sum(0)).ravel().astype(float)
        boot = np.asarray((X.T @ self.W[codes])).T            # B x columns
        return point, boot


def ci(arr):
    return np.nanpercentile(arr, [2.5, 97.5], axis=0)


def exclusion_tables(point, boot, col, terms):
    rows = []
    N, E = point[col["docs"]], point[col["excluded"]]
    bN, bE = boot[:, col["docs"]], boot[:, col["excluded"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in terms:
            d, e, o = (point[col[f"{t}|{k}"]] for k in ("docs", "excluded", "excluded_other"))
            bd, be, bo = (boot[:, col[f"{t}|{k}"]] for k in ("docs", "excluded", "excluded_other"))
            pmi, bpmi = np.log(e * N / (d * E)), np.log(be * bN / (bd * bE))
            pmo, bpmo = np.log(o * N / (d * E)), np.log(bo * bN / (bd * bE))
            rate, brate = 100 * e / d, 100 * be / bd
            rows.append({"term": t, "docs": int(d), "excluded": int(e), "excluded_%": rate,
                         "excluded_%_lo": ci(brate)[0], "excluded_%_hi": ci(brate)[1],
                         "pmi": pmi, "pmi_lo": ci(bpmi)[0], "pmi_hi": ci(bpmi)[1],
                         "pmi_nonmechanical": pmo, "pmi_nonmech_lo": ci(bpmo)[0], "pmi_nonmech_hi": ci(bpmo)[1],
                         "paper_fig5": PAPER_FIG5.get(t)})
    return pd.DataFrame(rows), (bN, bE)


def pair_gaps(point, boot, col, metric):
    """metric(point_or_boot, term) -> value; returns female - male per pair with CI."""
    rows = []
    for f, m in PAIRS:
        try:
            p = metric(point, f) - metric(point, m)
            b = metric(boot, f) - metric(boot, m)
        except KeyError:
            continue
        lo, hi = ci(b)
        rows.append({"female": f, "male": m, "gap": p, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def eth_zoom(point, boot, col, value):
    """Gender first, then race: per ethnicity, value(<eth> women) - value(<eth> men), and how far that gap is from
    the gap pooled over ethnicities (all "<eth> women" counts summed vs all "<eth> men"). value(arr, [terms]) sums
    the terms' counts before computing. Bootstrap p-values, Holm-corrected over ethnicities."""
    from analyze_probe_stats import holm, p_boot
    have = [e for e in ETH if f"{e} women" in col and f"{e} men" in col]
    fem, mal = [f"{e} women" for e in have], [f"{e} men" for e in have]
    pooled, bpooled = value(point, fem) - value(point, mal), value(boot, fem) - value(boot, mal)
    rows = []
    for e in have:
        g, bg = value(point, [f"{e} women"]) - value(point, [f"{e} men"]),             value(boot, [f"{e} women"]) - value(boot, [f"{e} men"])
        d = bg - bpooled
        rows.append({"ethnicity": e, "women_minus_men": g, "ci_lo": ci(bg)[0], "ci_hi": ci(bg)[1],
                     "minus_pooled_gap": g - pooled, "diff_lo": ci(d)[0], "diff_hi": ci(d)[1], "p_boot": p_boot(d)})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = holm(t["p_boot"].fillna(1))
        t["anomaly"] = t["p_holm"] < 0.05
    return t, pooled


def positivity(arr, col, t, lex):
    pos, neg = arr[..., col[f"{t}|{lex}:pos"]], arr[..., col[f"{t}|{lex}:neg"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * pos / (pos + neg)


def sentiment_tables(point, boot, col, terms):
    rows = []
    for t in terms:
        for lex in LEXICONS:
            v, b = positivity(point, col, t, lex), positivity(boot, col, t, lex)
            lo, hi = ci(b)
            rows.append({"term": t, "lexicon": lex, "paragraphs": int(point[col[f"{t}|paragraphs"]]),
                         "sentiment_tokens": int(point[col[f"{t}|{lex}:pos"]] + point[col[f"{t}|{lex}:neg"]]),
                         "positivity": v, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_pairs(df, value, lo, hi, xlabel, title, path, plt, ref=0):
    df = df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(df) + 1.5))
    style(ax)
    y = np.arange(len(df))
    v = df[value].to_numpy(float)
    ax.barh(y, v - ref, left=ref, color=[C_F if x >= ref else C_M for x in v], height=0.6, edgecolor=SURFACE)
    ax.errorbar(v, y, xerr=[v - df[lo], df[hi] - v], fmt="none", ecolor=INK2, capsize=3, elinewidth=1.1)
    ax.axvline(ref, color=INK2, linewidth=1)
    ax.set_yticks(y, df["female"] + " vs " + df["male"], fontsize=8.5, color=INK2)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_title(title, loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_terms(df, value, lo, hi, xlabel, title, path, plt, extra=None):
    df = df.sort_values(value).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.26 * len(df) + 1.5))
    style(ax)
    y = np.arange(len(df))
    v = df[value].to_numpy(float)
    ax.errorbar(v, y, xerr=[v - df[lo], df[hi] - v], fmt="o", color=INK, ms=4, ecolor=INK2, capsize=2.5)
    if extra is not None:
        e = df[extra[0]].to_numpy(float)
        ax.scatter(e, y, marker=extra[2], s=28, facecolor=SURFACE, edgecolor=MUTED, zorder=3, label=extra[1])
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_yticks(y, df["term"], fontsize=8, color=INK2)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_title(title, loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def md(df, cols=None):
    d = (df if cols is None else df[cols]).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    return "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join(
        "| " + " | ".join(str(x) for x in r) + " |" for r in d.itertuples(index=False)) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_docs", type=int, default=30, help="terms with fewer documents are listed but not plotted")
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plt = None
    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    report = ["# Gender and age in C4: exclusion and sentiment\n"]

    # ---------- exclusion ----------
    X, cols, parts, nshards = load(args.scan, "doc_counts")
    col = {c: i for i, c in enumerate(cols)}
    terms = [c[:-5] for c in cols if c.endswith("|docs") and not c.startswith("dialect:") and c != "docs"]
    host, year = parts[0].to_numpy(), parts[1].to_numpy()
    bt = Boot(X, host, args.B, args.seed)
    point, boot = bt.totals()
    ex, _ = exclusion_tables(point, boot, col, terms)
    ex.round(3).to_csv(os.path.join(args.out_dir, "exclusion_by_term.csv"), index=False)
    report.append(f"{nshards} en.noblocklist shards, {int(point[col['docs']]):,} documents, "
                  f"{100 * point[col['excluded']] / point[col['docs']]:.2f}% excluded by the blocklist.\n")
    report.append("## Exclusion by term (PMI with being excluded; 95% CI over domains)\n")
    report.append(md(ex.sort_values("pmi", ascending=False),
                     ["term", "docs", "excluded_%", "pmi", "pmi_lo", "pmi_hi", "pmi_nonmechanical", "paper_fig5"]))

    def pmi_of(arr, t, kind="excluded"):
        if f"{t}|docs" not in col:
            raise KeyError(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(arr[..., col[f"{t}|{kind}"]] * arr[..., col["docs"]]
                          / (arr[..., col[f"{t}|docs"]] * arr[..., col["excluded"]]))
    gp = pair_gaps(point, boot, col, pmi_of)
    gpo = pair_gaps(point, boot, col, lambda a, t: pmi_of(a, t, "excluded_other"))
    gp = gp.merge(gpo.rename(columns={"gap": "gap_nonmechanical", "ci_lo": "nm_lo", "ci_hi": "nm_hi"}),
                  on=["female", "male"])
    gp.round(3).to_csv(os.path.join(args.out_dir, "exclusion_gaps.csv"), index=False)

    def excl_rate(arr, ts):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 100 * sum(arr[..., col[f"{t}|excluded"]] for t in ts) / sum(arr[..., col[f"{t}|docs"]] for t in ts)
    ez, ezp = eth_zoom(point, boot, col, excl_rate)
    ez.round(3).to_csv(os.path.join(args.out_dir, "exclusion_gender_by_ethnicity.csv"), index=False)
    report.append(f"## Exclusion, gender first then race: removal rate of '<eth> women' minus '<eth> men' "
                  f"(pooled gap {ezp:.2f} points)\n")
    report.append(md(ez))
    report.append("## Exclusion: female minus male PMI for matched pairs\n")
    report.append(md(gp))

    # by URL year
    yrows = []
    for yb in YEARS:
        p, b = bt.totals(year == yb)
        e, _ = exclusion_tables(p, b, col, terms)
        g = pair_gaps(p, b, col, pmi_of)
        e["year"], g["year"] = yb, yb
        yrows.append((e, g))
    ey = pd.concat([e for e, _ in yrows])
    gy = pd.concat([g for _, g in yrows])
    ey.round(3).to_csv(os.path.join(args.out_dir, "exclusion_by_year.csv"), index=False)
    gy.round(3).to_csv(os.path.join(args.out_dir, "exclusion_gaps_by_year.csv"), index=False)
    share = {yb: int(bt.totals(year == yb)[0][col["docs"]]) for yb in YEARS}
    report.append(f"URL-year coverage (documents): {share}\n")

    # dialect x gender
    drows = []
    for d in ["AAE", "Hispanic", "Other", "WAE", "unassigned"]:
        for g in ("all", "female", "male"):
            k = f"dialect:{d}|{g}"
            if f"{k}|docs" not in col:
                continue
            n, e = point[col[f"{k}|docs"]], point[col[f"{k}|excluded"]]
            with np.errstate(divide="ignore", invalid="ignore"):
                lo, hi = ci(100 * boot[:, col[f"{k}|excluded"]] / boot[:, col[f"{k}|docs"]])
            drows.append({"dialect": d, "documents mentioning": g, "docs": int(n),
                          "removed_%": 100 * e / n if n else np.nan, "ci_lo": lo, "ci_hi": hi})
    if drows and sum(r["docs"] for r in drows):
        dg = pd.DataFrame(drows)
        dg.round(2).to_csv(os.path.join(args.out_dir, "dialect_by_gender.csv"), index=False)
        report.append("## Blocklist removal by dialect and gender mention (paper overall: AAE 42%, Hispanic 32%, WAE 6.2%)\n")
        report.append(md(dg))

    # ---------- sentiment ----------
    S, scols, sparts, _ = load(args.scan, "sent_counts")
    scol = {c: i for i, c in enumerate(scols)}
    sterms = [c[:-11] for c in scols if c.endswith("|paragraphs")]
    shost, syear, skept = sparts[0].to_numpy(), sparts[1].to_numpy(), sparts[2].to_numpy()
    sb = Boot(S, shost, args.B, args.seed)
    sent_out, gaps_out = [], []
    for label, mask in (("kept (C4.EN)", skept == "1"), ("excluded", skept == "0")):
        p, b = sb.totals(mask)
        t = sentiment_tables(p, b, scol, sterms)
        t.insert(0, "documents", label)
        sent_out.append(t)
        for lex in LEXICONS:
            g = pair_gaps(p, b, scol, lambda a, tt: positivity(a, scol, tt, lex))
            g.insert(0, "lexicon", lex), g.insert(0, "documents", label)
            gaps_out.append(g)
        if label.startswith("kept"):
            for lex in ("SocialSent", "VADER"):
                def pos_sum(arr, ts, lex=lex):
                    pp = sum(arr[..., scol[f"{t}|{lex}:pos"]] for t in ts)
                    nn = sum(arr[..., scol[f"{t}|{lex}:neg"]] for t in ts)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return 100 * pp / (pp + nn)
                sz, szp = eth_zoom(p, b, scol, pos_sum)
                sz.round(3).to_csv(os.path.join(args.out_dir, f"sentiment_gender_by_ethnicity_{lex}.csv"), index=False)
                report.append(f"## Sentiment ({lex}, C4.EN), gender first then race: positivity of '<eth> women' "
                              f"minus '<eth> men' (pooled gap {szp:.2f} points)\n")
                report.append(md(sz))
            ctrl = {}
            for lex in ("SocialSent", "VADER"):
                v = positivity(p, scol, "Jewish (paper)", lex) - positivity(p, scol, "Arab (paper)", lex)
                bb = positivity(b, scol, "Jewish (paper)", lex) - positivity(b, scol, "Arab (paper)", lex)
                ctrl[lex] = {"ours": round(float(v), 2), "ci": [round(float(x), 2) for x in ci(bb)],
                             "arm_b_c4en_12_shards": ARM_B[lex]}
            report.append(f"## Positive control: Jewish - Arab positivity gap (paper-style matching)\n\n{json.dumps(ctrl)}\n")
    st = pd.concat(sent_out)
    sg = pd.concat(gaps_out)
    st.round(2).to_csv(os.path.join(args.out_dir, "sentiment_by_term.csv"), index=False)
    sg.round(2).to_csv(os.path.join(args.out_dir, "sentiment_gaps.csv"), index=False)
    report.append("## Sentiment: female minus male positivity (points), kept documents\n")
    report.append(md(sg[sg["documents"].str.startswith("kept")]))
    report.append("## Sentiment by age term, kept documents (SocialSent)\n")
    report.append(md(st[(st["documents"].str.startswith("kept")) & (st["lexicon"] == "SocialSent")
                        & st["term"].isin(AGES)], ["term", "paragraphs", "positivity", "ci_lo", "ci_hi"]))

    yrows = []
    for yb in YEARS:
        p, b = sb.totals((syear == yb) & (skept == "1"))
        for lex in ("SocialSent", "VADER"):
            g = pair_gaps(p, b, scol, lambda a, tt: positivity(a, scol, tt, lex))
            g.insert(0, "lexicon", lex), g.insert(0, "year", yb)
            yrows.append(g)
    sgy = pd.concat(yrows)
    sgy.round(2).to_csv(os.path.join(args.out_dir, "sentiment_gaps_by_year.csv"), index=False)

    # ---------- classifier ----------
    ps = os.path.join(args.scan, "para_scores.csv.gz")
    if os.path.exists(ps):
        c = pd.read_csv(ps)
        c["host"] = c["url"].str.extract(r"^\w+://([^/]+)")[0]
        crow = []
        rng = np.random.default_rng(args.seed)
        for (t, k), d in c.groupby(["term", "kept"]):
            h = d.groupby("host")[["p_pos", "p_neg"]].sum()
            W = rng.poisson(1.0, size=(len(h), args.B))
            bp, bn = h["p_pos"].to_numpy() @ W, h["p_neg"].to_numpy() @ W
            with np.errstate(divide="ignore", invalid="ignore"):
                lo, hi = ci(100 * bp / (bp + bn))
            crow.append({"term": t, "kept": k, "paragraphs": len(d),
                         "positivity": 100 * h["p_pos"].sum() / (h["p_pos"].sum() + h["p_neg"].sum()),
                         "ci_lo": lo, "ci_hi": hi, "mean_pos_minus_neg": float((d["p_pos"] - d["p_neg"]).mean())})
        ct = pd.DataFrame(crow)
        ct.round(2).to_csv(os.path.join(args.out_dir, "classifier_by_term.csv"), index=False)
        kept = ct[ct["kept"] == 1].set_index("term")
        cg = [{"female": f, "male": m, "gap": kept.loc[f, "positivity"] - kept.loc[m, "positivity"]}
              for f, m in PAIRS if f in kept.index and m in kept.index]
        report.append("## Classifier (RoBERTa sentiment) positivity, kept documents: female minus male\n")
        report.append(md(pd.DataFrame(cg)))

    # ---------- figures ----------
    if plt is not None:
        big = ex[ex["docs"] >= args.min_docs]
        fig_terms(big, "pmi", "pmi_lo", "pmi_hi", "PMI with being removed by the blocklist (ln; 95% CI over domains)",
                  "Which gender / age mentions the blocklist removes (hollow: without mechanical matches)",
                  os.path.join(args.out_dir, "fig_exclusion_pmi.png"), plt, extra=("pmi_nonmechanical", "non-mechanical", "o"))
        okp = gp.dropna(subset=["gap"])
        fig_pairs(okp, "gap", "ci_lo", "ci_hi", "PMI(female term) - PMI(male term); > 0: female mentions removed more",
                  "Blocklist: female vs male mentions", os.path.join(args.out_dir, "fig_exclusion_gaps.png"), plt)
        k = sg[(sg["documents"].str.startswith("kept")) & (sg["lexicon"] == "SocialSent")].dropna(subset=["gap"])
        fig_pairs(k, "gap", "ci_lo", "ci_hi", "positivity(female) - positivity(male), points; SocialSent, C4.EN",
                  "Sentiment of paragraphs mentioning female vs male terms",
                  os.path.join(args.out_dir, "fig_sentiment_gaps.png"), plt)
        kt = st[(st["documents"].str.startswith("kept")) & (st["lexicon"] == "SocialSent") & (st["paragraphs"] >= args.min_docs)]
        fig_terms(kt, "positivity", "ci_lo", "ci_hi", "SocialSent positivity of paragraphs (%; 95% CI over domains)",
                  "Sentiment of C4.EN paragraphs mentioning each term", os.path.join(args.out_dir, "fig_sentiment_terms.png"), plt)
        fig, ax = plt.subplots(figsize=(7, 4))
        style(ax)
        for (f, m), colr in zip([("women", "men"), ("she/her", "he/him"), ("girl", "boy")], [C_F, "#6250d6", "#1baf7a"]):
            g = sgy[(sgy["female"] == f) & (sgy["lexicon"] == "SocialSent") & (sgy["year"] != "none")]
            x = np.arange(len(g))
            ax.errorbar(x, g["gap"], yerr=[g["gap"] - g["ci_lo"], g["ci_hi"] - g["gap"]], marker="o", color=colr,
                        capsize=3, label=f"{f} - {m}")
            ax.set_xticks(x, g["year"])
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_ylabel("female - male positivity (points)", color=INK2)
        ax.set_xlabel("year in the page's URL", color=INK2)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Sentiment gap by URL year (C4.EN, SocialSent)", loc="left", fontsize=10.5)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_sentiment_gap_by_year.png"), dpi=200, facecolor=SURFACE)
        plt.close(fig)

    open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(report))
    print("\n".join(report[:3]))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
