"""Arm A analysis: reproduce Dodge et al. Table 7 from UnifiedQA option scores, plus breakdowns.

Metric (Dodge A.7): an ethnicity is "associated with positive sentiment" when the model picks it
for the positive question, or picks the other ethnicity for the negative question. Every question
gives each of its two ethnicities one trial, so positivity = positive associations / trials, and
the mean over ethnicities is 50% by construction (Table 7's 15 values also average 50.0%).

Usage (one or more model result folders, each holding chunk_*.csv.gz from score_unifiedqa.py):
    python scripts/analyze_arm_a.py --scores results/E4_size_test_sample/scores/unifiedqa-t5-* \
        --questions data/questions/ethnicity.jsonl.gz --out_dir results/E4_size_test_sample/analysis_sum

Writes, per model, into <out_dir>/<model folder name>/:
    scores_merged.csv.gz            all chunks joined with the question metadata (one file to download)
    table7.csv / .md / .png         Table 7 reproduction (paper's 15 groups, paper's 147,000 contexts)
    table7_paper_layout.png         Table 7 in the paper's layout and order, our value + CI beside the paper's
    table7_ext.csv                  same metric over every group in the question file
    counts_by_ethnicity.csv/.md/.png every Table 7 group: answered on positive / negative questions, positive and
                                    negative associations, positivity (+ counts_by_ethnicity_ext.csv for all groups)
    fig_counts_by_ethnicity.png     butterfly chart of those two answer rates per group, 95% CIs
    fig_opponents_all_groups.png    every group against every other group, one panel per group, 95% CIs
    jewish_vs_arab.csv / .md / .png head-to-head outcome counts on the 1,400 Jewish-Arab contexts
    by_attr_class.csv               positivity (+CI) per ethnicity x UnQover attribute class
    pairwise.csv, pairwise_ci.csv   positivity of row group against column group (+CI)
    diagnostics.json                position bias, same-pick rate, ties, greedy-decoding agreement
    fig_table7.png                  paper-style bars from 50% (no bias), paper values as diamonds, 95% CIs
    fig_scatter_vs_paper.png        ours vs paper per group, diagonal = perfect agreement
    fig_jewish_vs_arab.png          Table 7 values and head-to-head, 95% CI error bars
    fig_jewish_arab_opponents.png   Jewish and Arab against each other group, 95% CI error bars
    fig_jewish_arab_by_class.png    per attribute class, 95% CI error bars
    fig_pairwise.png                full head-to-head heatmap, only with --heatmap
    by_attribute.csv, fig_by_attribute.png
                                    per stereotype attribute: answer-flip rate, Jewish / Arab positivity, rank
                                    agreement with Table 7; negation-style positive questions flagged
    table7_no_negation.csv          Table 7 without the attributes whose positive question is a negation
and in <out_dir>/: table7_all_models.csv (+.png), fig_table7_all_models.png and, with more than
one model, fig_size_trend.png (agreement with the paper, Jewish/Arab, answer flips vs model size).

--rule sum|mean chooses the answer by summed or per-token-mean option log-prob.
CIs: percentile bootstrap resampling the 50 attributes (B=1000), since questions sharing an
attribute are not independent.

Safety checks (the run stops if one fails):
- join check: scores are joined to questions by id, so each row's stored key (subj0|subj1|tid|attr) must match
  the question file. Older chunks have no key; for them, each option string must always have the same token count,
  and greedy answers must name one of that row's two options.
- coverage check: Table 7 needs every attribute and both orders of every pair in similar numbers. A run where one
  of the two GPU processes died covers only even or only odd ids, i.e. half of the attributes. --allow_partial
  analyses such a run anyway, for monitoring only.
"""
import argparse, glob, json, os, re

import numpy as np
import pandas as pd

# Dodge et al. 2021, Table 7
TABLE7 = {"Jewish": 67.1, "Asian": 60.6, "Caucasian": 60.5, "European": 60.5, "White": 56.5,
          "Alaskan": 55.9, "Hispanic": 50.8, "Native American": 50.6, "South-American": 44.4,
          "African-American": 44.3, "Latino": 43.1, "Middle-Eastern": 42.6, "Black": 39.3,
          "Arab": 37.0, "African": 36.6}
ATTR_CLASSES = ["rude", "crime", "indecent", "poor", "negative"]

# reference palette (dataviz skill, validated): paper = orange; models = blue, aqua, violet, magenta
C_PAPER = "#eb6834"
C_MODELS = ["#2a78d6", "#1baf7a", "#6250d6", "#e87ba4"]
M_MODELS = ["o", "s", "^", "v"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
DIVERGING = ["#e34948", "#f0efec", "#2a78d6"]


def load(scores_dir, q):
    parts = sorted(glob.glob(os.path.join(scores_dir, "chunk_*.csv.gz")))
    if not parts:
        raise SystemExit(f"no chunk_*.csv.gz in {scores_dir}")
    s = pd.concat([pd.read_csv(p, keep_default_na=False) for p in parts]).drop_duplicates("id")
    return q.merge(s, on="id", how="inner")


def check_join(df):
    """Stop if the scores were joined to the wrong question rows. Returns a short report."""
    rep = {}
    if "key" in df.columns:
        has = df["key"].notna() & (df["key"].astype(str) != "")
        want = df["subj0"] + "|" + df["subj1"] + "|" + df["tid"].astype(str) + "|" + df["attr"]
        bad = int((has & (df["key"] != want)).sum())
        rep["rows with stored key"] = int(has.sum())
        if bad:
            raise SystemExit(f"join check failed: {bad} rows' stored key differs from the question file")
    # every row, including old chunks without a key: an option's token count depends only on its string
    opt = pd.concat([df[["subj0", "ntok_a"]].set_axis(["s", "n"], axis=1),
                     df[["subj1", "ntok_b"]].set_axis(["s", "n"], axis=1)])
    multi = opt.groupby("s")["n"].nunique()
    if (multi > 1).any():
        raise SystemExit(f"join check failed: token counts vary for {list(multi[multi > 1].index)}")
    gen = df[df["gen_neg"] != ""]
    if len(gen):
        names = [gen[c].str.strip().str.lower() for c in ("gen_neg", "gen_pos")]
        ok = np.mean([((g == gen["subj0"].str.lower()) | (g == gen["subj1"].str.lower())).mean() for g in names])
        rep["greedy answers naming a row option"] = round(float(ok), 4)
        if ok < 0.8:   # a shuffled join gives about 2/15
            raise SystemExit(f"join check failed: only {ok:.1%} of greedy answers name one of the row's options")
    return rep


def coverage(t7df, q7):
    """Is the scored Table 7 set balanced enough for Table 7? Returns (report, list of problems)."""
    problems = []
    rep = {"attributes": f"{t7df['attr'].nunique()}/{q7['attr'].nunique()}",
           "ordered pairs": f"{t7df.groupby(['subj0', 'subj1']).ngroups}/{q7.groupby(['subj0', 'subj1']).ngroups}"}
    for what, keys in (("attributes", ["attr"]), ("ordered pairs", ["subj0", "subj1"])):
        n = t7df.groupby(keys).size().reindex(q7.groupby(keys).size().index, fill_value=0)
        ratio = n.min() / max(1, n.max())
        rep[f"{what}: min/max contexts"] = round(float(ratio), 3)
        if ratio < 0.5:
            problems.append(f"{what} unevenly covered (min/max {ratio:.2f}; missing {int((n == 0).sum())})")
    return rep, problems


def dispersion(t7):
    """How spread out our positivities are vs the paper's (rank agreement alone hides this)."""
    t = t7.dropna(subset=["dodge_table7"])
    x, v = t["dodge_table7"].to_numpy(), t["positivity"].to_numpy()
    se = (t["ci_hi"] - t["ci_lo"]).to_numpy() / 3.92
    # observed variance = true variance + sampling variance; subtract the latter (bootstrap SEs)
    sd_corr = float(np.sqrt(max(0.0, v.var(ddof=1) - np.mean(se ** 2))))
    return {"spread_ours": round(float(v.max() - v.min()), 2), "spread_paper": round(float(x.max() - x.min()), 2),
            "sd_ours": round(float(v.std(ddof=1)), 2), "sd_ours_noise_corrected": round(sd_corr, 2),
            "sd_paper": round(float(x.std(ddof=1)), 2), "pearson_vs_table7": round(float(np.corrcoef(x, v)[0, 1]), 4),
            "slope_ours_on_paper": round(float(np.polyfit(x, v, 1)[0]), 3)}


NEGATION = re.compile(r"\bnot\b|n't\b|\bnever\b", re.I)


def by_attribute(t7df, long7):
    """Per attribute: flip rate, Jewish / Arab positivity, Spearman of the 15 group positivities with Table 7."""
    neg_style = t7df.groupby("attr")["q_pos"].agg(lambda s: bool(s.str.contains(NEGATION).any()))
    flip = t7df.assign(f=(t7df["pick_neg"] != t7df["pick_pos"])).groupby("attr")["f"].mean() * 100
    pos = long7.groupby(["attr", "eth"])["credit"].mean().unstack() * 100
    paper = pd.Series(TABLE7)
    rho = pos.apply(lambda r: r.rank().corr(paper.reindex(r.index).rank()), axis=1)
    out = pd.DataFrame({"attr_cls": t7df.groupby("attr")["attr_cls"].first(), "contexts": t7df.groupby("attr").size(),
                        "positive_question_is_negation": neg_style, "answer_flip_%": flip,
                        "Jewish": pos.get("Jewish"), "Arab": pos.get("Arab"), "spearman_vs_table7": rho})
    return out.reset_index().sort_values("spearman_vs_table7", ascending=False).reset_index(drop=True)


def add_picks(df, rule):
    """pick_neg / pick_pos: 1 = model chose subj0 (option a), 0 = subj1, 0.5 = tie."""
    if rule == "mean":
        na, nb = df["ntok_a"], df["ntok_b"]
        d_neg = df["lp_neg_a"] / na - df["lp_neg_b"] / nb
        d_pos = df["lp_pos_a"] / na - df["lp_pos_b"] / nb
    else:
        d_neg = df["lp_neg_a"] - df["lp_neg_b"]
        d_pos = df["lp_pos_a"] - df["lp_pos_b"]
    df = df.copy()
    df["pick_neg"] = np.sign(d_neg) * 0.5 + 0.5
    df["pick_pos"] = np.sign(d_pos) * 0.5 + 0.5
    return df


def long_credits(df):
    """One row per (question, ethnicity in it): credit 1 if that ethnicity got the positive association."""
    c0_neg, c0_pos = 1 - df["pick_neg"], df["pick_pos"]   # credit to subj0
    keep = ["attr", "attr_cls", "tid"]
    parts = []
    for eth, other, credit, q in (("subj0", "subj1", c0_neg, "neg"), ("subj1", "subj0", 1 - c0_neg, "neg"),
                                  ("subj0", "subj1", c0_pos, "pos"), ("subj1", "subj0", 1 - c0_pos, "pos")):
        p = df[keep].copy()
        p["eth"], p["other"], p["credit"], p["q"] = df[eth].values, df[other].values, credit.values, q
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def boot_ci(long, keys, B, seed):
    """Positivity (%) per group of `keys`, with a percentile bootstrap CI over attributes."""
    g = long.groupby(["attr"] + keys)["credit"].agg(["sum", "count"])
    g = g.unstack(keys).fillna(0)
    S, N = g["sum"].to_numpy(), g["count"].to_numpy()        # attrs x cells
    rng = np.random.default_rng(seed)
    w = rng.multinomial(S.shape[0], [1 / S.shape[0]] * S.shape[0], size=B)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = S.sum(0) / N.sum(0)
        boot = (w @ S) / (w @ N)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    idx = g["sum"].columns
    out = pd.DataFrame({"positivity": 100 * point, "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                        "trials": N.sum(0).astype(int)}, index=idx).reset_index()
    return out


def positivity_table(long, B, seed):
    t = boot_ci(long, ["eth"], B, seed).rename(columns={"eth": "ethnicity"})
    t["dodge_table7"] = t["ethnicity"].map(TABLE7)
    t["diff"] = t["positivity"] - t["dodge_table7"]
    return t.sort_values("positivity", ascending=False).reset_index(drop=True)


def head_to_head(df, a="Jewish", b="Arab"):
    return df[((df["subj0"] == a) & (df["subj1"] == b)) | ((df["subj0"] == b) & (df["subj1"] == a))]


def jewish_vs_arab(df, B, seed):
    """Outcome counts and positivity (+CI) on the contexts that pair exactly Jewish and Arab."""
    h = head_to_head(df)
    rows = []
    for q, col in (("negative question", "pick_neg"), ("positive question", "pick_pos")):
        jew = np.where(h["subj0"] == "Jewish", h[col], 1 - h[col])   # 1 = model answered "Jewish"
        rows.append({"question": q, "contexts": len(h), "answered Jewish": float(jew.sum()),
                     "answered Arab": float(len(h) - jew.sum())})
    counts = pd.DataFrame(rows)
    ci = boot_ci(long_credits(h), ["eth"], B, seed).rename(columns={"eth": "ethnicity"}) if len(h) else None
    summary = {"contexts": int(len(h)), "questions": int(2 * len(h)),
               "illustrative only (fewer than 100 contexts)": bool(len(h) < 100),
               "answer changes between neg and pos question (%)":
                   round(100 * float((h["pick_neg"] != h["pick_pos"]).mean()), 2) if len(h) else None}
    if ci is not None:
        for _, r in ci.iterrows():
            summary[f"{r['ethnicity']} positivity vs the other (%)"] = round(r["positivity"], 2)
            summary[f"{r['ethnicity']} 95% CI"] = [round(r["ci_lo"], 2), round(r["ci_hi"], 2)]
    return counts, ci, summary


def diagnostics(df):
    gen = df[df["gen_neg"] != ""]
    d = {"contexts scored": int(len(df)),
         "share of picks = option (a)": round(float(np.r_[df["pick_neg"], df["pick_pos"]].mean()), 4),
         "same pick for neg and pos question": round(float((df["pick_neg"] == df["pick_pos"]).mean()), 4),
         "ties": int(((df["pick_neg"] == 0.5) | (df["pick_pos"] == 0.5)).sum())}
    if len(gen):
        for q, col, gcol in (("neg", "pick_neg", "gen_neg"), ("pos", "pick_pos", "gen_pos")):
            g = gen[gcol].str.strip().str.lower()
            is_a, is_b = g == gen["subj0"].str.lower(), g == gen["subj1"].str.lower()
            d[f"{q}: greedy output is exactly one option"] = round(float((is_a | is_b).mean()), 4)
            agree = (is_a & (gen[col] == 1)) | (is_b & (gen[col] == 0))
            d[f"{q}: greedy output agrees with LL choice"] = round(float(agree.sum() / max(1, (is_a | is_b).sum())), 4)
        d["greedy rows checked"] = int(len(gen))
    return d


# ---------- figures ----------

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def errdots(ax, x, y, lo, hi, color, marker, label, size=7):
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt=marker, color=color, ms=size, mec=SURFACE, mew=1.5,
                ecolor=color, elinewidth=1.6, capsize=3.5, capthick=1.6, zorder=3, label=label)


def top_legend(ax, ncol=2):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, frameon=False, labelcolor=INK2, fontsize=9,
              borderaxespad=0.3)


def save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")


POS, NEG = "#2a78d6", "#e34948"   # diverging poles: above / below 50%


def devbars(ax, y, v, lo, hi, colors=None, height=0.62):
    """Horizontal bars that start at 50% (no bias), with 95% CI error bars."""
    colors = colors or [POS if x >= 50 else NEG for x in v]
    ax.barh(y, v - 50, left=50, height=height, color=colors, edgecolor=SURFACE, linewidth=1, zorder=2)
    ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, capthick=1.2, zorder=3)
    ax.axvline(50, color=INK2, linewidth=1, zorder=2)


def value_labels(ax, y, v, lo, hi):
    for yi, xi, l, h in zip(y, v, lo, hi):
        right = xi >= 50
        ax.text(h + 0.6 if right else l - 0.6, yi, f"{xi:.1f}", va="center", ha="left" if right else "right",
                fontsize=8, color=INK2)


def paper_marks(ax, x, y):
    ax.scatter(x, y, s=46, marker="D", facecolor=SURFACE, edgecolor=INK, linewidth=1.4, zorder=4)


def bar_legend(ax, paper=True):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(color=POS, label="ours: more often linked to positive answers"),
         Patch(color=NEG, label="ours: more often linked to negative answers"),
         Line2D([], [], color=INK2, marker="|", markersize=8, linestyle="-", label="95% CI")]
    if paper:
        h.append(Line2D([], [], marker="D", color=INK, markerfacecolor=SURFACE, linestyle="none",
                        label="Dodge et al. 2021, Table 7"))
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=INK2,
              fontsize=8.5, borderaxespad=0.3)


def symmetric_xlim(ax, *arrays, pad=5):
    lim = max(np.nanmax(np.abs(np.asarray(a, dtype=float) - 50)) for a in arrays) + pad
    ax.set_xlim(50 - lim, 50 + lim)


def fig_table7(t, title, path, dpi, plt):
    """Paper-style horizontal bars: positivity per group, bars start at 50% (no bias)."""
    t = t.dropna(subset=["dodge_table7"]).sort_values("positivity")
    y = np.arange(len(t))
    v, lo, hi = t["positivity"].values, t["ci_lo"].values, t["ci_hi"].values
    fig, ax = plt.subplots(figsize=(8, 6.8), facecolor=SURFACE)
    style(ax)
    devbars(ax, y, v, lo, hi)
    paper_marks(ax, t["dodge_table7"].values, y)
    value_labels(ax, y, v, lo, hi)
    ax.set_yticks(y, t["ethnicity"], color=INK)
    symmetric_xlim(ax, lo, hi, t["dodge_table7"])
    ax.set_xlabel("Positivity (%): share of questions where the group gets the positive association (50 = no bias)",
                  color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12, pad=44)
    bar_legend(ax)
    save(fig, path, dpi)
    return fig


def fig_scatter(t, title, path, dpi, plt):
    """Reproduction vs paper, one point per group; the dashed diagonal is perfect agreement."""
    t = t.dropna(subset=["dodge_table7"])
    x, v = t["dodge_table7"].values, t["positivity"].values
    lo = min(x.min(), t["ci_lo"].min()) - 3
    hi = max(x.max(), t["ci_hi"].max()) + 3
    fig, ax = plt.subplots(figsize=(6.8, 6.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.axhline(50, color=GRID, linewidth=1, zorder=1)
    ax.axvline(50, color=GRID, linewidth=1, zorder=1)
    ax.errorbar(x, v, yerr=[v - t["ci_lo"].values, t["ci_hi"].values - v], fmt="o", color=POS, ms=7, mec=SURFACE,
                mew=1.5, ecolor=POS, elinewidth=1.4, capsize=3, zorder=3)
    # labels to the right of each point; crowded labels are pushed down and joined by a leader line
    gap = (hi - lo) / 38
    placed = []
    for e, xi, yi in sorted(zip(t["ethnicity"], x, v), key=lambda r: -r[2]):
        key = e in ("Jewish", "Arab")
        ly = yi
        while any(abs(xi - px) < 9 and abs(ly - py) < gap for px, py in placed):
            ly -= gap / 2
        placed.append((xi, ly))
        ax.annotate(e, (xi, yi), xytext=(xi + (hi - lo) / 60, ly), textcoords="data", va="center",
                    fontsize=8.5 if key else 7.5, color=INK if key else INK2, fontweight="bold" if key else "normal",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6, shrinkA=0, shrinkB=4)
                    if abs(ly - yi) > gap / 3 else None)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Dodge et al. 2021, Table 7 positivity (%)", color=INK2)
    ax.set_ylabel("This reproduction (%), 95% CI", color=INK2)
    rho = pd.Series(v).rank().corr(pd.Series(x).rank())
    r = np.corrcoef(x, v)[0, 1]
    ax.set_title(f"{title}\nSpearman {rho:.2f}, Pearson {r:.2f}; dashed line = perfect agreement",
                 color=INK, loc="left", fontsize=11)
    save(fig, path, dpi)
    return fig


def fig_opponents(pwc, order, title, path, dpi, plt, groups=("Jewish", "Arab")):
    """For each focus group: positivity against every other group (a readable slice of the heatmap)."""
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 6.2), facecolor=SURFACE)
    lim = [a for g in groups for a in (pwc.loc[pwc["eth"] == g, "ci_lo"], pwc.loc[pwc["eth"] == g, "ci_hi"])]
    for ax, g in zip(np.atleast_1d(axes), groups):
        d = pwc[pwc["eth"] == g].set_index("other").reindex([o for o in order[::-1] if o != g]).dropna()
        y = np.arange(len(d))
        style(ax)
        devbars(ax, y, d["positivity"].values, d["ci_lo"].values, d["ci_hi"].values)
        value_labels(ax, y, d["positivity"].values, d["ci_lo"].values, d["ci_hi"].values)
        ax.set_yticks(y, [f"vs {o}" for o in d.index], color=INK)
        symmetric_xlim(ax, *lim, pad=9)
        ax.set_title(g, color=INK, loc="left", fontsize=12, pad=30)
        ax.set_xlabel(f"{g} positivity (%) in questions against that group", color=INK2)
    bar_legend(np.atleast_1d(axes)[0], paper=False)
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=12)
    save(fig, path, dpi)
    return fig


def fig_models(combined, cis, path, dpi, plt):
    t = combined.sort_values("dodge_table7")
    y = np.arange(len(t))
    names = sorted(cis, key=size_key)
    k = len(names) + 1
    off = np.linspace(-0.3, 0.3, k)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor=SURFACE)
    style(ax)
    ax.axvline(50, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.scatter(t["dodge_table7"], y + off[0], s=50, marker="D", color=C_PAPER, edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label="Dodge et al. 2021, Table 7")
    for i, name in enumerate(names[:4]):
        c = cis[name].set_index("ethnicity").reindex(t["ethnicity"])
        errdots(ax, c["positivity"].values, y + off[i + 1], c["ci_lo"].values, c["ci_hi"].values,
                C_MODELS[i], M_MODELS[i], f"{name} (95% CI)", size=6)
    ax.set_yticks(y, t["ethnicity"], color=INK)
    ax.set_xlabel("Positivity (%)", color=INK2)
    ax.set_title("Table 7: paper vs UnifiedQA sizes", color=INK, loc="left", fontsize=12, pad=30)
    top_legend(ax, ncol=k)
    save(fig, path, dpi)
    return fig


def fig_jewish_vs_arab(t7, ci_h2h, title, path, dpi, plt):
    rows = []
    for eth in ("Jewish", "Arab"):
        r = t7.set_index("ethnicity").loc[eth]
        rows.append((f"{eth}: vs all 14 groups (Table 7)", r["positivity"], r["ci_lo"], r["ci_hi"], TABLE7[eth]))
        if ci_h2h is not None:
            h = ci_h2h.set_index("ethnicity").loc[eth]
            rows.append((f"{eth}: vs {'Arab' if eth == 'Jewish' else 'Jewish'} only", h["positivity"], h["ci_lo"], h["ci_hi"], np.nan))
    rows = rows[::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    style(ax)
    ax.axvline(50, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    x = np.array([r[1] for r in rows])
    errdots(ax, x, y, np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
            C_MODELS[0], "o", "This reproduction (95% CI)")
    py = [i for i, r in enumerate(rows) if not np.isnan(r[4])]
    ax.scatter([rows[i][4] for i in py], py, s=55, marker="D", color=C_PAPER, edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label="Dodge et al. 2021, Table 7")
    for yi, xi, hi in zip(y, x, [r[3] for r in rows]):
        ax.annotate(f"{xi:.1f}", (hi, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], color=INK)
    ax.set_xlabel("Positivity (%)", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12, pad=30)
    top_legend(ax)
    save(fig, path, dpi)
    return fig


def fig_by_class(bc, title, path, dpi, plt):
    """Jewish vs Arab per UnQover attribute class: grouped bars from 50%, coloured by group."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rows = [c for c in ATTR_CLASSES if c in set(bc["attr_cls"])][::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    style(ax)
    arrays = []
    for eth, color, dy in (("Jewish", C_MODELS[0], 0.19), ("Arab", C_PAPER, -0.19)):
        c = bc[bc["eth"] == eth].set_index("attr_cls").reindex(rows)
        devbars(ax, y + dy, c["positivity"].values, c["ci_lo"].values, c["ci_hi"].values,
                colors=[color] * len(rows), height=0.36)
        arrays += [c["ci_lo"], c["ci_hi"]]
    ax.set_yticks(y, rows, color=INK)
    symmetric_xlim(ax, *arrays)
    ax.set_xlabel("Positivity (%) against all other Table 7 groups (50 = no bias)", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12, pad=30)
    ax.legend(handles=[Patch(color=C_MODELS[0], label="Jewish"), Patch(color=C_PAPER, label="Arab"),
                       Line2D([], [], color=INK2, marker="|", markersize=8, label="95% CI")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, labelcolor=INK2, fontsize=9)
    save(fig, path, dpi)
    return fig


def fig_by_attribute(ba, title, path, dpi, plt):
    """Per attribute: rank agreement with Table 7 (left) and how often the answer flips with sentiment (right)."""
    from matplotlib.patches import Patch
    t = ba.sort_values("spearman_vs_table7")
    y = np.arange(len(t))
    colors = [NEG if n else POS for n in t["positive_question_is_negation"]]
    fig, (axr, axf) = plt.subplots(1, 2, figsize=(12, 0.24 * len(t) + 2), sharey=True, facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [3, 2]})
    for ax in (axr, axf):
        style(ax)
    axr.scatter(t["spearman_vs_table7"], y, s=26, color=colors, zorder=3)
    axr.axvline(0, color=INK2, linewidth=1)
    axr.set_xlim(-1, 1)
    axr.set_xlabel("Spearman of the 15 group positivities (this attribute only) with Table 7", color=INK2)
    axf.barh(y, t["answer_flip_%"], height=0.7, color=colors, edgecolor=SURFACE, zorder=2)
    axf.set_xlim(0, 100)
    axf.set_xlabel("Contexts where the answer flips with sentiment (%)", color=INK2)
    axr.set_yticks(y, [a if len(a) < 48 else a[:46] + "…" for a in t["attr"]], color=INK, fontsize=7)
    axr.legend(handles=[Patch(color=POS, label="positive question is a rewrite"),
                        Patch(color=NEG, label="positive question is a negation (not / n't / never)")],
               loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=INK2, fontsize=8.5)
    fig.suptitle(f"{title}\nFew contexts per attribute on a sample: read the pattern, not single rows",
                 color=INK, x=0.01, ha="left", fontsize=11)
    save(fig, path, dpi)
    return fig


def fig_pairwise(m, title, path, dpi, plt):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=SURFACE)
    im = ax.imshow(m.values, cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=60, ha="right", color=INK2)
    ax.set_yticks(range(len(m.index)), m.index, color=INK)
    ax.set_xlabel("against", color=INK2)
    ax.set_ylabel("group", color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(len(m.index)):
        for j in range(len(m.columns)):
            v = m.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("positivity of row group vs column group (%)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(title + "  (95% CIs: pairwise_ci.csv)", color=INK, loc="left", fontsize=12)
    save(fig, path, dpi)
    return fig


SIZE_ORDER = ["small", "base", "large", "3b", "11b"]


def size_key(name):
    parts = name.split("-")          # also matches variants such as unifiedqa-t5-3b-int8
    for i, k in enumerate(SIZE_ORDER):
        if k in parts:
            return i
    return len(SIZE_ORDER)


def fig_size_trend(cis, diags, path, dpi, plt):
    """How the reproduction changes with UnifiedQA size."""
    names = sorted(cis, key=size_key)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)

    ax = axes[0]
    rho = [cis[n]["positivity"].rank().corr(cis[n]["dodge_table7"].rank()) for n in names]
    ax.plot(x, rho, color=POS, linewidth=2, marker="o", ms=7, mec=SURFACE, mew=1.5)
    for xi, r in zip(x, rho):
        ax.annotate(f"{r:.2f}", (xi, r), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylim(-1, 1.1)
    ax.set_title("Rank agreement with Table 7 (Spearman)", color=INK, loc="left", fontsize=11)

    ax = axes[1]
    for eth, color, dx in (("Jewish", C_MODELS[0], -0.06), ("Arab", C_PAPER, 0.06)):
        r = [cis[n].set_index("ethnicity").loc[eth] for n in names]
        v = np.array([i["positivity"] for i in r])
        ax.errorbar(x + dx, v, yerr=[v - np.array([i["ci_lo"] for i in r]), np.array([i["ci_hi"] for i in r]) - v],
                    color=color, linewidth=2, marker="o", ms=7, mec=SURFACE, mew=1.5, capsize=3,
                    label=f"{eth}, ours (95% CI)")
        ax.axhline(TABLE7[eth], color=color, linewidth=1, linestyle=(0, (3, 3)), label=f"{eth}, paper ({TABLE7[eth]})")
    ax.axhline(50, color=MUTED, linewidth=1)
    ax.set_title("Jewish and Arab positivity (%)", color=INK, loc="left", fontsize=11)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK2, fontsize=8)

    ax = axes[2]
    flip = [100 * (1 - diags[n]["same pick for neg and pos question"]) for n in names]
    ax.plot(x, flip, color=POS, linewidth=2, marker="o", ms=7, mec=SURFACE, mew=1.5)
    for xi, f in zip(x, flip):
        ax.annotate(f"{f:.0f}%", (xi, f), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_title("Contexts where the answer flips with sentiment (%)", color=INK, loc="left", fontsize=11)

    for ax in axes:
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_xticks(x, [n.replace("unifiedqa-t5-", "") for n in names], color=INK)
        ax.set_xlabel("UnifiedQA (T5) size", color=INK2)
    save(fig, path, dpi)
    return fig


def ethnicity_counts(df, long, B, seed):
    """Per ethnicity: how often the model answered it on positive and on negative questions (A.7's two cases).

    positive associations = answered on a positive question + the other group answered on a negative question;
    positivity = positive associations / questions mentioning the group (the Table 7 value).
    """
    rows = {}
    for side, other in (("subj0", "subj1"), ("subj1", "subj0")):
        took_pos = df["pick_pos"] if side == "subj0" else 1 - df["pick_pos"]   # 1 = model answered this group
        took_neg = df["pick_neg"] if side == "subj0" else 1 - df["pick_neg"]
        g = pd.DataFrame({"eth": df[side], "took_pos": took_pos, "took_neg": took_neg}).groupby("eth").agg(
            contexts=("took_pos", "size"), answered_pos=("took_pos", "sum"), answered_neg=("took_neg", "sum"))
        for e, r in g.iterrows():
            acc = rows.setdefault(e, {"contexts": 0, "answered_pos": 0.0, "answered_neg": 0.0})
            for k in acc:
                acc[k] += r[k]
    t = pd.DataFrame.from_dict(rows, orient="index").rename_axis("ethnicity").reset_index()
    t["positive_questions"] = t["contexts"]
    t["negative_questions"] = t["contexts"]
    t["positive_associations"] = t["answered_pos"] + (t["negative_questions"] - t["answered_neg"])
    t["negative_associations"] = t["answered_neg"] + (t["positive_questions"] - t["answered_pos"])
    t["questions"] = t["positive_questions"] + t["negative_questions"]
    t["positivity"] = 100 * t["positive_associations"] / t["questions"]
    t["answered_on_positive_%"] = 100 * t["answered_pos"] / t["positive_questions"]
    t["answered_on_negative_%"] = 100 * t["answered_neg"] / t["negative_questions"]
    # CIs for the two rates: in long_credits, credit = answered (positive question) or not answered (negative)
    ci = boot_ci(long, ["q", "eth"], B, seed)
    for q, col, flip in (("pos", "answered_on_positive_%", False), ("neg", "answered_on_negative_%", True)):
        c = ci[ci["q"] == q].set_index("eth")
        lo, hi = (100 - c["ci_hi"], 100 - c["ci_lo"]) if flip else (c["ci_lo"], c["ci_hi"])
        t[f"{col}_ci_lo"] = t["ethnicity"].map(lo)
        t[f"{col}_ci_hi"] = t["ethnicity"].map(hi)
    t["dodge_table7"] = t["ethnicity"].map(TABLE7)
    whole = ["questions", "positive_questions", "negative_questions"]
    t[whole] = t[whole].astype(int)
    if not (t[["answered_pos", "answered_neg"]] % 1).any().any():   # no ties: counts are whole numbers
        for c in ("answered_pos", "answered_neg", "positive_associations", "negative_associations"):
            t[c] = t[c].astype(int)
    cols = ["ethnicity", "questions", "positive_questions", "answered_pos", "negative_questions", "answered_neg",
            "positive_associations", "negative_associations", "positivity", "dodge_table7",
            "answered_on_positive_%", "answered_on_positive_%_ci_lo", "answered_on_positive_%_ci_hi",
            "answered_on_negative_%", "answered_on_negative_%_ci_lo", "answered_on_negative_%_ci_hi"]
    return t[cols].sort_values("positivity", ascending=False).reset_index(drop=True)


def fig_counts(counts, title, path, dpi, plt):
    """Butterfly chart: how often each group is the model's answer to negative (left) and positive (right) questions."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    t = counts.sort_values("positivity")
    y = np.arange(len(t))
    fig, (axn, axp) = plt.subplots(1, 2, figsize=(11, 0.42 * len(t) + 1.8), sharey=True, facecolor=SURFACE)
    for ax, col, color, label in ((axn, "answered_on_negative_%", NEG, "negative questions (e.g. 'Who was a hateful extremist?')"),
                                  (axp, "answered_on_positive_%", POS, "positive questions (e.g. 'Who was a peaceful resident?')")):
        style(ax)
        v = t[col].values
        ax.barh(y, v - 50, left=50, height=0.62, color=color, edgecolor=SURFACE, linewidth=1, zorder=2)
        ax.errorbar(v, y, xerr=[v - t[f"{col}_ci_lo"].values, t[f"{col}_ci_hi"].values - v], fmt="none",
                    ecolor=INK2, elinewidth=1.1, capsize=2.5, zorder=3)
        ax.axvline(50, color=INK2, linewidth=1, zorder=2)
        for yi, xi, lo_, hi_ in zip(y, v, t[f"{col}_ci_lo"].values, t[f"{col}_ci_hi"].values):
            ax.text(hi_ + 1.2 if xi >= 50 else lo_ - 1.2, yi, f"{xi:.0f}", va="center",
                    ha="left" if xi >= 50 else "right", fontsize=7.5, color=INK2)
        lim = np.nanmax(np.abs(np.r_[t[f"{col}_ci_lo"], t[f"{col}_ci_hi"]] - 50)) + 12
        ax.set_xlim(50 - lim, 50 + lim)
        ax.set_xlabel(f"% of {label.split(' (')[0]} answered with this group", color=INK2)
        ax.set_title(label, color=INK2, loc="left", fontsize=9)
    axn.set_yticks(y, [f"{e}  ({p:.0f}%)" for e, p in zip(t["ethnicity"], t["positivity"])], color=INK)
    fig.suptitle(f"{title}\nLabel = Table 7 positivity. A biased-positive group is rarely the answer on the left "
                 "and often on the right; 50 = no preference", color=INK, x=0.01, ha="left", fontsize=11)
    axn.legend(handles=[Patch(color=NEG, label="answered on negative questions"),
                        Patch(color=POS, label="answered on positive questions"),
                        Line2D([], [], color=INK2, marker="|", markersize=8, label="95% CI")],
               loc="upper left", bbox_to_anchor=(0, -0.09), ncol=3, frameon=False, labelcolor=INK2, fontsize=8.5)
    save(fig, path, dpi)
    return fig


def fig_opponents_all(pwc, order, title, path, dpi, plt):
    """Every group against every other group: one small panel per group (rows of the full heatmap, with CIs)."""
    n = len(order)
    cols = 5
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.6 * cols, 3.9 * rows_), facecolor=SURFACE)
    lim = np.nanmax(np.abs(np.r_[pwc["ci_lo"], pwc["ci_hi"]] - 50)) + 4
    for ax, g in zip(axes.flat, order):
        d = pwc[pwc["eth"] == g].set_index("other").reindex([o for o in order[::-1] if o != g]).dropna()
        yy = np.arange(len(d))
        style(ax)
        devbars(ax, yy, d["positivity"].values, d["ci_lo"].values, d["ci_hi"].values, height=0.7)
        ax.set_yticks(yy, d.index, color=INK2, fontsize=7)
        ax.set_xlim(50 - lim, 50 + lim)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(g, color=INK, loc="left", fontsize=10)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"{title}\nEach panel: that group's positivity (%) against each other group; bars start at 50 "
                 "(no bias); blue = favoured, red = disfavoured; 95% CIs", color=INK, x=0.01, ha="left", fontsize=12)
    save(fig, path, dpi)
    return fig


def table_png(df, title, path, dpi, plt):
    """Render a table as an image (for slides / the write-up)."""
    cell = [[f"{v:.1f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if isinstance(v, float) else str(v))
             for v in r] for r in df.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(1.25 * len(df.columns) + 1, 0.3 * (len(df) + 1) + 0.4), facecolor=SURFACE)
    ax.axis("off")
    tb = ax.table(cellText=cell, colLabels=list(df.columns), loc="upper center", cellLoc="center", bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.25)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID)
        cl.set_facecolor(SURFACE if r else "#f0efec")
        cl.get_text().set_color(INK if r else INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    save(fig, path, dpi)
    return fig


def paper_layout(t7):
    """Table 7 as printed in the paper (Ethnicity, Positivity, paper's order), with our value beside it."""
    t = t7.dropna(subset=["dodge_table7"]).sort_values("dodge_table7", ascending=False)
    return pd.DataFrame({"Ethnicity": t["ethnicity"],
                         "Positivity (paper)": [f"{v:.1f}%" for v in t["dodge_table7"]],
                         "Positivity (ours)": [f"{v:.1f}%" for v in t["positivity"]],
                         "95% CI (ours)": [f"{l:.1f} - {h:.1f}" for l, h in zip(t["ci_lo"], t["ci_hi"])]})


def md_table(df, floatfmt=1):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


# ---------- driver ----------

def analyze(scores_dir, q, args, plt):
    name = os.path.basename(os.path.normpath(scores_dir))
    out = os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    df = add_picks(load(scores_dir, q), args.rule)
    join_rep = check_join(df)
    if not args.no_merged:
        df.to_csv(os.path.join(out, "scores_merged.csv.gz"), index=False, compression="gzip")

    t7df = df[df["subj0"].isin(TABLE7) & df["subj1"].isin(TABLE7)]
    print(f"\n== {name}: {len(df)} contexts scored, {len(t7df)} in the Table 7 set (paper: 147,000)")
    cov_rep, problems = coverage(t7df, q[q["subj0"].isin(TABLE7) & q["subj1"].isin(TABLE7)])
    if problems:
        msg = f"{name}: coverage check failed: " + "; ".join(problems)
        if not args.allow_partial:
            raise SystemExit(msg + " (a GPU process probably stopped; re-run scoring, or pass --allow_partial)")
        print("WARNING:", msg, "- analysed with --allow_partial; do not report these numbers")
    if len(t7df) < 147000:
        print("note: a sample, not the full set; position effects cancel only approximately")

    long7 = long_credits(t7df)
    t7 = positivity_table(long7, args.B, args.seed)
    t7.to_csv(os.path.join(out, "table7.csv"), index=False)
    rho = t7["positivity"].rank().corr(t7["dodge_table7"].rank())
    with open(os.path.join(out, "table7.md"), "w", encoding="utf-8") as f:
        f.write(f"# Table 7 reproduction: {name} (rule={args.rule})\n\n{md_table(t7)}\n"
                f"Spearman rank correlation with the paper: {rho:.3f}\n")
    print(t7.round(1).to_string(index=False))
    print(f"Spearman rank correlation with Table 7: {rho:.3f}")

    long_all = long_credits(df)
    positivity_table(long_all, args.B, args.seed).to_csv(os.path.join(out, "table7_ext.csv"), index=False)

    # every ethnicity, not only Jewish and Arab: answers on positive vs negative questions
    counts = ethnicity_counts(t7df, long7, args.B, args.seed)
    assert np.allclose(counts.set_index("ethnicity")["positivity"],
                       t7.set_index("ethnicity")["positivity"].reindex(counts["ethnicity"])), "count/positivity mismatch"
    counts.round(2).to_csv(os.path.join(out, "counts_by_ethnicity.csv"), index=False)
    with open(os.path.join(out, "counts_by_ethnicity.md"), "w", encoding="utf-8") as f:
        f.write(f"# Answers per ethnicity on positive vs negative questions: {name} (rule={args.rule})\n\n"
                + md_table(counts.iloc[:, :10].round(2)) + "\n")
    ethnicity_counts(df, long_all, args.B, args.seed).round(2).to_csv(
        os.path.join(out, "counts_by_ethnicity_ext.csv"), index=False)

    ja, ci_h2h, ja_sum = jewish_vs_arab(df, args.B, args.seed)
    ja.to_csv(os.path.join(out, "jewish_vs_arab.csv"), index=False)
    with open(os.path.join(out, "jewish_vs_arab.md"), "w", encoding="utf-8") as f:
        f.write(f"# Jewish vs Arab head-to-head: {name}\n\n{md_table(ja, 0)}\n")
        f.write("\n".join(f"- {k}: {v}" for k, v in ja_sum.items()) + "\n")
    print("Jewish vs Arab head-to-head:", json.dumps(ja_sum))

    bc = boot_ci(long7, ["attr_cls", "eth"], args.B, args.seed)
    bc.round(2).to_csv(os.path.join(out, "by_attr_class.csv"), index=False)

    order = list(t7.sort_values("dodge_table7", ascending=False)["ethnicity"])
    pwc = boot_ci(long7, ["eth", "other"], args.B, args.seed)
    pwc.round(2).to_csv(os.path.join(out, "pairwise_ci.csv"), index=False)
    pw = pwc.pivot(index="eth", columns="other", values="positivity").reindex(index=order, columns=order)
    pw.round(2).to_csv(os.path.join(out, "pairwise.csv"))

    ba = by_attribute(t7df, long7)
    ba.round(3).to_csv(os.path.join(out, "by_attribute.csv"), index=False)
    plain = set(ba.loc[~ba["positive_question_is_negation"], "attr"])
    t7nn = positivity_table(long7[long7["attr"].isin(plain)], args.B, args.seed)
    t7nn.to_csv(os.path.join(out, "table7_no_negation.csv"), index=False)
    rho_nn = t7nn["positivity"].rank().corr(t7nn["dodge_table7"].rank())

    # diagnostics on the Table 7 set only, so runs on the extended file compare like with like
    diag = {"contexts scored (all groups in the file)": int(len(df)), **diagnostics(t7df)}
    diag["spearman_vs_table7"] = round(float(rho), 4)
    diag.update(dispersion(t7))
    diag["attributes with a negation-style positive question"] = int(len(ba) - len(plain))
    diag["spearman_vs_table7_without_negation_attributes"] = round(float(rho_nn), 4)
    diag.update({f"join check: {k}": v for k, v in join_rep.items()})
    diag.update({f"coverage: {k}": v for k, v in cov_rep.items()})
    diag["coverage ok"] = not problems
    diag.update({f"jewish_vs_arab: {k}": v for k, v in ja_sum.items()})
    with open(os.path.join(out, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=1)
    for k, v in diag.items():
        print(f"  {k}: {v}")

    if plt:
        sub = f"{name}, rule={args.rule}"
        for fig in (
            fig_table7(t7, f"Table 7 reproduction ({sub})", os.path.join(out, "fig_table7.png"), args.dpi, plt),
            fig_scatter(t7, f"Reproduction vs paper ({sub})", os.path.join(out, "fig_scatter_vs_paper.png"), args.dpi, plt),
            fig_counts(counts, f"How often each group is the model's answer ({sub})",
                       os.path.join(out, "fig_counts_by_ethnicity.png"), args.dpi, plt),
            fig_opponents_all(pwc, order, f"Every group against every other group ({sub})",
                              os.path.join(out, "fig_opponents_all_groups.png"), args.dpi, plt),
            table_png(counts[["ethnicity", "answered_pos", "answered_neg", "positive_associations",
                              "negative_associations", "questions", "positivity", "dodge_table7"]],
                      f"Answers per ethnicity ({sub})", os.path.join(out, "counts_by_ethnicity.png"), args.dpi, plt),
            fig_jewish_vs_arab(t7, ci_h2h, f"Jewish vs Arab ({sub}); head-to-head n = {ja_sum['contexts']} contexts"
                               + (", illustrative only" if ja_sum["illustrative only (fewer than 100 contexts)"] else ""),
                               os.path.join(out, "fig_jewish_vs_arab.png"), args.dpi, plt),
            fig_by_attribute(ba, f"Per stereotype attribute ({sub})", os.path.join(out, "fig_by_attribute.png"), args.dpi, plt),
            fig_opponents(pwc, order, f"Jewish and Arab against each other group ({sub})",
                          os.path.join(out, "fig_jewish_arab_opponents.png"), args.dpi, plt),
            fig_by_class(bc, f"Jewish vs Arab by attribute class ({sub})", os.path.join(out, "fig_jewish_arab_by_class.png"), args.dpi, plt),
            table_png(t7[["ethnicity", "positivity", "ci_lo", "ci_hi", "dodge_table7", "diff"]],
                      f"Table 7 reproduction ({sub}); Spearman vs paper {rho:.2f}", os.path.join(out, "table7.png"), args.dpi, plt),
            table_png(ja, f"Jewish vs Arab head-to-head ({sub})", os.path.join(out, "jewish_vs_arab.png"), args.dpi, plt),
            table_png(paper_layout(t7), f"Table 7 in the paper's layout and order ({sub})",
                      os.path.join(out, "table7_paper_layout.png"), args.dpi, plt),
        ):
            plt.close(fig)
        if args.heatmap:
            plt.close(fig_pairwise(pw, f"Head-to-head positivity ({sub})", os.path.join(out, "fig_pairwise.png"), args.dpi, plt))
    return name, t7, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="one folder of chunk_*.csv.gz per model")
    ap.add_argument("--questions", default="data/questions/ethnicity_ext.jsonl.gz")
    ap.add_argument("--out_dir", default="results/analysis")
    ap.add_argument("--rule", default="sum", choices=["sum", "mean"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--heatmap", action="store_true", help="also draw the full pairwise heatmap")
    ap.add_argument("--no_merged", action="store_true", help="skip writing scores_merged.csv.gz")
    ap.add_argument("--allow_partial", action="store_true",
                    help="analyse a run that fails the coverage check (monitoring only, not for reporting)")
    args = ap.parse_args()

    plt = None
    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    pd.set_option("display.width", 160)
    q = pd.read_json(args.questions, lines=True, compression="gzip")
    os.makedirs(args.out_dir, exist_ok=True)

    combined = pd.DataFrame({"ethnicity": list(TABLE7), "dodge_table7": list(TABLE7.values())})
    cis, diags = {}, {}
    for d in args.scores:
        name, t7, diag = analyze(d, q, args, plt)
        cis[name], diags[name] = t7, diag
        combined = combined.merge(t7[["ethnicity", "positivity"]].rename(columns={"positivity": name}),
                                  on="ethnicity", how="left")
    combined.round(2).to_csv(os.path.join(args.out_dir, "table7_all_models.csv"), index=False)
    if plt:
        plt.close(fig_models(combined, cis, os.path.join(args.out_dir, "fig_table7_all_models.png"), args.dpi, plt))
        if len(cis) > 1:
            plt.close(fig_size_trend(cis, diags, os.path.join(args.out_dir, "fig_size_trend.png"), args.dpi, plt))
        plt.close(table_png(combined.round(1), f"Table 7: paper vs models (rule={args.rule})",
                            os.path.join(args.out_dir, "table7_all_models.png"), args.dpi, plt))
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
