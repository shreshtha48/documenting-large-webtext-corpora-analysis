"""Arm A summary figures: one question per figure, little text on the plot, interpretation in a captions file.

Reads the analysis that analyze_arm_a.py wrote for the full 3B run (E6) and the model-size test (E4), and writes
into --out_dir (default results/figures_arm_a/):
    A1_table7_paper_vs_ours.png   dumbbell: the paper's Table 7 value and ours (95% CI) for every group
    A2_answer_rates.png           how often each group is the answer to positive vs negative questions;
                                  the diagonal is "no bias", above it the group is favoured
    A3_model_size.png             rank agreement with the paper and answer flips, by UnifiedQA size
    A4_jewish_vs_arab.png         the 1,400 Jewish-vs-Arab contexts: who gets the negative / positive trait
    A5_head_to_head.png           every group against every other group (row group's positivity)
    README.md                     captions with the numbers, for the write-up

Usage: python scripts/make_figures_arm_a.py
"""
import argparse, json, os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

# reference palette (dataviz skill; Jewish/Arab pair validated for CVD separation)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIVERGING = ["#e34948", "#f0efec", "#2a78d6"]
SIZES = ["small", "base", "large", "3b"]


def style(ax, grid="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def title(ax, main, sub=None):
    ax.set_title(main, loc="left", color=INK, fontsize=12, pad=24 if sub else 10)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def save(fig, out, name, dpi):
    fig.savefig(os.path.join(out, name), dpi=dpi, facecolor=SURFACE, bbox_inches="tight")


def a1_dumbbell(t7, out, dpi, plt):
    t = t7.sort_values("dodge_table7").reset_index(drop=True)
    y = np.arange(len(t))
    off = (t["dodge_table7"] < t["ci_lo"]) | (t["dodge_table7"] > t["ci_hi"])
    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor=SURFACE)
    style(ax)
    ax.axvline(50, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.hlines(y, t["dodge_table7"], t["positivity"], color=GRID, linewidth=3, zorder=2)
    ax.hlines(y, t["ci_lo"], t["ci_hi"], color=BLUE, linewidth=1.4, alpha=0.6, zorder=3)
    ax.scatter(t["positivity"], y, s=48, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4, label="Ours (95% CI)")
    ax.scatter(t["dodge_table7"], y, s=44, marker="D", facecolor=SURFACE, edgecolor=INK, linewidth=1.3, zorder=5,
               label="Dodge et al. 2021")
    ax.set_yticks(y, t["ethnicity"])
    for lab, o in zip(ax.get_yticklabels(), off):
        lab.set_color(INK if o else INK2)
        lab.set_fontweight("bold" if o else "normal")
    ax.set_xlabel("Positivity (%): share of questions where the group gets the positive association", color=INK2,
                  fontsize=9)
    ax.text(50.4, len(t) - 0.4, "no bias", color=MUTED, fontsize=8, va="center")
    rho = t["positivity"].rank().corr(t["dodge_table7"].rank())
    title(ax, f"UnifiedQA-3B reproduces the paper's ranking (Spearman {rho:.2f})",
          "Bold: the paper's value lies outside our 95% CI")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    save(fig, out, "A1_table7_paper_vs_ours.png", dpi)
    plt.close(fig)
    return rho, list(t.loc[off, "ethnicity"])


def a2_answer_rates(counts, out, dpi, plt):
    x, y = counts["answered_on_negative_%"].to_numpy(), counts["answered_on_positive_%"].to_numpy()
    lo, hi = min(x.min(), y.min()) - 6, max(x.max(), y.max()) + 6
    fig, ax = plt.subplots(figsize=(6.6, 6.4), facecolor=SURFACE)
    style(ax, grid=None)
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=BLUE, alpha=0.05, linewidth=0)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=DIVERGING[0], alpha=0.05, linewidth=0)
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.scatter(x, y, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    placed = []
    for e, xi, yi in sorted(zip(counts["ethnicity"], x, y), key=lambda r: -r[2]):
        ly = yi
        while any(abs(xi - px) < 7 and abs(ly - py) < 1.8 for px, py in placed):
            ly -= 1.8
        placed.append((xi, ly))
        key = e in ("Jewish", "Arab")
        ax.annotate(e, (xi, yi), xytext=(xi + 0.9, ly), textcoords="data", va="center", fontsize=8.5,
                    color=INK if key else INK2, fontweight="bold" if key else "normal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.text(lo + 1, hi - 1.5, "favoured", color=BLUE, fontsize=9, va="top")
    ax.text(hi - 1, lo + 1.5, "disfavoured", color=DIVERGING[0], fontsize=9, ha="right")
    ax.set_xlabel("Picked as the answer to a negative question (%)", color=INK2, fontsize=9)
    ax.set_ylabel("Picked as the answer to a positive question (%)", color=INK2, fontsize=9)
    title(ax, "Bias = picked for the good trait, not for the bad one", "Dashed diagonal: no bias")
    save(fig, out, "A2_answer_rates.png", dpi)
    plt.close(fig)


def a3_model_size(e4_dir, e6_diag, out, dpi, plt):
    rows = []
    for s in SIZES:
        d = os.path.join(e4_dir, f"unifiedqa-t5-{s}")
        if os.path.exists(os.path.join(d, "diagnostics.json")):
            g = json.load(open(os.path.join(d, "diagnostics.json")))
            rows.append({"size": s, "rho": g["spearman_vs_table7"],
                         "flip": 100 * (1 - g["same pick for neg and pos question"])})
    r = pd.DataFrame(rows)
    x = np.arange(len(r))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), facecolor=SURFACE)
    for ax, col, ttl, lim in ((axes[0], "rho", "Agreement with Table 7 (Spearman)", (-0.2, 1)),
                              (axes[1], "flip", "Answer changes with the question (%)", (0, 60))):
        style(ax, grid="y")
        ax.plot(x, r[col], color=BLUE, linewidth=2, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=3)
        for i, (xi, v) in enumerate(zip(x, r[col])):
            last = col == "rho" and i == len(r) - 1 and e6_diag     # keep clear of the full-set diamond
            ax.annotate(f"{abs(v) if abs(v) < 0.005 else v:.2f}" if col == "rho" else f"{v:.0f}", (xi, v),
                        xytext=(0, -15 if last else 8), textcoords="offset points", ha="center", fontsize=8.5,
                        color=INK2)
        ax.set_xticks(x, ["small\n60M", "base\n220M", "large\n770M", "3B"][:len(r)])
        ax.set_ylim(*lim)
        ax.set_title(ttl, loc="left", color=INK, fontsize=10.5)
    if e6_diag:
        i3 = len(r) - 1
        axes[0].scatter([i3], [e6_diag["spearman_vs_table7"]], s=60, marker="D", facecolor=SURFACE, edgecolor=INK,
                        linewidth=1.3, zorder=4)
        axes[0].annotate(f"full set {e6_diag['spearman_vs_table7']:.2f}", (i3, e6_diag["spearman_vs_table7"]),
                         xytext=(-10, 4), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK)
        axes[0].text(0, 0.93, "dots: 1,516-context sample", transform=axes[0].transAxes, fontsize=8, color=INK2)
    axes[1].axhline(50, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    fig.suptitle("Only the 3B model answers by the question's sentiment, and only it reproduces the paper",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, out, "A3_model_size.png", dpi)
    plt.close(fig)
    return r


def a4_jewish_arab(ja, out, dpi, plt):
    ja = ja.set_index("question")
    rows = [("Negative question\n(e.g. 'Who was a hateful extremist?')", ja.loc["negative question"]),
            ("Positive question\n(e.g. 'Who was a peaceful resident?')", ja.loc["positive question"])]
    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor=SURFACE)
    style(ax, grid=None)
    for i, (lab, r) in enumerate(rows[::-1]):
        n = r["answered Jewish"] + r["answered Arab"]
        pj, pa = 100 * r["answered Jewish"] / n, 100 * r["answered Arab"] / n
        ax.barh(i, pj, color=BLUE, height=0.55, edgecolor=SURFACE, linewidth=2)
        ax.barh(i, pa, left=pj, color=ORANGE, height=0.55, edgecolor=SURFACE, linewidth=2)
        ax.text(pj / 2, i, f"Jewish {pj:.0f}%", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
        ax.text(pj + pa / 2, i, f"Arab {pa:.0f}%", ha="center", va="center", color="white", fontsize=9.5,
                fontweight="bold")
    ax.set_yticks([0, 1], [l for l, _ in rows[::-1]], fontsize=9)
    ax.set_xlim(0, 100)
    ax.axvline(50, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    n = int(rows[0][1]["contexts"])
    title(ax, "Asked to choose between a Jewish and an Arab person, the model puts the bad trait on the Arab one",
          f"Who the model picked, {n:,} contexts naming both groups (both orders)")
    save(fig, out, "A4_jewish_vs_arab.png", dpi)
    plt.close(fig)


def a5_heatmap(pw, t7, out, dpi, plt):
    from matplotlib.colors import LinearSegmentedColormap
    order = list(t7.sort_values("positivity", ascending=False)["ethnicity"])
    m = pw.reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(7.4, 6.4), facecolor=SURFACE)
    im = ax.imshow(m.values, cmap=LinearSegmentedColormap.from_list("div", DIVERGING), vmin=20, vmax=80)
    ax.set_xticks(range(len(order)), order, rotation=55, ha="right", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(order)), order, fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("against", color=INK2, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    cb.set_label("row group's positivity vs column group (%)", color=INK2, fontsize=8.5)
    title(ax, "A consistent hierarchy: groups high in Table 7 win against almost everyone",
          "Rows and columns ordered by our Table 7 value; blue = row favoured, red = row disfavoured")
    save(fig, out, "A5_head_to_head.png", dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e6", default=os.path.join(RES, "E6_full_3b", "analysis_sum", "unifiedqa-t5-3b"))
    ap.add_argument("--e4", default=os.path.join(RES, "E4_size_test_sample", "analysis_sum"))
    ap.add_argument("--out_dir", default=os.path.join(RES, "figures_arm_a"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "normal"})
    os.makedirs(args.out_dir, exist_ok=True)

    t7 = pd.read_csv(os.path.join(args.e6, "table7.csv"))
    counts = pd.read_csv(os.path.join(args.e6, "counts_by_ethnicity.csv"))
    ja = pd.read_csv(os.path.join(args.e6, "jewish_vs_arab.csv"))
    pw = pd.read_csv(os.path.join(args.e6, "pairwise.csv"), index_col=0)
    diag = json.load(open(os.path.join(args.e6, "diagnostics.json")))

    rho, off = a1_dumbbell(t7, args.out_dir, args.dpi, plt)
    a2_answer_rates(counts, args.out_dir, args.dpi, plt)
    size = a3_model_size(args.e4, diag, args.out_dir, args.dpi, plt)
    a4_jewish_arab(ja, args.out_dir, args.dpi, plt)
    a5_heatmap(pw, t7, args.out_dir, args.dpi, plt)

    j = t7.set_index("ethnicity")
    c = counts.set_index("ethnicity")
    jr = ja.set_index("question")
    neg_arab = 100 * jr.loc["negative question", "answered Arab"] / jr.loc["negative question", "contexts"]
    pos_jew = 100 * jr.loc["positive question", "answered Jewish"] / jr.loc["positive question", "contexts"]
    f = lambda e: f"{j.loc[e, 'positivity']:.1f} [{j.loc[e, 'ci_lo']:.1f}, {j.loc[e, 'ci_hi']:.1f}] vs paper {j.loc[e, 'dodge_table7']:.1f}"
    caps = f"""# Arm A figures (UnifiedQA-3B, all 147,000 contexts = 294,000 questions; rule = summed log-likelihood)

**A1. Table 7, paper vs ours.** Rank agreement with the paper: Spearman {rho:.2f} (Pearson {diag['pearson_vs_table7']:.2f}).
The paper's two focus groups reproduce: Jewish {f('Jewish')}, Arab {f('Arab')}.
Groups where the paper lies outside our 95% CI: {', '.join(off)}. Our values are somewhat more spread out than the
paper's (noise-corrected SD {diag['sd_ours_noise_corrected']} vs {diag['sd_paper']}; slope {diag['slope_ours_on_paper']}).
CIs: bootstrap over the 50 stereotype attributes.

**A2. Where a group's score comes from.** Positivity is the average of "picked for the positive question" and "not
picked for the negative one". Groups above the diagonal are favoured. For example, Black is picked for
{c.loc['Black', 'answered_on_negative_%']:.0f}% of negative questions but {c.loc['Black', 'answered_on_positive_%']:.0f}% of positive ones; European {c.loc['European', 'answered_on_negative_%']:.0f}% and
{c.loc['European', 'answered_on_positive_%']:.0f}%. Distance *along* the diagonal is a different thing: how often the model picks that name at all,
whatever the question. Native American sits on the diagonal at {c.loc['Native American', 'answered_on_negative_%']:.0f}% / {c.loc['Native American', 'answered_on_positive_%']:.0f}%, so its near-50% Table 7
value hides a strong name preference that Table 7 cannot show; Middle-Eastern is the opposite (rarely picked
either way).

**A3. Model size.** On the 1,516-context sample, rank agreement rises from {size['rho'].iloc[0]:.2f} (small) to
{size['rho'].iloc[-1]:.2f} (3B), and the share of contexts where the answer changes between the negative and
positive question rises from {size['flip'].iloc[0]:.0f}% to {size['flip'].iloc[-1]:.0f}%. A model that gives the same answer to both questions
scores every group near 50%, so only the 3B model can show the paper's bias. Diamond: the full 3B run
(Spearman {diag['spearman_vs_table7']:.2f}).

**A4. Jewish vs Arab head to head** ({int(jr.loc['negative question', 'contexts']):,} contexts naming both). For negative questions the model picks the
Arab person {neg_arab:.0f}% of the time; for positive ones it picks the Jewish person {pos_jew:.0f}% of the time. The paper
reports only the pooled Table 7 values; this is the direct comparison behind them.

**A5. Head to head, every pair.** Each cell is the row group's positivity in the contexts that name the row and column
group. The near-monotone pattern (blue upper-left, red lower-right) shows that Table 7 reflects a consistent ordering,
not a few extreme pairings. Per-cell CIs: `pairwise_ci.csv`.
"""
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(caps)
    print(caps)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
