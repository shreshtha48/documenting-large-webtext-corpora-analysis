"""Extension of §4.2: contamination of popular safety, bias and capability benchmarks in C4.EN. Reads the counts
written by reproduction/benchmark-contamination/scripts/query_infinigram.py (run with --bench_dir data/benchmarks).

Per field: share of sampled texts found verbatim in C4.EN (lower bound), Wilson 95% CI, and the same by text length
(1-7, 8-14, 15+ words), because short generic items ("Who invented the Internet?") can match by chance.
Per group (see build_benchmarks.py): pooled rate. The "written after Apr 2019" group is the negative control: its
rate on texts of 8+ words estimates how often the method calls a text "in C4" when it cannot be.

Usage:
    python scripts/analyze.py --counts results/E1_infinigram_200 --out_dir results/E1_infinigram_200/analysis
"""
import argparse, glob, json, math, os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "fields.json")
GROUP_ORDER = ["web text, before Apr 2019", "written before Apr 2019", "written after Apr 2019 (negative control)"]
INK2, GRID, SURFACE = "#52514e", "#e1e0d9", "#fcfcfb"
C_GROUP = {GROUP_ORDER[0]: "#eb6834", GROUP_ORDER[1]: "#6250d6", GROUP_ORDER[2]: "#2a78d6"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def bucket(n):
    return "1-7 words" if n < 8 else "8-14 words" if n < 15 else "15+ words"


def rate_row(d, **keys):
    k, n = int(d["matched"].sum()), len(d)
    lo, hi = wilson(k, n)
    return {**keys, "matched": k, "n": n, "rate_%": 100 * k / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi}


def md(df):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    return "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join(
        "| " + " | ".join(str(x) for x in r) + " |" for r in d.itertuples(index=False)) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    meta = json.load(open(META))
    rows = []
    for p in sorted(glob.glob(os.path.join(args.counts, "counts_*.jsonl"))):
        field = os.path.basename(p)[7:-6]
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            rows.append({"field": field, "matched": r["matched"], "error": r["error"], "words": len(r["text"].split()),
                         "count": max(r["counts"]) if r["counts"] else None, "text": r["text"]})
    df = pd.DataFrame(rows)
    errors = df[df["error"].notna()].groupby("field").size().to_dict()
    df = df[df["error"].isna()].copy()
    df["matched"] = df["matched"].astype(bool)
    df["length"] = df["words"].map(bucket)
    for k in ("benchmark", "part", "group", "year", "source"):
        df[k] = df["field"].map(lambda f: meta[f][k])

    per_field = pd.DataFrame([rate_row(d, field=f, benchmark=d["benchmark"].iat[0], part=d["part"].iat[0],
                                       group=d["group"].iat[0], year=d["year"].iat[0])
                              for f, d in df.groupby("field")])
    long8 = pd.DataFrame([rate_row(d[d["words"] >= 8], field=f) for f, d in df.groupby("field")])
    per_field = per_field.merge(long8[["field", "matched", "n", "rate_%"]].rename(columns={
        "matched": "matched_8+", "n": "n_8+", "rate_%": "rate_8+_%"}), on="field")
    per_field["group"] = pd.Categorical(per_field["group"], GROUP_ORDER, ordered=True)
    per_field = per_field.sort_values(["group", "rate_%"], ascending=[True, False])
    per_field.round(2).to_csv(os.path.join(args.out_dir, "by_field.csv"), index=False)
    by_len = pd.DataFrame([rate_row(d, field=f, length=b) for (f, b), d in df.groupby(["field", "length"])])
    by_len.round(2).to_csv(os.path.join(args.out_dir, "by_field_and_length.csv"), index=False)
    groups = pd.DataFrame([rate_row(d, group=g, texts="all") for g, d in df.groupby("group")]
                          + [rate_row(d[d["words"] >= 8], group=g, texts="8+ words") for g, d in df.groupby("group")])
    groups.round(2).to_csv(os.path.join(args.out_dir, "by_group.csv"), index=False)
    # the negative control's matches, listed for inspection
    ctrl = df[(df["group"] == GROUP_ORDER[2]) & df["matched"]].sort_values("words", ascending=False)
    ctrl[["field", "words", "count", "text"]].to_csv(os.path.join(args.out_dir, "negative_control_matches.csv"), index=False)

    rep = ["# Popular benchmarks in C4.EN (exact match, infini-gram, lower bound)\n",
           f"Sampled texts per field: up to {df.groupby('field').size().max()}; query errors left out: {errors or 'none'}.\n",
           "## By group\n", md(groups),
           "## By field\n", md(per_field[["benchmark", "part", "group", "year", "matched", "n", "rate_%", "ci_lo", "ci_hi",
                                          "matched_8+", "n_8+", "rate_8+_%"]]),
           f"## Negative-control matches ({len(ctrl)}), longest first\n",
           md(ctrl[["field", "words", "count", "text"]].head(25).assign(text=lambda x: x["text"].str[:100]))]
    open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(rep))
    print("\n".join(rep[:6]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = per_field.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(t) + 1.5))
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    y = range(len(t))
    ax.barh(y, t["rate_%"], color=[C_GROUP[g] for g in t["group"]], height=0.62, edgecolor=SURFACE)
    ax.errorbar(t["rate_%"], y, xerr=[t["rate_%"] - t["ci_lo"], t["ci_hi"] - t["rate_%"]], fmt="none", ecolor=INK2,
                capsize=3, elinewidth=1.1)
    ax.set_yticks(list(y), t["benchmark"] + " · " + t["part"], fontsize=8.5, color=INK2)
    ax.set_xlabel("texts found verbatim in C4.EN (%; exact match, lower bound; 95% Wilson CI)", color=INK2)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_GROUP[g], label=g) for g in GROUP_ORDER], frameon=False, fontsize=8,
              loc="lower right")
    ax.set_title("Popular safety, bias and capability benchmarks in C4.EN", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_by_field.png"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
