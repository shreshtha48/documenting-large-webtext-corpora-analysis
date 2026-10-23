"""Dodge et al. 2021 §4.2: Table 2 / Table 4 reproduction from query_infinigram.py output.

Writes into --out_dir:
    table2.csv / table2.md   per field: matched / queried, % with a 95% Wilson CI, the paper's %, difference
    fig_table2.png           ours (with CI) vs the paper per field, labels and inputs separated as in Table 2
    examples.md              up to 10 matched texts per label field (the paper's Table 5 shows XSum ones)
    positive_control.md      the 10 XSum summaries the paper lists in Table 5 as found in C4, located in our XSum
                             test file by their opening words and queried; all of them should be found

Our rates are lower bounds on the paper's (exact, case- and punctuation-sensitive matching; see
query_infinigram.py), so "ours below the paper" is expected; "ours above the paper" would be a real discrepancy.

Usage: python scripts/analyze_contamination.py --counts results/C1_infinigram_2000
"""
import argparse, glob, gzip, json, math, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmarks import PAPER  # noqa: E402
from query_infinigram import BENCH, score  # noqa: E402

# opening words of the paper's Table 5 summaries (enough to find each row in the XSum test set)
TABLE5_PREFIXES = ["The takeover of Bradford Bulls", "US presidential candidate Donald Trump has given",
                   "Two men who were sued over the Omagh bomb", "Grimsby fought back from two goals down",
                   "Doctors say a potential treatment for peanut allergy", "A breast surgeon who intentionally",
                   "Turkey has bombarded so-called Islamic State", "Peterborough United have signed forward",
                   "The first major trial to see if losing weight", "Villarreal central defender Eric Bailly"]
LABELS = {"boolq_passage": ("BoolQ passage (ours)", "input")}

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
C_OURS, C_PAPER = "#2a78d6", "#eb6834"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def table(counts_dir):
    rows, matched = [], {}
    for p in sorted(glob.glob(os.path.join(counts_dir, "counts_*.jsonl"))):
        field = os.path.basename(p)[7:-6]
        res = [json.loads(l) for l in open(p, encoding="utf-8")]
        ok = [r for r in res if r["error"] is None]
        k, n = sum(r["matched"] for r in ok), len(ok)
        lo, hi = wilson(k, n)
        name, part, pk, pn = PAPER.get(field, (*LABELS.get(field, (field, "input")), None, None))
        with gzip.open(os.path.join(BENCH, f"{field}.jsonl.gz"), "rt", encoding="utf-8") as f:
            total = sum(1 for _ in f)
        rows.append({"field": field, "dataset": name, "part": part, "texts_in_dataset": total, "queried": n,
                     "errors": len(res) - n, "matched": k, "matched_%": 100 * k / n if n else np.nan,
                     "ci_lo": lo, "ci_hi": hi, "paper_%": 100 * pk / pn if pk else np.nan,
                     "paper_matched/total": f"{pk}/{pn}" if pk else ""})
        matched[field] = [r["text"] for r in ok if r["matched"]]
    t = pd.DataFrame(rows)
    t["diff_vs_paper"] = t["matched_%"] - t["paper_%"]
    order = {f: i for i, f in enumerate(list(PAPER) + list(LABELS))}
    return t.sort_values("field", key=lambda s: s.map(order).fillna(99)).reset_index(drop=True), matched


def figure(t, out, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    t = t.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(t) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    v = t["matched_%"].to_numpy()
    ax.errorbar(v, y, xerr=[v - t["ci_lo"], t["ci_hi"] - v], fmt="o", color=C_OURS, ms=6, mec=SURFACE, mew=1.2,
                ecolor=C_OURS, elinewidth=1.4, capsize=2.5, zorder=3)
    ax.scatter(t["paper_%"], y, s=46, marker="D", facecolor=SURFACE, edgecolor=C_PAPER, linewidth=1.6, zorder=4)
    split = [i for i in range(1, len(t)) if t.loc[i, "part"] != t.loc[i - 1, "part"]]
    for i in split:
        ax.axhline(i - 0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(y, [f"{d}  ({p})" for d, p in zip(t["dataset"], t["part"])], color=INK, fontsize=8.5)
    ax.set_xlabel("Texts found verbatim in C4.EN (%)", color=INK2)
    ax.legend(handles=[Line2D([], [], marker="o", color=C_OURS, linestyle="none",
                              label="ours: infini-gram exact match, all of C4.EN (95% Wilson CI); a lower bound"),
                       Line2D([], [], marker="D", color=C_PAPER, markerfacecolor=SURFACE, linestyle="none",
                              label="Dodge et al. 2021, Table 4 (normalized for case and punctuation)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title("Table 2 reproduction: benchmark texts found in C4.EN", color=INK, loc="left", fontsize=12, pad=40)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_table2.png"), dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def positive_control():
    path = os.path.join(BENCH, "xsum.jsonl.gz")
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
    out = []
    for pre in TABLE5_PREFIXES:
        hit = [r for r in rows if r["text"].startswith(pre)]
        res = score(hit[0]) if hit else None
        out.append({"table5_summary_starts": pre + " ...", "in_our_xsum_test": bool(hit),
                    "found_in_c4": None if res is None else res["matched"],
                    "count": None if res is None else (res["counts"] or [None])[-1]})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True, help="folder written by query_infinigram.py")
    ap.add_argument("--out_dir", default=None, help="default: <counts>/analysis")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no_control", action="store_true")
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.counts, "analysis")
    os.makedirs(out, exist_ok=True)
    t, matched = table(args.counts)
    t.round(3).to_csv(os.path.join(out, "table2.csv"), index=False)
    show = t[["dataset", "part", "queried", "errors", "matched", "matched_%", "ci_lo", "ci_hi", "paper_%",
              "paper_matched/total"]].round(2)
    with open(os.path.join(out, "table2.md"), "w", encoding="utf-8") as f:
        f.write("# Table 2 reproduction (exact match in all of C4.EN via infini-gram; lower bounds)\n\n" + md_table(show))
    figure(t, out, args.dpi)
    with open(os.path.join(out, "examples.md"), "w", encoding="utf-8") as f:
        for field in (x for x in matched if t.set_index("field").loc[x, "part"] == "label"):
            f.write(f"## {field}\n\n" + "".join(f"- {s}\n" for s in matched[field][:10]) + "\n")
    print(show.to_string(index=False))
    if not args.no_control:
        pc = positive_control()
        if pc is not None:
            with open(os.path.join(out, "positive_control.md"), "w", encoding="utf-8") as f:
                f.write("# Positive control: the paper's Table 5 XSum summaries\n\n" + md_table(pc))
            print("\npositive control (paper Table 5):\n" + pc.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
