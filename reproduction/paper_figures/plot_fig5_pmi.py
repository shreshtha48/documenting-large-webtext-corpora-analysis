"""Replicate Dodge et al. 2021 Figure 5 (§5.2): PMI between identity mentions and blocklist filtering.

Input: the one-shard pilot output pilots/shard_check_output.txt (shard c4-train.00000-of-01024 of
en.noblocklist, 384,171 documents; natural-log PMI at document level; 95% bootstrap CI over
documents, B=200). Only the 22 Table 6 terms are plotted; the pilot's extension terms (teen, girl,
boy) are left out because they are not in the paper's figure.

The paper prints Figure 5 without numbers, so this is a replication of the chart, not an overlay.

Usage: python reproduction/paper_figures/plot_fig5_pmi.py [--dpi 300]
"""
import argparse, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PILOT = os.path.join(ROOT, "pilots", "shard_check_output.txt")
EXTENSION_TERMS = {"teen", "girl", "boy"}
# category per term, as grouped in Dodge §5.2 footnote 21 (gender identity, sexual orientation, race, religion)
CATEGORY = {"lesbian": "sexual orientation", "gay": "sexual orientation", "homosexual": "sexual orientation",
            "bisexual": "sexual orientation", "heterosexual": "sexual orientation", "straight": "sexual orientation",
            "nonbinary": "gender identity", "trans": "gender identity", "woman": "gender identity",
            "man": "gender identity", "female": "gender identity", "male": "gender identity",
            "black": "race / ethnicity", "african american": "race / ethnicity", "latinx": "race / ethnicity",
            "asian": "race / ethnicity", "white": "race / ethnicity", "caucasian": "race / ethnicity",
            "european": "race / ethnicity", "jew": "religion", "muslim": "religion", "christian": "religion"}
# reference palette (dataviz skill, validated categorical slots 1-4)
COLORS = {"sexual orientation": "#2a78d6", "gender identity": "#eb6834", "race / ethnicity": "#1baf7a",
          "religion": "#6250d6"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def parse(path):
    rows = []
    pat = re.compile(r"^(\S.*?)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.-]+)\s+\[([\d.-]+),([\d.-]+)\]")
    for line in open(path, encoding="utf-8"):
        m = pat.match(line)
        if m and m.group(1) not in EXTENSION_TERMS:
            rows.append(dict(term=m.group(1), docs=int(m.group(2)), filtered=int(m.group(3)),
                             rate=float(m.group(4)), pmi=float(m.group(5)), lo=float(m.group(6)), hi=float(m.group(7))))
    return sorted(rows, key=lambda r: r["pmi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(HERE, "fig5_pmi_identity_blocklist.png"))
    args = ap.parse_args()
    rows = parse(PILOT)
    assert len(rows) == 22, f"expected the 22 Table 6 terms, got {len(rows)}"

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(rows))
    ax.barh(y, [r["pmi"] for r in rows], color=[COLORS[CATEGORY[r["term"]]] for r in rows], height=0.66,
            edgecolor=SURFACE, linewidth=1, zorder=2)
    ax.errorbar([r["pmi"] for r in rows], y, xerr=[[r["pmi"] - r["lo"] for r in rows], [r["hi"] - r["pmi"] for r in rows]],
                fmt="none", ecolor=INK2, elinewidth=1.1, capsize=2.5, zorder=3)
    for yi, r in zip(y, rows):
        ax.text(r["hi"] + 0.03, yi, f"{r['pmi']:.2f}  ({100 * r['rate']:.0f}% of {r['docs']:,} docs removed)",
                va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(list(y), [r["term"] for r in rows], color=INK)
    ax.set_xlim(0, 3.6)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_xlabel("PMI(identity term; filtered by blocklist), natural log", color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in COLORS.items()], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title("Figure 5 replication: identity mentions vs blocklist filtering\n"
                 "C4 en.noblocklist, one shard (384,171 docs), 95% bootstrap CI", color=INK, loc="left",
                 fontsize=11, pad=28)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
