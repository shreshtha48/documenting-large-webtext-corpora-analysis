"""Build the Kaggle notebook for Dodge et al. 2021 §5 (what the blocklist excluded) from the scripts here.

Kaggle rather than Colab for the main run: "Save Version -> Save & Run All" runs in the background with the
browser closed (up to 12 h), and the CPU-only run uses no GPU quota.

Writes, next to this file:
    f5_excluded_kaggle.ipynb   import into Kaggle (File -> Import Notebook)
    f5_excluded_cells.txt      the same cells as plain text

Usage: python kaggle/build_kaggle_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = ["dialect.py", "scan_noblocklist.py", "analyze_excluded.py"]
W = "/kaggle/working/f5"
OUT = "/kaggle/working/f5/results/S1_noblocklist_12_shards"


def md(text):
    return ("markdown", text.strip("\n"))


def code(text):
    return ("code", text.strip("\n"))


def script_cell(name):
    with open(os.path.join(ROOT, "scripts", name), encoding="utf-8") as f:
        return code(f"%%writefile {W}/scripts/{name}\n" + f.read())


CELLS = [
    md("""
# Dodge et al. 2021 §5: what the blocklist excluded from C4 (Kaggle, CPU)

| Paper | What it reports | Output |
|---|---|---|
| §5.1, A.6, **Fig. 6** | k-means (k=50) of 100,000 excluded docs; 16 of 50 clusters "largely sexual", 31% of docs | `fig6_clusters.png`, `clusters.csv` |
| §5.2, **Fig. 5**, **Table 6** | PMI between 22 identity regexes and exclusion | `fig5_pmi.png`, `fig5_pmi.csv`, `table6.md` |
| §5.2 manual check | 22% ("lesbian") / 36% ("gay") of 50 excluded docs are non-offensive or non-sexual | `inspect_lesbian_gay.csv` (to label by hand) |
| §5.3 | removal by dialect: AAE 42%, Hispanic 32%, WAE 6.2%, other 7.2%; C4.EN 97.8% WAE, 0.07% AAE, 0.09% Hispanic | `fig_dialects.png`, `dialects.csv` |

**Settings:** Accelerator **None** (CPU), Internet **on**. Then **Save Version -> Save & Run All**; the browser can be
closed. About 1-1.5 h with 4 workers (12 shards, each ~400 MB download + ~15 min CPU). When it finishes, download
`f5_S1_results.zip` from the Output tab.

The method, and what is ours rather than the paper's, is in the folder's README.md. A one-shard run (S0) already
reproduces all three parts; this is the 12-shard run (S1).

**Content warning:** excluded documents include pornographic and hateful text (the clustering sample and the
inspection sheet contain it).
"""),
    md("## 1. Setup: copy the dataset into a writable folder"),
    code(f"""
%%bash
SRC=$(dirname "$(dirname "$(find /kaggle/input -path '*scripts/scan_noblocklist.py' | head -1)")")
echo "project folder: $SRC"
test -f "$SRC/data/ldnoobw_en_25e679f.txt" || {{ echo "data/ldnoobw_en_25e679f.txt not found under $SRC"; exit 1; }}
mkdir -p {W}
cp -rn "$SRC"/. {W}/          # -n: never overwrite results already in /kaggle/working
pip install -q huggingface_hub
nproc; df -h /kaggle/working | tail -1
"""),
    md("## 2. Scripts (embedded from the repo, so the notebook runs the current code)"),
    *[script_cell(n) for n in SCRIPTS],
    md("""
## 3. Scan 12 shards of `en.noblocklist`

The same shard indices as Arm B's reproduction (seed 0). Each line reports the share excluded (about 7% expected)
and the fast-vs-exact dialect agreement (above 99% expected).
"""),
    code(f"""
%%bash
cd {W}
python scripts/scan_noblocklist.py --n 12 --seed 0 --workers 4 --cache_dir /kaggle/tmp/hf_cache \\
    --out_dir {OUT} 2>&1 | grep -v -i warning
"""),
    md("## 4. Analysis: Fig. 5 + Table 6, §5.3 dialects, Fig. 6 clusters (about 15 min)"),
    code(f"""
%%bash
cd {W}
export LOKY_MAX_CPU_COUNT=4
python scripts/analyze_excluded.py --scan {OUT} --out_dir {OUT}/analysis 2>&1 | grep -v -i warning
"""),
    md("## 5. Results"),
    code(f"""
import os
import pandas as pd
from IPython.display import Image, Markdown, display
A = "{OUT}/analysis"
show = lambda p, w=900: display(Image(p, width=w)) if os.path.exists(p) else print("missing:", p)
if os.path.exists(f"{{A}}/summary.md"):
    display(Markdown(open(f"{{A}}/summary.md").read().split("## ")[0]))
    show(f"{{A}}/fig5_pmi.png"); display(Markdown(open(f"{{A}}/table6.md").read()))
    display(pd.read_csv(f"{{A}}/dialects.csv").round(3)); show(f"{{A}}/fig_dialects.png", 1100)
    show(f"{{A}}/fig6_clusters.png", 1200)
    if os.path.exists(f"{{A}}/clusters.csv"):
        c = pd.read_csv(f"{{A}}/clusters.csv")
        display(pd.DataFrame([{{"seed": s, "sexual clusters": int(g.auto_sexual.sum()),
                               "share of docs %": g.loc[g.auto_sexual, "share_%"].sum()}} for s, g in c.groupby("seed")]))
else:
    print("analysis missing: check the output of sections 3 and 4")
"""),
    md("""
## 6. Download

`f5_S1_results.zip` (Output tab) holds the analysis, the per-shard counts and stats, and the manual-check sheet. The
excluded-document samples (large, explicit) are left out of the zip; they stay in this version's output folder.

**Manual check (§5.2):** open `analysis/inspect_lesbian_gay.csv` and fill `label` with `offensive`, `sexual` or
`neither` for each row. Send it back, and the share labelled `neither` is computed with a 95% CI (paper: 22% / 36%).
"""),
    code(f"""
%%bash
cd {OUT}
zip -qr /kaggle/working/f5_S1_results.zip analysis counts_*.npz stats_*.json shards.json
ls -la /kaggle/working/f5_S1_results.zip
"""),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": []}
    lines = []
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        nb["cells"].append(cell)
        lines.append(f"# ======================= CELL {i} ({kind}) =======================\n{src}\n")
    with open(os.path.join(HERE, "f5_excluded_kaggle.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, "f5_excluded_cells.txt"), "w", encoding="utf-8") as f:
        f.write("Paste each block into its own Kaggle cell (markdown blocks into markdown cells).\n"
                "Do not paste the '# ===== CELL' separator lines.\n\n" + "\n".join(lines))
    print(f"wrote {len(CELLS)} cells")


if __name__ == "__main__":
    main()
