"""Build the Google Colab notebook for Dodge et al. 2021 §5 (what the blocklist excluded) from the scripts here.

Writes, next to this file:
    f5_excluded_colab.ipynb   open in Colab (File -> Upload notebook)
    f5_excluded_cells.txt     the same cells as plain text, for copy-paste

The scripts are embedded with %%writefile, so the notebook runs the repo's current code. CPU only (free tier).
Re-run this builder after editing a script.

Usage: python colab/build_colab_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = ["dialect.py", "scan_noblocklist.py", "analyze_excluded.py"]
W = "/content/f5"   # working copy of reproduction/f5_excluded_data


def md(text):
    return ("markdown", text.strip("\n"))


def code(text):
    return ("code", text.strip("\n"))


def script_cell(name):
    with open(os.path.join(ROOT, "scripts", name), encoding="utf-8") as f:
        return code(f"%%writefile {W}/scripts/{name}\n" + f.read())


CELLS = [
    md("""
# Dodge et al. 2021 §5: what the blocklist excluded from C4

C4.EN drops every document that contains a word from the "List of Dirty, Naughty, Obscene, and Otherwise Bad Words".
§5 asks what that removed. This notebook reproduces all three parts on a free Colab CPU.

| Paper | What it reports | Output here |
|---|---|---|
| §5.1, A.6, **Fig. 6** | k-means (k=50) of 100,000 excluded docs (TF-IDF), t-SNE; 16 of 50 clusters "largely sexual", 31% of docs | `fig6_clusters.png`, `clusters.csv` |
| §5.2, **Fig. 5**, **Table 6** | PMI between 22 identity regexes and exclusion; sexual-orientation terms highest | `fig5_pmi.png`, `fig5_pmi.csv`, `table6.md` |
| §5.2 manual check | of 50 excluded docs each, 22% ("lesbian") and 36% ("gay") are non-offensive or non-sexual | `inspect_lesbian_gay.csv` (you label it), section 6 |
| §5.3 | removal rate by dialect: AAE 42%, Hispanic 32%, WAE 6.2%, other 7.2%; C4.EN is 97.8% WAE, 0.07% AAE, 0.09% Hispanic | `fig_dialects.png`, `dialects.csv` |

**How.** The excluded documents are the `en.noblocklist` documents that match C4's own bad-words regex (TFDS v3.0.1,
the version AI2 used); the paper defines them as "in C4.EN.NOBLOCKLIST but not in C4.EN", which is the same set.
One pass per shard records, for every document: excluded or not, the identity terms it mentions, and its dialect
(Blodgett et al. 2016, ported to Python 3; validated in every shard against an exact port). Counts are kept per web
domain, and the 95% CIs come from a bootstrap over domains.

**Not in the paper, ours:** TF-IDF and k-means settings, three k-means seeds, an automatic "sexual cluster" rule (with a
column for hand labels), the tokenizer for the dialect model, and all CIs. Each is stated in the scripts.

**Free tier.** 12 shards (1.2% of C4; the same shard indices as the Arm B reproduction) take about 2 h with 2
workers: each shard is a ~400 MB download plus about 15 minutes of CPU (measured: 16 min on a laptop core). The output goes to Google Drive, so a session that
stops can be resumed by re-running the cell. More shards (section 7) narrow the CIs for the rare dialects.

**Content warning.** The excluded documents include pornographic and hateful text. The inspection sheet and the
clustering sample contain such text.
"""),
    md("""
## 1. Get the project

Either set `REPO_URL` to your GitHub repository, or leave it empty and upload a zip of the
`reproduction/f5_excluded_data` folder when asked.
"""),
    code(f"""
import os, shutil, subprocess, zipfile
REPO_URL = ""   # e.g. "https://github.com/<you>/<repo>.git"
W = "{W}"
if not os.path.isdir(f"{{W}}/data"):
    if REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/repo"], check=True)
        shutil.copytree("/content/repo/reproduction/f5_excluded_data", W)
    else:
        from google.colab import files
        up = files.upload()                      # choose the zip of reproduction/f5_excluded_data
        name = next(iter(up))
        zipfile.ZipFile(name).extractall("/content/_upload")
        src = next(r for r, d, f in os.walk("/content/_upload") if "scripts" in d and "data" in d)
        shutil.copytree(src, W)
os.chdir(W)
os.makedirs("scripts", exist_ok=True)
print("working folder:", os.getcwd(), "| blocklist present:", os.path.exists("data/ldnoobw_en_25e679f.txt"))
"""),
    code("!pip -q install huggingface_hub scikit-learn scipy pandas matplotlib"),
    md("## 2. Scripts (embedded from the repo, so the notebook runs the current code)"),
    *[script_cell(n) for n in SCRIPTS],
    md("""
## 3. Scan 12 shards of `en.noblocklist` into Google Drive

Re-run this cell after a timeout: finished shards are skipped. Each line reports the share excluded (about 7% expected)
and how often the fast dialect inference agrees with the exact one (expected above 99%).
"""),
    code("""
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/f5_S1_noblocklist_12_shards"
"""),
    code(f"""
%cd {W}
!python scripts/scan_noblocklist.py --n 12 --seed 0 --workers 2 --cache_dir /content/hf_cache --out_dir "$OUT"
"""),
    md("""
## 4. Analysis: Fig. 5 + Table 6, §5.3 dialects, Fig. 6 clusters

About 20-30 minutes on free Colab. Fig. 5, Table 6, the dialect results and the inspection sheet are saved first;
then one k-means run (as in the paper; about 10 min), after which `clusters.csv` and `fig6_clusters.png` are saved
straight away. Progress is printed at each step.
"""),
    code(f"""
%cd {W}
!python scripts/analyze_excluded.py --scan "$OUT" --out_dir "$OUT/analysis"
"""),
    md("""
**If the session dropped during clustering:** the other outputs are already on Drive. After reconnecting, run
section 1 (upload the zip), the `pip` cell, the section 2 script cells and the Drive cell at the top of section 3
(not the scan), then only this cell. It redoes just §5.1. Add `--cluster_seeds 0 1 2` for the optional
seed-robustness check (about 10 more minutes per seed).
"""),
    code(f"""
%cd {W}
!python scripts/analyze_excluded.py --scan "$OUT" --out_dir "$OUT/analysis" --clusters_only
"""),
    md("## 5. Results"),
    code("""
import pandas as pd
from IPython.display import Image, Markdown, display
A = f"{OUT}/analysis"
display(Markdown(open(f"{A}/summary.md").read().split("## ")[0]))
display(Markdown("### §5.2 Figure 5 and Table 6"))
display(Image(f"{A}/fig5_pmi.png", width=900)); display(Markdown(open(f"{A}/table6.md").read()))
display(Markdown("### §5.3 dialects"))
display(pd.read_csv(f"{A}/dialects.csv").round(3)); display(Image(f"{A}/fig_dialects.png", width=1100))
display(Markdown("### §5.1 Figure 6"))
display(Image(f"{A}/fig6_clusters.png", width=1200))
c = pd.read_csv(f"{A}/clusters.csv")
display(pd.DataFrame([{"seed": s, "sexual clusters": int(g.auto_sexual.sum()),
                       "share of docs %": g.loc[g.auto_sexual, "share_%"].sum()} for s, g in c.groupby("seed")]))
display(c[c.seed == c.seed.min()].sort_values("docs", ascending=False)[["cluster", "docs", "share_%", "top10", "auto_sexual"]])
"""),
    md("""
## 6. §5.2 manual check (needs a person)

The paper read 50 excluded documents mentioning "lesbian" and 50 mentioning "gay", and found 22% and 36% to be
non-offensive or non-sexual. `inspect_lesbian_gay.csv` (in the analysis folder on Drive) holds the same kind of
random sample. Fill its `label` column with one of:

- `offensive`: hateful or derogatory;
- `sexual`: pornographic or sexual content;
- `neither`: non-offensive and non-sexual (e.g. news, rights, marriage, health).

Then run the next cell. It reports the share labelled `neither`, with a 95% Wilson interval; with 50 documents the
interval is about ±13 points.
"""),
    code("""
import math
import pandas as pd
s = pd.read_csv(f"{OUT}/analysis/inspect_lesbian_gay.csv").fillna("")
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)
for term, paper in (("lesbian", 22), ("gay", 36)):
    d = s[(s.term == term) & (s.label != "")]
    if len(d) == 0:
        print(term, ": not labelled yet"); continue
    k = (d.label.str.strip().str.lower() == "neither").sum()
    lo, hi = wilson(k, len(d))
    print(f"{term}: {100 * k / len(d):.0f}% neither [{lo:.0f}-{hi:.0f}] of {len(d)} labelled (paper {paper}%)")
"""),
    md("""
## 7. Optional: more shards for tighter dialect CIs

AAE and Hispanic documents are rare (well under 1%), so their removal rates have wide CIs on 12 shards. This adds
a contiguous block of shards to a second folder; re-run it in later sessions to continue. Then point section 4 at
`OUT2`.
"""),
    code(f"""
OUT2 = "/content/drive/MyDrive/f5_S2_noblocklist_more"
%cd {W}
!python scripts/scan_noblocklist.py --range 0 48 --workers 2 --cache_dir /content/hf_cache --out_dir "$OUT2"
"""),
    md("## 8. Download the analysis"),
    code("""
!cd "$OUT" && zip -qr /content/f5_analysis.zip analysis stats_*.json shards.json
from google.colab import files
files.download("/content/f5_analysis.zip")
"""),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "cells": []}
    lines = []
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        nb["cells"].append(cell)
        lines.append(f"# ======================= CELL {i} ({kind}) =======================\n{src}\n")
    with open(os.path.join(HERE, "f5_excluded_colab.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, "f5_excluded_cells.txt"), "w", encoding="utf-8") as f:
        f.write("Paste each block into its own Colab cell (markdown blocks into text cells).\n"
                "Do not paste the '# ===== CELL' separator lines.\n\n" + "\n".join(lines))
    print(f"wrote {len(CELLS)} cells")


if __name__ == "__main__":
    main()
