"""Build the Google Colab notebook for Arm B (C4 paragraph sentiment) from the scripts in this repo.

Writes, next to this file:
    f4_arm_b_colab.ipynb   open in Colab (File -> Upload notebook)

The scripts are embedded with %%writefile, so the notebook runs the repo's current code. No GPU is needed.
Re-run this builder after editing a script.

Usage: python colab/build_colab_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = ["build_lexicons.py", "scan_c4_paragraphs.py", "count_sentiment.py"]
W = "/content/f4"   # working copy of reproduction/demographic_bias


def md(text):
    return ("markdown", text.strip("\n"))


def code(text):
    return ("code", text.strip("\n"))


def script_cell(name):
    with open(os.path.join(ROOT, "scripts", name), encoding="utf-8") as f:
        return code(f"%%writefile {W}/scripts/{name}\n" + f.read())


CELLS = [
    md("""
# Dodge et al. 2021 §4.3, Arm B: sentiment of C4 paragraphs about "Jewish" and "Arab"

**What the paper did (A.7).** It picked "Jewish" and "Arab", the two groups with the most extreme model bias in
Table 7 that simple string matching could find. It then found every C4.EN paragraph containing " Jewish " or " Arab "
(spaces on both sides) and counted the positive and negative words from three lexicons (SocialSent-derived, VADER,
and a small one from the UnQover question words).
Result: SocialSent positivity 73.2% vs 65.7%, a gap of 7.5 points; VADER gap 2.0; UnQover gap 1.4.

| ID | Run | Status |
|---|---|---|
| B0 | One shard (00000) | done: reproduces (gap 7.3) |
| B1 | **Reproduction**: 12 seeded shards, Jewish and Arab | done: reproduces (gap 7.1 [5.6–8.7]; VADER 2.0) |
| B2 | Extension: all 15 Table 7 groups, one shard | done: corpus positivity does not track Table 7 (Spearman −0.05) |
| B3 | Extension: all 15 groups, the same 12 shards as B1 | **this notebook, optional** |
| B4 | Larger sample: a quarter of C4.EN (256 shards), so NYT / Al Jazeera get enough tokens | **this notebook, optional, free tier, multi-session** |

CPU only (free tier is enough). Each shard is a ~319 MB download from Hugging Face plus about 2 minutes of scanning;
`--workers` scans several shards in parallel. Runs resume, and a folder can only hold one set of target words.

**Robustness (audit 2026-10-23).** Besides the paper's numbers, every analysis now writes `robustness.csv` and
`fig_robustness.png`: the gap without "United Arab Emirates" paragraphs (22% of the "Arab" paragraphs), without
other geopolitical "Arab" phrases, without noisy SocialSent words, and without religious words. The UnQover lexicon
is v2 (tokenised like the corpus; v1 split hyphenated words and counted "well" / "mean" as sentiment).
"""),
    md("""
## 1. Get the project

Either set `REPO_URL` to your GitHub repository, or leave it empty and upload a zip of the
`reproduction/demographic_bias` folder when asked.
"""),
    code(f"""
import os, shutil, subprocess, zipfile
REPO_URL = ""   # e.g. "https://github.com/<you>/<repo>.git"
W = "{W}"
if not os.path.isdir(f"{{W}}/scripts"):
    if REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/repo"], check=True)
        shutil.copytree("/content/repo/reproduction/demographic_bias", W)
    else:
        from google.colab import files
        up = files.upload()                      # choose the zip of reproduction/demographic_bias
        name = next(iter(up))
        zipfile.ZipFile(name).extractall("/content/_upload")
        src = next(r for r, d, f in os.walk("/content/_upload") if "scripts" in d and "third_party" in d)
        shutil.copytree(src, W)
os.chdir(W)
os.makedirs("scripts", exist_ok=True)
print("working folder:", os.getcwd(), "| has UnQover word lists:", os.path.isdir("third_party/unqover"))
"""),
    code("!pip -q install huggingface_hub pandas matplotlib"),
    md("## 2. Scripts (embedded from the repo, so the notebook runs the current code)"),
    *[script_cell(n) for n in SCRIPTS],
    md("""
## 3. Lexicons

SocialSent: the 250 subreddit lexicons released by Hamilton et al. (2016), here from the copy bundled in
`shifterator` (pinned commit). The `socialsent` Python package only contains the code that induces lexicons, not
these ready-made ones. Averaged over lexicons, words kept if present in at least 40, ethnic terms removed, top 250
on each side. VADER: words scoring above +1.0 or below −1.0. UnQover: our reconstruction (the paper's was not released).
"""),
    code(f"""
%cd {W}
!python scripts/build_lexicons.py
"""),
    md("""
## 4. B1, the reproduction: Jewish and Arab on 12 seeded shards

About 10-30 min with 3 workers. Skips shards that are already done (B1 results ship with the repo, so this may finish at once).
"""),
    code(f"""
%cd {W}
!python scripts/scan_c4_paragraphs.py --n 12 --seed 0 --workers 3 --out_dir results/B1_c4_12_shards
!python scripts/count_sentiment.py --paragraphs results/B1_c4_12_shards --out_dir results/B1_c4_12_shards/analysis
"""),
    code(f"""
import pandas as pd
from IPython.display import Image, Markdown, display
d = "{W}/results/B1_c4_12_shards/analysis"
display(Markdown("**Positivity per lexicon** (paper: SocialSent 73.2 / 65.7)"))
display(pd.read_csv(f"{{d}}/positivity.csv"))
display(Markdown("**Jewish minus Arab gap** (paper: 7.5 / 2.0 / 1.4)"))
display(pd.read_csv(f"{{d}}/gaps.csv"))
display(Markdown("**Counts vs the paper's totals scaled to 12/1024 of C4**"))
display(pd.read_csv(f"{{d}}/scale.csv"))
display(Markdown("**NYT / Al Jazeera** (too few tokens on a sample; the paper used the full corpus)"))
display(pd.read_csv(f"{{d}}/domains.csv"))
display(Image(f"{{d}}/fig_positivity.png", width=850)); display(Image(f"{{d}}/fig_gap.png", width=850))
display(Markdown("**Robustness: each row changes one thing from the paper's setting**"))
display(pd.read_csv(f"{{d}}/robustness.csv").round(2))
display(Image(f"{{d}}/fig_robustness.png", width=1100))
"""),
    md("""
## 5. B3, extension (optional): all 15 Table 7 groups on the same 12 shards

Not in the paper. Each group is matched by its spelling variants (e.g. "Middle-Eastern" or "Middle Eastern").
Caveats the paper itself gives: "African" also matches "African American", and "Black" and "White" have very common
non-ethnic senses. Read these numbers with that in mind. About 30 min.
"""),
    code(f"""
%cd {W}
!python scripts/scan_c4_paragraphs.py --table7_groups --n 12 --seed 0 --workers 3 --out_dir results/B3_ext_all_groups_12_shards
!python scripts/count_sentiment.py --paragraphs results/B3_ext_all_groups_12_shards --out_dir results/B3_ext_all_groups_12_shards/analysis
"""),
    code(f"""
import pandas as pd
from IPython.display import Image, display
d = "{W}/results/B3_ext_all_groups_12_shards/analysis"
p = pd.read_csv(f"{{d}}/positivity.csv")
display(p[p.lexicon == "SocialSent"].sort_values("positivity", ascending=False))
display(Image(f"{{d}}/fig_positivity_all_groups.png", width=1100)); display(Image(f"{{d}}/fig_corpus_vs_table7.png", width=750))
"""),
    md("""
## 6. B4 (optional, free tier, several sessions): a quarter of C4.EN

The paper's NYT / Al Jazeera numbers came from 1.6k-7.9k sentiment tokens on the full corpus; 12 shards give
30-101. 256 shards (1/4 of C4.EN) give roughly 400-2,000, enough to compare. The full 1,024 shards are possible
the same way but take about four times as long.

The output goes to **Google Drive**, so a session that times out loses nothing: re-run this cell in a new session
and it continues. Free Colab stops idle sessions, so keep the tab open. The first shards print their time; multiply
by 256 / workers for the total, and split it over sessions as needed. The paragraphs take roughly 100 MB on Drive.
"""),
    code(f"""
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/f4_B4_c4_256_shards"      # one folder, reused by every session
%cd {W}
!python scripts/scan_c4_paragraphs.py --range 0 256 --workers 4 --cache_dir /content/hf_cache --out_dir "$OUT"
"""),
    code(f"""
# run when the scan reports 0 shards to scan (also works on a partial scan, with fewer shards)
%cd {W}
!python scripts/count_sentiment.py --paragraphs "$OUT" --out_dir "$OUT/analysis"
import pandas as pd
from IPython.display import Image, display
display(pd.read_csv(f"{{OUT}}/analysis/domains.csv")); display(pd.read_csv(f"{{OUT}}/analysis/gaps.csv"))
display(Image(f"{{OUT}}/analysis/fig_robustness.png", width=1100))
"""),
    md("## 7. Download the results (or copy them to Google Drive)"),
    code(f"""
%cd {W}
!zip -qr /content/f4_arm_b_results.zip results/B* data/lexicons/MANIFEST.json
from google.colab import files
files.download("/content/f4_arm_b_results.zip")
# Drive instead: from google.colab import drive; drive.mount("/content/drive"); !cp -r results /content/drive/MyDrive/f4_results
"""),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        nb["cells"].append(cell)
    with open(os.path.join(HERE, "f4_arm_b_colab.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(CELLS)} cells")


if __name__ == "__main__":
    main()
