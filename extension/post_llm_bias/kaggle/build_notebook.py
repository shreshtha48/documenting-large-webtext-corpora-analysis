"""Build the Kaggle notebook for the 2019-vs-2024 group-sentiment comparison from the scripts in this repo.

Writes kaggle/post_llm_bias.ipynb next to this file. Every script (and the three Arm B lexicons) is embedded with
%%writefile, so the notebook runs the repo's current code. Re-run this builder after editing any of them.

Usage: python extension/post_llm_bias/kaggle/build_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
W = "/kaggle/working/plb"
SCRIPTS = [
    "extension/machine_text/scripts/filter_c4_pipeline.py",
    "extension/machine_text/scripts/run_wet_shards.py",
    "reproduction/demographic_bias/scripts/scan_c4_paragraphs.py",
    "reproduction/demographic_bias/scripts/count_sentiment.py",
    "extension/post_llm_bias/scripts/scan_groups.py",
    "extension/post_llm_bias/scripts/classify_sentiment.py",
    "extension/post_llm_bias/scripts/compare_years.py",
]
LEXICONS = ["socialsent_250.tsv", "vader.tsv", "unqover.tsv"]


def md(t):
    return ("markdown", t.strip("\n"))


def code(t):
    return ("code", t.strip("\n"))


def embed(rel, dest):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return code(f"%%writefile {dest}\n" + f.read())


CELLS = [
    md("""
# Group sentiment in C4: 2019 vs post-LLM 2024

Arm B (Dodge et al. 2021, §4.3 / A.7) run on two corpora built with **the same C4 filters**:
- **2019**: C4.EN from `allenai/c4`, the 12 seeded shards of Arm B's B1 run (seed 0).
- **2024**: our C4.EN, made by running TFDS v3.0.1's C4 filters (`filter_c4_pipeline.py`, checked page by page
  against the TFDS code) on Common Crawl WET shards from **CC-MAIN-2024-10** (Feb/Mar 2024) and **CC-MAIN-2024-51**
  (Dec 2024). Shards are drawn at random over the whole crawl with a fixed seed.

**What is measured**, for all 15 Table 7 groups (Arm B's string matching):
- positivity of paragraphs mentioning a group: Arm B's three lexicons + a RoBERTa sentiment classifier (GPU);
- the same **relative to a baseline** of random paragraphs from the same corpus, so a general change in web tone is
  not read as a change towards a group (difference-in-differences);
- the paper's Jewish − Arab gap, also without geopolitical "Arab" phrases ("United Arab Emirates", ...);
- dispersion of the groups' relative positivity (how unequal they are), raw and noise-corrected.

95% CIs: web-domain bootstrap, B = 1000, corpora resampled independently.

**Known limits** (say them in the write-up):
- One 2024 WET shard gives ~9.5k C4.EN docs (1/37 of a 2019 C4 shard). URL and line dedupe run per WET shard,
  so 2024 keeps more boilerplate than real C4.
- 2019 → 2024 differences are not attributable to LLM text: world events (Oct 2023 onwards) change how groups are
  written about. The machine-text notebook addresses the LLM question within 2024.
- "Black", "White", "African", "European" also match non-ethnic uses.

**Runtime** (T4 ×2 + internet; "Save Version → Save & Run All" runs it in the background):
2024 shards ~1–1.5 h (CPU, 4 in parallel) · 2019 shards ~15 min · classifier ~20 min (GPU) · analysis ~5 min.
Every step resumes if re-run.
"""),
    md("""
## Hypothesis (write this yourself before running)

- **Prediction:** …
- **What result would change your mind:** …
- **Main alternative explanation and how the baseline / robustness rows address it:** …
"""),
    code(f"""
# Setup. Kaggle: Accelerator = GPU T4 x2, Internet = on.
!pip install -q langdetect "nltk>=3.8.2"
import nltk; nltk.download("punkt_tab", quiet=True)
import os, glob, shutil, subprocess, torch
W = "{W}"
os.makedirs(f"{{W}}/scripts", exist_ok=True); os.makedirs(f"{{W}}/data/lexicons", exist_ok=True)
os.chdir(W)
print("GPUs:", torch.cuda.device_count(), "| CPUs:", os.cpu_count())

# ---- settings ----
CRAWLS = ["CC-MAIN-2024-10", "CC-MAIN-2024-51"]
N_PER_CRAWL = 24        # WET shards per crawl (48 in total)
INCLUDE = [0, 3]        # always in the sample (the pilot shards)
WORKERS = 4             # parallel WET shards; Kaggle has 4 vCPUs
N_2019 = 12             # Arm B's B1 shards
OUT_2024 = f"{{W}}/post_llm_c4"
PAR = f"{{W}}/paragraphs"
ANALYSIS = f"{{W}}/analysis"

# Optional: pilot outputs uploaded as a Kaggle dataset (e.g. the post-llm-crawl folder) are copied in, so those
# shards are not processed again. Kaggle unpacks .gz files in datasets; they are re-compressed here so every
# shard in OUT_2024 is .jsonl.gz.
import gzip
pilot = glob.glob("/kaggle/input/**/c4_*_CC-MAIN-*_shard*.jsonl*", recursive=True)
pilot += glob.glob("/kaggle/input/**/stats_CC-MAIN-*_shard*.json", recursive=True)
for f in pilot:
    crawl = next((c for c in CRAWLS if c in os.path.basename(f)), None)
    if crawl is None:
        continue
    os.makedirs(f"{{OUT_2024}}/{{crawl}}", exist_ok=True)
    name = os.path.basename(f) + (".gz" if f.endswith(".jsonl") else "")
    dst = f"{{OUT_2024}}/{{crawl}}/{{name}}"
    if os.path.isdir(f):        # Kaggle may unpack x.jsonl.gz into a folder named x.jsonl holding the file
        f = next(os.path.join(r, x) for r, _, xs in os.walk(f) for x in xs)
    if not os.path.exists(dst):
        if not f.endswith(".gz"):
            with open(f, "rb") as a, gzip.open(dst, "wb") as b:
                shutil.copyfileobj(a, b)
        else:
            shutil.copy(f, dst)
        print("reused", dst)
"""),
    md("## Scripts and lexicons (embedded from the repo)"),
    *[embed(s, f"{W}/scripts/{os.path.basename(s)}") for s in SCRIPTS],
    *[embed(f"reproduction/demographic_bias/data/lexicons/{l}", f"{W}/data/lexicons/{l}") for l in LEXICONS],
    code(f"""
# Check that every cell above ran (Kaggle wipes /kaggle/working when a new session starts)
need = {[f"scripts/{os.path.basename(s)}" for s in SCRIPTS] + [f"data/lexicons/{l}" for l in LEXICONS]}
missing = [p for p in need if not os.path.exists(f"{{W}}/{{p}}")]
assert not missing, f"run the %%writefile cells above again; missing: {{missing}}"
print("all scripts and lexicons in place")
"""),
    md("""
## 1. Build post-LLM C4 from 2024 WET shards (CPU)

Downloads each WET shard (~100 MB), runs the C4 filters, writes C4.EN / C4.BLOCKLIST / stats, deletes the WET
file. Re-run to resume or retry failed shards.
"""),
    code("""
crawls, include = " ".join(CRAWLS), " ".join(map(str, INCLUDE))
!python scripts/run_wet_shards.py --crawls {crawls} --n_per_crawl {N_PER_CRAWL} --include {include} --workers {WORKERS} --out_dir {OUT_2024} --cache_dir /kaggle/tmp/wet_cache
"""),
    code("""
import json, pandas as pd
rows = [json.load(open(p)) for p in sorted(glob.glob(f"{OUT_2024}/*/stats_*.json"))]
st = pd.DataFrame([{"crawl": r["crawl_or_shard"], "wet": r["wet_url_or_file"].rsplit("/", 1)[-1],
                    "pages": r["total_documents_scanned"], "c4_en": r["c4_en_kept"],
                    "blocklist": r["blocklist_excluded"]} for r in rows])
display(st.groupby("crawl")[["pages", "c4_en", "blocklist"]].agg(["count", "sum"]))
"""),
    md("## 2. Paragraphs mentioning each group, plus a random baseline (CPU)"),
    code("""
!python scripts/scan_groups.py --c4_2019 {N_2019} --out_dir {PAR}/2019 --workers 3 --cache_dir /kaggle/tmp/hf
for crawl in CRAWLS:
    !python scripts/scan_groups.py --c4_files "{OUT_2024}/{crawl}/c4_en_*.jsonl.gz" --out_dir {PAR}/{crawl} --workers 4
"""),
    md("## 3. RoBERTa sentiment of every paragraph (GPU)"),
    code("""
dirs = " ".join([f"{PAR}/2019"] + [f"{PAR}/{c}" for c in CRAWLS])
!python scripts/classify_sentiment.py --paragraphs {dirs} --batch 256
"""),
    md("## 4. Compare 2019 with each 2024 crawl and with both pooled"),
    code("""
pooled = ",".join(f"{PAR}/{c}" for c in CRAWLS)
args = f'--corpus "2019={PAR}/2019" ' + " ".join(f'--corpus "{c}={PAR}/{c}"' for c in CRAWLS)
args += f' --corpus "2024 pooled={pooled}"'
!python scripts/compare_years.py {args} --out_dir {ANALYSIS}
"""),
    code("""
from IPython.display import Image, Markdown, display
display(Markdown(open(f"{ANALYSIS}/summary.md", encoding="utf-8").read().split("## Positivity")[0]))
for fig in ["fig_gap_by_corpus", "fig_change_vs_baseline", "fig_positivity_by_corpus"]:
    display(Image(f"{ANALYSIS}/{fig}.png"))
"""),
    md("""
## 5. Save

- `post_llm_bias_results.zip`: analysis, paragraphs, classifier scores and per-shard stats. Put its contents in
  `extension/post_llm_bias/results/<run id>/` and add a row to the experiment log.
- `post_llm_c4_2024.zip`: the 48 shards of post-LLM C4.EN / C4.BLOCKLIST (~0.5 GB), reusable for the
  benchmark-contamination notebook.
"""),
    code("""
os.chdir(W)
!zip -qr /kaggle/working/post_llm_bias_results.zip analysis paragraphs post_llm_c4/*/stats_*.json post_llm_c4/*/shards.json
!zip -qr /kaggle/working/post_llm_c4_2024.zip post_llm_c4 -x "*/logs/*"
!ls -lh /kaggle/working/*.zip
"""),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    out = os.path.join(HERE, "post_llm_bias.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
