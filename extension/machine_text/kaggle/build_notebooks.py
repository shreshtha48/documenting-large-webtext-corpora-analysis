"""Build the Kaggle notebooks of extension/machine_text from the scripts in this repo.

Writes, next to this file:
    machine_text_4shards.ipynb   machine-generated text in the 4 pilot shards (Feb vs Dec 2024, vs 2019 C4)
    c4_filter_pipeline.ipynb     only its %%writefile cell is refreshed from scripts/filter_c4_pipeline.py

Scripts and lexicons are embedded with %%writefile, so the notebooks run the repo's current code. Re-run this
builder after editing any of them.

Usage: python extension/machine_text/kaggle/build_notebooks.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
W = "/kaggle/working/mt"
SCRIPTS = [
    "reproduction/demographic_bias/scripts/scan_c4_paragraphs.py",
    "reproduction/demographic_bias/scripts/count_sentiment.py",
    "extension/machine_text/scripts/machine_text.py",
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
# Machine-generated text in post-LLM C4: Feb vs Dec 2024, against 2019 C4

**Data.** The 4 pilot shards of post-LLM C4 (C4.EN + C4.BLOCKLIST, made with TFDS v3.0.1's C4 filters):
CC-MAIN-2024-10 (Feb/Mar 2024) shards 0 and 3, CC-MAIN-2024-51 (Dec 2024) shards 0 and 3. Upload the
`extension/post-llm-crawl` folder as a Kaggle dataset. 2019 C4 (`allenai/c4` shard 00000) is the **negative control**:
it predates ChatGPT, so detector flags there are false positives.
Only documents with ≥ 100 words are used (detectors are unreliable on short text).

**Two independent measures of machine text:**
1. **Marker words** (Kobak et al. 2024 style): share of documents whose first 300 words contain a pre-registered LLM
   style word ("delve", "underscores", "showcasing", "tapestry", ...), against control words. Lower bound on the
   LLM-processed share: f ≥ (p − q) / (1 − q).
2. **Binoculars** (Hans et al. 2024), zero-shot, GPU: thresholds set at the 1% and 5% lowest scores of 2019 C4, so
   "flagged" = more machine-like than 99% (95%) of 2019 text. The flag rate per ethnicity group in 2019 checks
   whether the detector flags some groups' text more (non-native English is flagged more, Liang et al. 2023).

**Tone:** lexicon (Arm B) and RoBERTa positivity of flagged vs unflagged documents, and of group-mentioning
paragraphs inside them.

**Limits.** 4 shards ≈ 25k documents, so group-level CIs are wide; they are reported, not hidden (future work: more
shards). Binoculars with the small Qwen pair is weaker than the paper's Falcon-7B pair. 2019 → 2024 also differs
in topics and sites, so the Feb → Dec change is the cleaner comparison.

**Runtime** (T4 ×2): docs ~5 min · Binoculars ~1–1.5 h for every document (`MAX_PER_CORPUS` subsamples) ·
RoBERTa ~5 min · analysis ~3 min. Every step resumes.
"""),
    md("""
## Hypothesis (write this yourself before running)

- **Prediction:** …
- **What result would change your mind:** …
- **Main alternative explanation:** …
"""),
    code(f"""
# Setup. Kaggle: Accelerator = GPU T4 x2, Internet = on; add the post-llm-crawl dataset as input.
import os, glob, torch
W = "{W}"
os.makedirs(f"{{W}}/scripts", exist_ok=True); os.makedirs(f"{{W}}/data/lexicons", exist_ok=True)
os.chdir(W)
print("GPUs:", torch.cuda.device_count())

INPUT_GLOB = "/kaggle/input/**/c4_*_CC-MAIN-*_shard*.jsonl*"   # Kaggle unpacks .gz, so .jsonl or .jsonl.gz
print(len(glob.glob(INPUT_GLOB, recursive=True)), "input files (expect 8: 4 shards x C4.EN / blocklist)")
OUT = f"{{W}}/work"
# Binoculars pair (observer = base, performer = instruct, same tokenizer).
# Default fits one T4. Stronger options: Qwen/Qwen2.5-3B (+ -Instruct, needs 2 GPUs);
# the paper's tiiuae/falcon-7b + tiiuae/falcon-7b-instruct does not fit T4s in fp16.
OBSERVER, PERFORMER = "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"
MAX_PER_CORPUS = None   # e.g. 5000 to score a random subset of each crawl (2019 is always scored in full)
"""),
    md("## Scripts and lexicons (embedded from the repo)"),
    *[embed(s, f"{W}/scripts/{os.path.basename(s)}") for s in SCRIPTS],
    *[embed(f"reproduction/demographic_bias/data/lexicons/{l}", f"{W}/data/lexicons/{l}") for l in LEXICONS],
    md("## 1. Documents: the 4 post-LLM shards + 2019 C4 control (CPU)"),
    code("""
!python scripts/machine_text.py docs --c4_files "{INPUT_GLOB}" --out {OUT} --cache_dir /kaggle/tmp/hf
"""),
    md("## 2. Binoculars scores (GPU)"),
    code("""
limit = f"--max_per_corpus {MAX_PER_CORPUS}" if MAX_PER_CORPUS else ""
!python scripts/machine_text.py detect --out {OUT} --observer {OBSERVER} --performer {PERFORMER} {limit}
"""),
    md("## 3. RoBERTa sentiment of each document (GPU)"),
    code("""
!python scripts/machine_text.py classify --out {OUT}
"""),
    md("## 4. Analysis"),
    code("""
!python scripts/machine_text.py analyze --out {OUT}
from IPython.display import Image, Markdown, display
display(Markdown(open(f"{OUT}/analysis/summary.md", encoding="utf-8").read()))
for fig in ["fig_marker_words", "fig_binoculars_ecdf", "fig_tone"]:
    if os.path.exists(f"{OUT}/analysis/{fig}.png"):
        display(Image(f"{OUT}/analysis/{fig}.png"))
"""),
    md("""
## 5. Save

`machine_text_results.zip` holds the analysis and the per-document Binoculars / RoBERTa scores. Put it in
`extension/machine_text/results/<run id>/` and add a row to the experiment log.
"""),
    code("""
os.chdir(OUT)
!zip -qr /kaggle/working/machine_text_results.zip analysis binoculars.jsonl roberta.npy
!ls -lh /kaggle/working/*.zip
"""),
]


def write_nb(cells, path):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(cells):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {path} ({len(cells)} cells)")


def refresh_filter_notebook():
    path = os.path.join(HERE, "c4_filter_pipeline.ipynb")
    nb = json.load(open(path, encoding="utf-8"))
    with open(os.path.join(ROOT, "extension/machine_text/scripts/filter_c4_pipeline.py"), encoding="utf-8") as f:
        script = f.read()
    cells = [c for c in nb["cells"] if "".join(c["source"]).startswith("%%writefile filter_c4_pipeline.py")]
    assert len(cells) == 1, "expected one %%writefile cell"
    cells[0]["source"] = ("%%writefile filter_c4_pipeline.py\n" + script).splitlines(keepends=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"refreshed {path}")


if __name__ == "__main__":
    write_nb(CELLS, os.path.join(HERE, "machine_text_4shards.ipynb"))
    refresh_filter_notebook()
