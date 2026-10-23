"""Build the three Kaggle notebooks for the gender / age extension from the scripts and data in this repo.

Writes, next to this file:
    ga_probe_account1.ipynb   GPU T4 x2: UnifiedQA-3B on gender_names (+ its same-gender control) and gender_nouns
                              (50,400 contexts)
    ga_probe_account2.ipynb   GPU T4 x2: UnifiedQA-3B on age, gender_age, gender_ethnicity (28,400 contexts), then BBQ
                              Age / Gender_identity / Race_x_gender (25,296 examples, 3 options + generation)
    ga_corpus.ipynb           CPU: 12 en.noblocklist shards -> exclusion, dialect x gender, sentiment, gender-first
                              race zoom; then the sentiment classifier

Every script and data file is embedded with %%writefile, so a notebook needs no Kaggle dataset: import it
(File -> Import Notebook), turn Internet on, pick the accelerator, Save Version -> Save & Run All.
Re-run this builder after editing any script.

Usage: python kaggle/build_notebooks.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))
W = "/kaggle/working/ga"
S = os.path.join(ROOT, "scripts")
REPRO = os.path.join(REPO, "reproduction")

SCRIPTS = {   # name in the notebook -> source in the repo
    "generate_questions.py": os.path.join(S, "generate_questions.py"),
    "score_unifiedqa.py": os.path.join(REPRO, "demographic_bias", "scripts", "score_unifiedqa.py"),
    "analyze_arm_a.py": os.path.join(REPRO, "demographic_bias", "scripts", "analyze_arm_a.py"),
    "analyze_probe_stats.py": os.path.join(S, "analyze_probe_stats.py"),
    "analyze_probe.py": os.path.join(S, "analyze_probe.py"),
    "prepare_bbq.py": os.path.join(S, "prepare_bbq.py"),
    "score_bbq.py": os.path.join(S, "score_bbq.py"),
    "analyze_bbq.py": os.path.join(S, "analyze_bbq.py"),
    "scan_corpus.py": os.path.join(S, "scan_corpus.py"),
    "dialect.py": os.path.join(REPRO, "excluded_data", "scripts", "dialect.py"),
    "analyze_corpus.py": os.path.join(S, "analyze_corpus.py"),
    "classify_paragraphs.py": os.path.join(S, "classify_paragraphs.py"),
}
PROBE_SCRIPTS = ["generate_questions.py", "score_unifiedqa.py", "analyze_arm_a.py", "analyze_probe_stats.py",
                 "analyze_probe.py"]
BBQ_SCRIPTS = ["prepare_bbq.py", "score_bbq.py", "analyze_bbq.py"]
CORPUS_SCRIPTS = ["analyze_probe_stats.py", "scan_corpus.py", "dialect.py", "analyze_corpus.py", "classify_paragraphs.py"]
# vendored UnQover (byte-identical to allenai/unqover @ 3e47969); generate_questions.py adds its NOT UPSTREAM inputs
UNQOVER = ["LICENSE", "templates/__init__.py", "templates/lists.py", "templates/underspecified_templates.py",
           "templates/generate_underspecified_templates.py", "word_lists/activities/biased_ethnicity",
           "word_lists/slots/gender_noact", "word_lists/nouns/subjects/female", "word_lists/nouns/subjects/male"]
CORPUS_DATA = ["ldnoobw_en_25e679f.txt", "socialsent_250.tsv", "vader.tsv", "unqover.tsv"]


def md(t):
    return ("markdown", t.strip("\n"))


def code(t):
    return ("code", t.strip("\n"))


def embed(path, target):
    with open(path, encoding="utf-8") as f:
        return code(f"%%writefile {target}\n" + f.read())


SETUP = code(f"""
%%bash
mkdir -p {W}/scripts {W}/data {W}/results {W}/third_party/unqover/templates \\
         {W}/third_party/unqover/word_lists/activities {W}/third_party/unqover/word_lists/slots \\
         {W}/third_party/unqover/word_lists/fillers {W}/third_party/unqover/word_lists/nouns/subjects
nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU"
""")


def files_cells(scripts, data=(), unqover=False):
    cells = [md("## Scripts and data (embedded from the repo; these cells only write files)")]
    cells += [embed(SCRIPTS[s], f"{W}/scripts/{s}") for s in scripts]
    cells += [embed(os.path.join(ROOT, "data", d), f"{W}/data/{d}") for d in data]
    if unqover:
        cells += [embed(os.path.join(ROOT, "third_party", "unqover", f), f"{W}/third_party/unqover/{f}") for f in UNQOVER]
    return cells


def score_cell(run, exp="P1_unifiedqa3b"):
    return code(f"""
%%bash
cd {W}
BS=16
for L in {run}; do
  for P in 0 1; do
    CUDA_VISIBLE_DEVICES=$P python scripts/score_unifiedqa.py --questions data/questions/$L.jsonl.gz \\
      --model allenai/unifiedqa-t5-3b --out_dir results/{exp}/scores/$L \\
      --part $P --nparts 2 --batch_size $BS --gen_check 2000 > logs_${{L}}_$P.txt 2>&1 &
  done
  wait
  echo "== $L"; tail -n 2 logs_${{L}}_0.txt logs_${{L}}_1.txt
done
""")


def show_probe(lists):
    return code(f"""
import json, os
import pandas as pd
from IPython.display import Image, Markdown, display
os.chdir("{W}")
for L in {lists!r}:
    d = f"results/P1_unifiedqa3b/analysis/{{L}}"
    if not os.path.exists(f"{{d}}/diagnostics.json"):
        print("no analysis for", L); continue
    display(Markdown(f"### {{L}}"))
    diag = json.load(open(f"{{d}}/diagnostics.json"))
    display(Markdown("```\\n" + json.dumps(diag.get("summary", {{}}), indent=1)[:3000] + "\\n```"))
    display(pd.read_csv(f"{{d}}/contrasts.csv").query("within == 'all'"))
    for f in ("female_positivity_by_ethnicity.csv", "names_spread.csv"):
        if os.path.exists(f"{{d}}/{{f}}"):
            display(Markdown(f"**{{f}}**")); display(pd.read_csv(f"{{d}}/{{f}}"))
    if os.path.exists(f"{{d}}/fig_contrasts.png"):
        display(Image(f"{{d}}/fig_contrasts.png", width=900))
""")


def probe_notebook(account, lists, est, bbq):
    run = " ".join(lists)
    shown = [l for l in lists if l != "gender_names_ctrl"]
    cells = [
        md(f"""
# Gender / age extension, probe: account {account} of 2

UnifiedQA-3B (the size that reproduced Table 7 in E6). Questions come from **UnQover's own generator, unchanged**,
with the same 50 negative / positive attributes Dodge et al. used; only the people named change (subject lists and
two slot files derived from UnQover's mixed-gender `gender_noact` templates are added, marked NOT UPSTREAM).
Lists here: **{", ".join(lists)}** ({est}).{" Then **BBQ** (Parrish et al. 2022), the published benchmark for gender, age and race x gender." if bbq else ""}

**Settings:** Accelerator **GPU T4 x2**, Internet **on**. *Save Version -> Save & Run All (Commit)*, so it runs in the
background (up to 12 h). Download `ga_probe_account{account}.zip` from the Output tab when it finishes.

Resumable: scores are written in chunks; re-running a scoring cell skips finished chunks.
"""),
        SETUP,
        *files_cells(PROBE_SCRIPTS + (BBQ_SCRIPTS if bbq else []), unqover=True),
        md("## 1. Questions (UnQover generator, no model, seconds)"),
        code(f"""
%%bash
cd {W}
pip install -q sentencepiece
python scripts/generate_questions.py --lists {run}
"""),
        md("## 2. Smoke test: 64 contexts per list on one GPU. Stop here if a log shows an error."),
        code(f"""
%%bash
cd {W}
for L in {run}; do
  CUDA_VISIBLE_DEVICES=0 python scripts/score_unifiedqa.py --questions data/questions/$L.jsonl.gz \\
    --model allenai/unifiedqa-t5-3b --out_dir results/smoke/$L --limit 64 --batch_size 16 --gen_check 64 2>&1 | tail -n 2
done
"""),
        md("## 3. Full scoring, both GPUs (each takes half of each list). If `CUDA out of memory`: `BS=8`, re-run."),
        score_cell(run),
        md("## 4. Analysis: clean contrasts, gender gaps, gender-first race zoom, controls, diagnostics"),
        code(f"""
%%bash
cd {W}
python scripts/analyze_probe.py --scores results/P1_unifiedqa3b/scores --lists {" ".join(shown)} \\
  --out_dir results/P1_unifiedqa3b/analysis
"""),
        show_probe(shown),
    ]
    if bbq:
        cells += [
            md("""
## 5. BBQ: Age, Gender_identity, Race_x_gender (UnifiedQA-3B)

Pinned BBQ data (nyu-mll/BBQ @ bea11bd). RACE-style input, three options. Both answer rules are recorded: BBQ's
(generated answer matched to an option) and ours (log-likelihood). The analysis computes BBQ's official bias scores,
then, for Race_x_gender, gender first (Black woman vs Black man...) and race within gender (Black woman vs White
woman...), with anomaly tests. It also re-analyses the authors' published UnifiedQA-11B predictions for comparison.
About 1 h on 2 x T4.
"""),
            code(f"""
%%bash
cd {W}
python scripts/prepare_bbq.py
for P in 0 1; do
  CUDA_VISIBLE_DEVICES=$P python scripts/score_bbq.py --model allenai/unifiedqa-t5-3b --out_dir results/P2_bbq_3b/scores \\
    --part $P --nparts 2 --batch_size 16 > logs_bbq_$P.txt 2>&1 &
done
wait; tail -n 2 logs_bbq_0.txt logs_bbq_1.txt
python scripts/analyze_bbq.py --scores results/P2_bbq_3b/scores --out_dir results/P2_bbq_3b/analysis | tail -n 40
python scripts/analyze_bbq.py --published_11b --out_dir results/P2_bbq_11b_published | tail -n 25
"""),
        ]
    cells += [
        md("## Download (Output tab)"),
        code(f"""
%%bash
cd {W}
zip -qr /kaggle/working/ga_probe_account{account}.zip results logs_*.txt data/questions -x 'results/smoke/*'
ls -la /kaggle/working/ga_probe_account{account}.zip
"""),
    ]
    return cells


def corpus_notebook():
    return [
        md("""
# Gender / age extension, corpus: blocklist exclusion, dialect x gender, sentiment, gender-first race zoom

One pass over 12 C4.EN.NOBLOCKLIST shards (the reproduction's shard indices, seed 0): for every document, whether
C4's blocklist removed it (and whether only because of a blocklist entry containing the term's own words), which
gender / age / gender x ethnicity terms it mentions, its dialect (Blodgett et al.), and lexicon sentiment of the
paragraphs that mention each term. Then the analysis, then a RoBERTa sentiment classifier on sampled paragraphs.
The URL-year (time) breakdown is computed but **on hold**: C4 has no dated snapshots after April 2019.

**Settings:** Accelerator **None (CPU)** (the classifier then scores at most 150 paragraphs per term; with a GPU it
scores all). Internet **on**. *Save Version -> Save & Run All (Commit)*. About 1.5-2 h. Download `ga_corpus.zip`.

Resumable: a finished shard has a `stats_<shard>.json`; re-running skips it.
"""),
        SETUP,
        *files_cells(CORPUS_SCRIPTS, CORPUS_DATA),
        md("## 1. Smoke test: first 2,000 documents of one shard"),
        code(f"""
%%bash
cd {W}
python - <<'EOF'
import gzip
from huggingface_hub import hf_hub_download
p = hf_hub_download("allenai/c4", "en.noblocklist/c4-train.00000-of-01024.json.gz", repo_type="dataset")
with gzip.open(p, "rt", encoding="utf-8") as f, gzip.open("smoke_c4-train.00000-of-01024.json.gz", "wt", encoding="utf-8") as g:
    for i, line in enumerate(f):
        if i >= 2000: break
        g.write(line)
EOF
python scripts/scan_corpus.py --files smoke_c4-train.00000-of-01024.json.gz --out_dir results/smoke_corpus
python scripts/analyze_corpus.py --scan results/smoke_corpus --out_dir results/smoke_corpus/analysis --B 20 --no_figures | tail -n 3
"""),
        md("## 2. Full scan: 12 shards, 4 in parallel"),
        code(f"""
%%bash
cd {W}
python scripts/scan_corpus.py --n 12 --seed 0 --workers 4 --out_dir results/C1_noblocklist_12_shards 2>&1 | tail -n 15
"""),
        md("## 3. Sentiment classifier on the sampled paragraphs"),
        code(f"""
%%bash
cd {W}
if nvidia-smi > /dev/null 2>&1; then CAP=0; else CAP=150; fi
python scripts/classify_paragraphs.py --scan results/C1_noblocklist_12_shards --max_per_term $CAP --batch_size 32 2>&1 | tail -n 3
"""),
        md("## 4. Analysis"),
        code(f"""
%%bash
cd {W}
python scripts/analyze_corpus.py --scan results/C1_noblocklist_12_shards --out_dir results/C1_noblocklist_12_shards/analysis --B 200
"""),
        code(f"""
import os
from IPython.display import Image, Markdown, display
os.chdir("{W}/results/C1_noblocklist_12_shards/analysis")
display(Markdown(open("summary.md").read()))
for f in ("fig_exclusion_pmi.png", "fig_exclusion_gaps.png", "fig_sentiment_gaps.png", "fig_sentiment_terms.png"):
    if os.path.exists(f):
        display(Image(f, width=900))
"""),
        md("## 5. Download (Output tab). Raw shards are deleted after scanning."),
        code(f"""
%%bash
cd {W}
zip -qr /kaggle/working/ga_corpus.zip results -x 'results/smoke*'
ls -la /kaggle/working/ga_corpus.zip
"""),
    ]


def write(cells, name):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(cells):
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        nb["cells"].append(cell)
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"{name}: {len(cells)} cells")


def main():
    write(probe_notebook(1, ["gender_names", "gender_names_ctrl", "gender_nouns"],
                         "50,400 contexts, ~1.4 h on 2 x T4", bbq=False), "ga_probe_account1.ipynb")
    write(probe_notebook(2, ["age", "gender_age", "gender_ethnicity"], "28,400 contexts, ~0.8 h on 2 x T4", bbq=True),
          "ga_probe_account2.ipynb")
    write(corpus_notebook(), "ga_corpus.ipynb")


if __name__ == "__main__":
    main()
