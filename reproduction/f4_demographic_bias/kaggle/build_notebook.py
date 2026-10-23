"""Build the end-to-end Kaggle notebook for Arm A from the scripts in this repo.

Writes, next to this file:
    f4_arm_a_end_to_end.ipynb   import into Kaggle (File -> Import Notebook)
    f4_arm_a_cells.txt          the same cells as plain text, for copy-paste

The scripts are embedded with %%writefile, so the notebook always runs the repo's current code,
whatever version of the Kaggle dataset is attached. Re-run this builder after editing a script.

Usage: python kaggle/build_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = ["generate_questions.py", "score_unifiedqa.py", "analyze_arm_a.py", "summarize_experiments.py"]


def md(text):
    return ("markdown", text.strip("\n"))


def code(text):
    return ("code", text.strip("\n"))


def script_cell(name):
    with open(os.path.join(ROOT, "scripts", name), encoding="utf-8") as f:
        return code(f"%%writefile /kaggle/working/f4/scripts/{name}\n" + f.read())


CELLS = [
    md("""
# Dodge et al. 2021 §4.3, Arm A: UnifiedQA ethnicity bias (Table 7)

End-to-end and resumable. Every scoring cell skips work whose results already exist, so re-running
the notebook (or attaching a dataset that already contains `results/`) does not redo finished runs.

| ID | Experiment | Status |
|---|---|---|
| E1 | CPU smoke tests (pipeline, resume, analysis) | done locally, recorded in EXPERIMENTS.md |
| E2 | CPU strided-sample check; found the single-attribute sampling bug | done locally |
| E3 | UnifiedQA-small, full extended question set (266,000 contexts) | done: **did not reproduce** |
| E4 | Size test: small / base / large / 3B on a 1-in-97 sample (1,516 contexts) | done: **only 3B reproduces** |
| E5 | Answer rule: summed vs per-token-mean log-likelihood (on E3, E4, E6) | done: no material difference |
| E6 | UnifiedQA-3B, the paper's full 147,000 contexts | **this notebook** |

Settings: GPU T4 x2, internet on. Run the cells top to bottom.

Safety checks (audit 2026-10-23): the analysis stops if scores do not match their questions (join check) or if a run
covers only part of the attributes or pair orders, e.g. because one GPU process died (coverage check). It also
reports how spread out the values are vs the paper, not only the rank agreement. UnifiedQA-11B is not tested:
it does not fit the free Kaggle / Colab tiers (~45 GB checkpoint).
"""),
    md("## 1. Setup: copy the dataset into a writable folder"),
    code("""
%%bash
SRC=$(dirname "$(dirname "$(find /kaggle/input -path '*scripts/generate_questions.py' | head -1)")")
echo "project folder: $SRC"
test -d "$SRC/third_party/unqover" || { echo "third_party/unqover not found under $SRC"; exit 1; }
mkdir -p /kaggle/working/f4
cp -rn "$SRC"/. /kaggle/working/f4/          # -n: never overwrite results already in /kaggle/working
pip install -q sentencepiece
nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md("""
## 2. Scripts (embedded from the repo)

These cells overwrite the dataset's copies, so the notebook always runs the current code.
"""),
    *[script_cell(n) for n in SCRIPTS],
    md("## 3. Questions (UnQover templates, no LLM, no C4 data)"),
    code("""
%%bash
cd /kaggle/working/f4
python scripts/generate_questions.py
# keeps the shipped files (scores are tied to their rows); --force regenerates
# expected: ethnicity 147000 contexts / 294000 questions (the paper's set); ethnicity_ext 266000 / 532000
"""),
    md("""
## 4. Move results from earlier sessions into the experiment layout

Earlier cells wrote to `results/unifiedqa-t5-small` (E3), `results_sample/` (E4) and `results_full/` (E6).
This moves them to `results/<experiment>/scores/<model>/`. Safe to run more than once.
"""),
    code("""
%%bash
cd /kaggle/working/f4
mkdir -p results          # a fresh session has no results yet; that is fine
mv_if () { if [ -d "$1" ] && [ ! -d "$2" ]; then mkdir -p "$(dirname "$2")"; mv "$1" "$2"; echo "moved $1 -> $2"; fi; }
mv_if results/unifiedqa-t5-small results/E3_full_small_ext/scores/unifiedqa-t5-small
for m in small base large 3b; do mv_if results_sample/unifiedqa-t5-$m results/E4_size_test_sample/scores/unifiedqa-t5-$m; done
mv_if results_full/unifiedqa-t5-3b results/E6_full_3b/scores/unifiedqa-t5-3b
echo "model results present:"; find results -mindepth 3 -maxdepth 3 -type d -name 'unifiedqa-*' | sort
"""),
    md("""
## 5. E3 (recorded, did not reproduce): UnifiedQA-small on the extended set

Kept for the record. Already complete if its chunks are present ("0 to do"); otherwise roughly 10 min on 2 x T4.
"""),
    code("""
%%bash
cd /kaggle/working/f4
for P in 0 1; do
  CUDA_VISIBLE_DEVICES=$P python scripts/score_unifiedqa.py --questions data/questions/ethnicity_ext.jsonl.gz \\
    --model allenai/unifiedqa-t5-small --out_dir results/E3_full_small_ext/scores/unifiedqa-t5-small \\
    --part $P --nparts 2 --batch_size 64 > logs_E3_$P.txt 2>&1 &
done
wait; tail -n 2 logs_E3_*.txt
"""),
    md("""
## 6. E4 (recorded): size test on a 1-in-97 sample

1,516 of the paper's 147,000 contexts (every pair, both orders, all 50 attributes). Stride 97 is
coprime with the 50 attributes and 14 templates; a stride of 50 picks a single attribute (bug found in E2).
About 10 min in total.
"""),
    code("""
%%bash
cd /kaggle/working/f4
Q=data/questions/ethnicity.jsonl.gz
run () { CUDA_VISIBLE_DEVICES=$2 python scripts/score_unifiedqa.py --questions $Q --model allenai/unifiedqa-t5-$1 \\
           --out_dir results/E4_size_test_sample/scores/unifiedqa-t5-$1 --part 0 --nparts 97 \\
           --batch_size $3 --gen_check 300 > logs_E4_$1.txt 2>&1; }
(run base 0 64; run small 0 64) &
(run large 1 32; run 3b 1 8) &
wait; tail -n 2 logs_E4_*.txt
"""),
    md("""
## 7. E6: UnifiedQA-3B on the paper's full 147,000 contexts

Roughly 3.5-4 h on 2 x T4 (5-6 contexts/s per GPU in E4). The cell blocks until both halves finish,
which keeps the session alive. If it stops (timeout, out of memory), re-run it: finished chunks are kept.
If a log shows `CUDA out of memory`, change `BS=16` to `BS=8`. Both log tails must say `done`: if only one
half finished, the analysis refuses the run (coverage check) instead of reporting half the attributes.
"""),
    code("""
%%bash
cd /kaggle/working/f4
BS=16
for P in 0 1; do
  CUDA_VISIBLE_DEVICES=$P python scripts/score_unifiedqa.py --questions data/questions/ethnicity.jsonl.gz \\
    --model allenai/unifiedqa-t5-3b --out_dir results/E6_full_3b/scores/unifiedqa-t5-3b \\
    --part $P --nparts 2 --batch_size $BS --gen_check 2000 > logs_E6_$P.txt 2>&1 &
done
wait; tail -n 3 logs_E6_*.txt
"""),
    md("""
## 8. Analysis of every experiment (both answer rules), then the summary table

E3 used the extended question file; E4 and E6 use the paper's 147,000-context file.
"""),
    code("""
import glob, os, subprocess
os.chdir("/kaggle/working/f4")
QUESTIONS = {"E3_full_small_ext": "data/questions/ethnicity_ext.jsonl.gz"}   # others: the paper's file
for exp in sorted(d for d in glob.glob("results/E*") if os.path.isdir(d)):
    name = os.path.basename(exp)
    models = sorted(d for d in glob.glob(f"{exp}/scores/unifiedqa-t5-*") if glob.glob(f"{d}/chunk_*.csv.gz"))
    if not models:
        continue
    for rule in ("sum", "mean"):
        r = subprocess.run(["python", "scripts/analyze_arm_a.py", "--scores", *models,
                            "--questions", QUESTIONS.get(name, "data/questions/ethnicity.jsonl.gz"),
                            "--out_dir", f"{exp}/analysis_{rule}", "--rule", rule, "--dpi", "300", "--no_merged"],
                           capture_output=True, text=True)
        print(f"{name} rule={rule}: {'ok' if r.returncode == 0 else 'FAILED'}")
        if r.returncode:
            print(r.stderr[-2000:])
print(subprocess.run(["python", "scripts/summarize_experiments.py"], capture_output=True, text=True).stdout)
"""),
    md("""
## 9. Figures and tables

In order: (1) Table 7 for all 15 groups; (2) for **every** group, how often it is the model's answer on positive vs
negative questions, and how it fares against each other group; (3) Jewish vs Arab, the pair the paper takes to the
corpus study (Arm B).
"""),
    code("""
import json, os
import pandas as pd
from IPython.display import Image, Markdown, display
os.chdir("/kaggle/working/f4")
show = lambda p, w=900: display(Image(p, width=w)) if os.path.exists(p) else print("missing:", p)

display(Markdown("### All experiments"))
display(pd.read_csv("results/experiments_summary.csv"))

display(Markdown("### E4 size test: why small/base/large fail and 3B reproduces"))
show("results/E4_size_test_sample/analysis_sum/fig_size_trend.png", 1200)
show("results/E4_size_test_sample/analysis_sum/fig_table7_all_models.png")

for exp, model in (("E6_full_3b", "unifiedqa-t5-3b"), ("E3_full_small_ext", "unifiedqa-t5-small")):
    d = f"results/{exp}/analysis_sum/{model}"
    if not os.path.exists(f"{d}/diagnostics.json"):   # not run, or the analysis refused it (see section 8)
        print("no finished analysis for", exp)
        continue
    display(Markdown(f"### {exp}: {model}"))
    # 1) the paper's Table 7, all 15 groups
    show(f"{d}/table7_paper_layout.png", 700)
    for f in ("fig_table7.png", "fig_scatter_vs_paper.png"):
        show(f"{d}/{f}")
    diag = json.load(open(f"{d}/diagnostics.json"))
    display(Markdown(f"Spread: ours {diag['spread_ours']} vs paper {diag['spread_paper']}; noise-corrected SD "
                     f"{diag['sd_ours_noise_corrected']} vs paper {diag['sd_paper']}; slope {diag['slope_ours_on_paper']}"))
    show(f"{d}/fig_by_attribute.png", 1000)
    # 2) every group: answers on positive vs negative questions, and every pairing
    display(Markdown(open(f"{d}/counts_by_ethnicity.md").read()))
    for f in ("fig_counts_by_ethnicity.png", "fig_opponents_all_groups.png"):
        show(f"{d}/{f}", 1100)
    # 3) the pair the paper takes to the corpus study (Arm B)
    for f in ("fig_jewish_vs_arab.png", "fig_jewish_arab_opponents.png", "fig_jewish_arab_by_class.png"):
        show(f"{d}/{f}")
    display(Markdown(open(f"{d}/jewish_vs_arab.md").read()))
"""),
    md("## 10. Download everything as one zip (Output panel, right sidebar)"),
    code("""
%%bash
cd /kaggle/working/f4
zip -qr /kaggle/working/f4_arm_a_results.zip results logs_*.txt -x '*scores_merged*'
ls -la /kaggle/working/f4_arm_a_results.zip
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
    with open(os.path.join(HERE, "f4_arm_a_end_to_end.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, "f4_arm_a_cells.txt"), "w", encoding="utf-8") as f:
        f.write("Paste each block into its own Kaggle cell (markdown blocks into markdown cells).\n"
                "Do not paste the '# ===== CELL' separator lines.\n\n" + "\n".join(lines))
    print(f"wrote {len(CELLS)} cells")


if __name__ == "__main__":
    main()
