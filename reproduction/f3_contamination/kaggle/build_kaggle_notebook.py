"""Build the Kaggle notebook for Dodge et al. 2021 §4.2 (benchmark contamination) from the scripts here.

The work is API calls to infini-gram at one per second, so it needs no GPU: run it with Accelerator "None".
"Save Version -> Save & Run All" runs it in the background (up to 12 h).

Writes, next to this file:
    f3_contamination_kaggle.ipynb   import into Kaggle (File -> Import Notebook)
    f3_contamination_cells.txt      the same cells as plain text

Usage: python kaggle/build_kaggle_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = ["benchmarks.py", "query_infinigram.py", "analyze_contamination.py"]
W = "/kaggle/working/f3"
OUT = "/kaggle/working/f3/results/C1_infinigram_key6_300"
# the paper's headline range: lowest and highest label and input fields, XSum (Table 5) and LAMA T-REx
KEY_FIELDS = "tifu_short tifu_long xsum lama_trex qnli_sentence qnli_question"


def md(text):
    return ("markdown", text.strip("\n"))


def code(text):
    return ("code", text.strip("\n"))


def script_cell(name):
    with open(os.path.join(ROOT, "scripts", name), encoding="utf-8") as f:
        return code(f"%%writefile {W}/scripts/{name}\n" + f.read())


CELLS = [
    md("""
# Dodge et al. 2021 §4.2: benchmark texts found in C4.EN (Table 2 / Table 4)

For each benchmark field (XSum summaries, TIFU, LAMA probes, GLUE and BoolQ test inputs) we ask the free
[infini-gram](https://infini-gram.io) API whether the text occurs in **all of C4.EN** (364,868,892 documents). The
paper matched texts after normalizing case and punctuation; infini-gram matches exactly, so our rates are **lower
bounds**. A positive control re-finds the paper's own Table 5 XSum examples: 8 of 10 were found in a local test (the
other 2 differ only in punctuation), so exact matching recovers roughly 80% of the paper's matches.

**Settings:** Accelerator **None** (no GPU needed), Internet **on**. Then **Save Version -> Save & Run All**; the
browser can be closed. About 35-40 min: the 6 fields behind the paper's headline claims, 300 texts each
(about 1,800 API calls at one per second):

| Field | Paper | Why |
|---|---|---|
| TIFU-short / TIFU-long | 24.88% / 1.87% | highest and lowest summary targets |
| XSum | 15.49% | the paper's worked example (Table 5) |
| LAMA T-REx | 4.6% | knowledge probes |
| QNLI sentence / question | 53.6% / 1.8% | highest and lowest inputs |

95% CIs: about ±5 points for the high rates, about ±1.5 for the low ones. Other fields can be added later by
re-running section 4 with more `--fields` (finished texts are kept). The API blocks
IPs that query too fast, so please keep `--rate 1`.

When it finishes, download `f3_results.zip` from the Output tab.
"""),
    md("## 1. Setup: copy the dataset into a writable folder"),
    code(f"""
%%bash
SRC=$(dirname "$(dirname "$(find /kaggle/input -path '*scripts/query_infinigram.py' | head -1)")")
echo "project folder: $SRC"
test -f "$SRC/scripts/query_infinigram.py" || {{ echo "attach the f3_contamination dataset (Add Input)"; exit 1; }}
mkdir -p {W}
cp -rn "$SRC"/. {W}/          # -n: never overwrite results already in /kaggle/working
curl -s -o /dev/null -w "infini-gram API: HTTP %{{http_code}}\\n" -X POST https://api.infini-gram.io/ \\
  -H "Content-Type: application/json" \\
  -d '{{"index":"v4_c4train_llama","query_type":"count","query":"Grimsby fought back from two goals down"}}'
"""),
    md("## 2. Scripts (embedded from the repo, so the notebook runs the current code)"),
    *[script_cell(n) for n in SCRIPTS],
    md("""
## 3. Build the benchmark texts (about 5 min, ~250 MB of downloads)

Expected counts match the paper's Table 4: XSum 11,334; TIFU-short 79,740; TIFU-long 42,139; Google-RE 5,528;
BoolQ 3,245; GLUE test sets as in the paper (T-REx 34,039 vs the paper's 34,014).
"""),
    code(f"""
%%bash
cd {W}
python scripts/benchmarks.py 2>&1 | grep -v downloading
"""),
    md("""
## 4. Query infini-gram, 6 key fields (about 35-40 min)

One request per second; a block (HTTP 403) pauses the run for 10 minutes and continues; network errors are retried.
Progress is printed every 50 texts. If the script still exits early, the cell restarts it (up to 3 times) and it
resumes where it stopped; re-running this cell does the same.
"""),
    code(f"""
%%bash
cd {W}
for attempt in 1 2 3; do
  python scripts/query_infinigram.py --out_dir {OUT} --fields {KEY_FIELDS} --sample 300 --rate 1 && break
  echo "query run stopped (attempt $attempt); resuming in 60 s"; sleep 60
done
"""),
    md("## 5. Analysis: Table 2 vs the paper, figure, matched examples, positive control"),
    code(f"""
%%bash
cd {W}
python scripts/analyze_contamination.py --counts {OUT}
"""),
    code(f"""
import os
from IPython.display import Image, Markdown, display
A = "{OUT}/analysis"
for f in ("table2.md", "positive_control.md"):
    p = f"{{A}}/{{f}}"
    display(Markdown(open(p).read()) if os.path.exists(p) else Markdown(f"missing: {{p}}"))
if os.path.exists(f"{{A}}/fig_table2.png"):
    display(Image(f"{{A}}/fig_table2.png", width=900))
"""),
    md("## 6. Download"),
    code(f"""
%%bash
cd {W}/results
zip -qr /kaggle/working/f3_results.zip C1_infinigram_key6_300
ls -la /kaggle/working/f3_results.zip
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
    with open(os.path.join(HERE, "f3_contamination_kaggle.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, "f3_contamination_cells.txt"), "w", encoding="utf-8") as f:
        f.write("Paste each block into its own Kaggle cell (markdown blocks into markdown cells).\n"
                "Do not paste the '# ===== CELL' separator lines.\n\n" + "\n".join(lines))
    print(f"wrote {len(CELLS)} cells")


if __name__ == "__main__":
    main()
