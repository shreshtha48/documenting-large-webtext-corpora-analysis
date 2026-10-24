# Reproduction — Section 4.2: Benchmark Contamination

> **What this folder does in one sentence:**
> It checks whether standard AI evaluation tests (such as GLUE, XSum, and TIFU) were accidentally included
> in the C4 dataset used to train modern language models — a problem called *training contamination* or
> *benchmark leakage*.

**Paper section:** §4.2, Table 2 / Table 4  
**Status (2026-10-23):** 6 of 21 fields reproduced — the ranking of contamination rates matches the paper exactly. The paper's value falls inside our confidence interval for 4 of the 6 fields. See [RESULTS.md](RESULTS.md) for the full run log.

---

## Table of Contents

1. [Why This Matters — Plain-Language Summary](#1-why-this-matters--plain-language-summary)
2. [Glossary — Every Term Explained](#2-glossary--every-term-explained)
3. [Methodology — How the Code Works](#3-methodology--how-the-code-works)
4. [Results — What the Numbers Mean](#4-results--what-the-numbers-mean)
5. [Execution Guide — How to Run Everything](#5-execution-guide--how-to-run-everything)
6. [Standalone Checklist](#6-standalone-checklist)
7. [Redundant Data — What Should NOT Go on GitHub](#7-redundant-data--what-should-not-go-on-github)

---

## 1. Why This Matters — Plain-Language Summary

### The Problem: Did the Model Study the Exam Before Taking It?

Imagine a student who, before sitting an exam, was given access to the exact questions — not the answers, just the questions. If those same questions then appear on the actual exam, the student's score is inflated; it does not reflect genuine ability.

Language models face exactly this problem. They are trained on enormous amounts of text from the web. Researchers then *evaluate* these models by asking them questions from standard benchmarks (pre-made test sets like GLUE or XSum). But what if those test questions were already present in the training data?

If a model "saw" a benchmark test question during training, its score on that benchmark is unfair — not because it understood the question, but because it may have effectively memorized it.

**This experiment asks:** *How many sentences from standard AI benchmarks appear word-for-word in C4 — the dataset used to train many modern language models?*

### What We Found

| Benchmark | Contamination Rate (Paper) | Plain meaning |
|---|---|---|
| QNLI sentence | **53.6%** | More than half of all QNLI test sentences appeared in C4 training data |
| TIFU-short | 24.9% | About 1 in 4 Reddit titles appeared in C4 |
| XSum summary | 15.5% | 1 in 6 BBC news summaries appeared in C4 |
| LAMA T-REx | 4.7% | ~1 in 20 factual probes appeared in C4 |
| TIFU-long | 1.9% | ~1 in 50 Reddit TL;DRs appeared in C4 |
| QNLI question | 1.8% | ~1 in 55 QNLI questions appeared in C4 |

**Our reproduction (C1 run, 300 texts per field):** We matched the ranking perfectly. Our rates are slightly lower because the paper normalized capitalization and punctuation before matching, while our tool matches exactly — see [Section 3](#3-methodology--how-the-code-works) for why.

---

## 2. Glossary — Every Term Explained

| Term | Full form / Meaning |
|---|---|
| **C4** | Colossal Clean Crawled Corpus. A large English text dataset built from web pages, widely used to train language models. |
| **C4.EN** | The English-only portion of C4. Contains 364,868,892 documents. |
| **Benchmark contamination** | Also called *training leakage* or *data contamination*. Occurs when test examples from evaluation benchmarks appear in the training data, potentially inflating model scores. |
| **GLUE** | General Language Understanding Evaluation. A collection of 9 natural language tasks (CoLA, MNLI, MRPC, QNLI, RTE, SST-2, STS-B, WNLI) used to measure how well a model understands English. |
| **QNLI** | Question Natural Language Inference (part of GLUE). Tests whether a sentence from Wikipedia answers a specific question. We check both the *question* and the *sentence* (context paragraph) separately. |
| **CoLA** | Corpus of Linguistic Acceptability (part of GLUE). Sentences labeled as grammatically acceptable or not. |
| **MNLI** | Multi-Genre Natural Language Inference (part of GLUE). Pairs of sentences labeled entailment / contradiction / neutral. |
| **MRPC** | Microsoft Research Paraphrase Corpus (part of GLUE). Pairs of sentences labeled as paraphrase or not. |
| **RTE** | Recognizing Textual Entailment (part of GLUE). Tests whether one sentence follows logically from another. |
| **SST-2** | Stanford Sentiment Treebank v2 (part of GLUE). Movie review sentences labeled positive or negative. |
| **STS-B** | Semantic Textual Similarity Benchmark (part of GLUE). Sentence pairs scored for similarity on a 0–5 scale. |
| **WNLI** | Winograd Natural Language Inference (part of GLUE). Tests pronoun resolution. |
| **XSum** | Extreme Summarization. A dataset of 11,334 BBC news articles, each paired with a one-sentence summary written by the BBC journalist. We check whether that summary appears in C4. |
| **TIFU** | "Today I F***ed Up." A Reddit community. Two fields: *TIFU-short* (post title, ~7 words) and *TIFU-long* (TL;DR summary at the end of the post). |
| **LAMA** | Language Model Analysis. A benchmark that probes what factual knowledge is "stored" inside a language model using fill-in-the-blank templates (e.g., "The capital of France is ___"). |
| **T-REx** | A subset of LAMA. 34,039 factual templates derived from Wikidata triples. |
| **Google-RE** | Another LAMA subset. 5,528 templates about birth dates, birth places, and employer relationships from Google's Relation Extraction corpus. |
| **BoolQ** | Boolean Question. 3,245 yes/no questions paired with Wikipedia passages. We check both the question text and the Wikipedia passage separately. |
| **WikiBio** | A dataset of biography-style text generated from Wikipedia infoboxes. Text is stored lowercased and tokenized (e.g., parentheses appear as `-lrb-`), making it incompatible with our case-sensitive index. Reported as "not comparable." |
| **AMR-to-text** | Abstract Meaning Representation to Text. Licensed (LDC2017T10), not freely available. Not tested. |
| **infini-gram** | A free public search engine that indexes all of C4.EN (364,868,892 documents) and can instantly answer "does this exact phrase appear anywhere in C4?" Developed by UW NLP. Website: https://infini-gram.io |
| **WIMBD** | What's In My Big Data? A tool from AI2 (Allen Institute for Artificial Intelligence) with an Elasticsearch index over C4 that normalizes capitalization and punctuation before matching — exactly as the paper did. Requires an API key (form: https://forms.gle/yMz7uTFhd1dKNYTk7). |
| **infini-gram index `v4_c4train_llama`** | The specific infini-gram index covering the full C4.EN training set (all 364,868,892 documents). This is the correct index to use for this reproduction. |
| **Lower bound** | Our contamination rates are *at least as large* as the true value under the paper's normalization, but they may be *smaller* (hence "lower bound"). This is because we match text exactly while the paper first normalized capitalization and punctuation. Exact matching misses cases where C4 stores the text with slightly different formatting. |
| **OR-query / group testing** | A trick where multiple texts are sent to infini-gram in one API call as an "any of these" request. If the group count is zero, none of the texts are contaminated — all resolved at once. If non-zero, the group is binary-split to find which texts matched. Saves ~35–40% of API calls. |
| **95% Wilson confidence interval (CI)** | A statistically calculated range that has a 95% chance of containing the true contamination rate. Written as `[low, high]`. Wider when the sample is small. The paper reports no uncertainty; we report CIs because we use random samples rather than full datasets. |
| **Positive control** | A sanity check. The paper's Table 5 lists 10 specific XSum summaries that the authors found in C4. We search for those same 10 texts. If we find them, our pipeline is working correctly. If we miss them, something is broken. (We found 8/10; the 2 misses differ only in punctuation.) |
| **Spearman rank correlation** | A number between −1 and +1 measuring whether two ranked lists agree in order (not necessarily in value). A Spearman of 1.0 means perfect rank agreement. Our six-field ranking has Spearman 1.0 vs the paper. |
| **HTTP 403** | An error code meaning "Forbidden" — the infini-gram API returns this when a client sends too many requests too quickly (rate-limited). The client automatically waits 10 minutes and retries. |
| **HuggingFace (HF)** | Hugging Face. An AI platform that hosts datasets and model weights. We download benchmark datasets from here. |
| **JSONL / JSONL.GZ** | JSON Lines: one JSON object per line (easy to stream). `.gz` means gzip-compressed. Our intermediate files use this format. |
| **Wilson CI** | Short for Wilson confidence interval. See "95% Wilson confidence interval" above. |
| **T4** | An NVIDIA GPU card (Tesla T4). This experiment does NOT need a GPU — all computation is network API calls. |

---

## 3. Methodology — How the Code Works

There are three scripts, run in sequence:

```
scripts/
  benchmarks.py            ← Step 1: Download benchmark texts
  query_infinigram.py      ← Step 2: Check each text against C4 via the API
  analyze_contamination.py ← Step 3: Count hits, compute rates, make figures
```

---

### Step 1 — `benchmarks.py`: Gathering the test questions

**What it does:** Downloads the standard benchmark datasets from HuggingFace and saves each field (a specific column from a specific dataset) as a `.jsonl.gz` file in `data/benchmarks/`. Each line is one text to be checked.

**Special handling:**
- **TIFU:** HuggingFace stores TIFU titles in lowercase. We look up each title in the original Reddit post data to recover the original casing before querying.
- **LAMA:** We fill in the LAMA "template" with the actual subject and object values to form a complete sentence (e.g., `"Marie Curie was born in Warsaw."`) rather than querying the template itself.
- **WikiBio:** Only built if you pass `--wikibio` flag. The data is lowercased + tokenized, making it incompatible with a case-sensitive index. Results are reported as "not comparable."
- **AMR-to-text:** Skipped — requires a paid LDC license.

**Output files** (written to `data/benchmarks/`):

| File | Contents | Count |
|---|---|---|
| `lama_trex.jsonl.gz` | Factual probe sentences (T-REx) | 34,039 |
| `lama_google_re.jsonl.gz` | Factual probe sentences (Google-RE) | 5,528 |
| `xsum.jsonl.gz` | BBC one-sentence summaries | 11,334 |
| `tifu_short.jsonl.gz` | Reddit post titles | 79,740 |
| `tifu_long.jsonl.gz` | Reddit TL;DR summaries | 42,139 |
| `boolq.jsonl.gz` + `boolq_passage.jsonl.gz` | Yes/no questions + Wikipedia paragraphs | 3,245 each |
| `cola.jsonl.gz`, `mnli.jsonl.gz`, `mrpc.jsonl.gz`, `qnli_sentence.jsonl.gz`, `qnli_question.jsonl.gz`, `rte.jsonl.gz`, `sst2.jsonl.gz`, `stsb.jsonl.gz`, `wnli.jsonl.gz` | GLUE test split inputs | varies |

---

### Step 2 — `query_infinigram.py`: Checking each text against all of C4

**What it does:** For each text in each benchmark field, it sends an API request to infini-gram asking:

> "Does this exact sequence of words appear anywhere in the 364,868,892 documents of C4.EN?"

If the answer is "yes (count > 0)", the text is **contaminated**. If "no (count = 0)", it is **clean**.

**Key design decisions:**

#### OR-group testing (saves API calls)
Rather than sending one text per request, we send a batch of texts and ask "does *any* of these appear in C4?" If the answer is zero — none of them are contaminated — we resolve the whole batch in one request. If non-zero, we binary-split the batch to find which specific texts matched. This reduces the number of API calls by roughly 35–40%.

#### Two query variants per text
To partially compensate for not normalizing like the paper did, each text is tried in two forms:
1. **Original text** with trailing punctuation stripped (e.g., remove a trailing period).
2. **Sentence-cased version** of a fully lowercased text (capitalize only the first letter).

If either variant is found, the text counts as contaminated.

#### Rate limiting
The infini-gram API is a free shared service. Sending too many requests too fast (faster than ~1 per second) triggers an IP block (HTTP 403). Our client sends **one request every 5 seconds** by default. A block triggers a 10-minute automatic pause before resuming.

**Never run multiple instances of `query_infinigram.py` in parallel** — this will get your IP blocked.

#### Resumability
Every result is appended to `counts_<field>.jsonl` immediately. If the script stops (network error, timeout, keyboard interrupt), re-running it picks up where it left off and skips already-processed texts.

**Output** (written to `--out_dir`, e.g., `results/C1_infinigram_key6_300/`):

| File | Contents |
|---|---|
| `counts_qnli_sentence.jsonl` | One line per queried text: `{"id": ..., "text": "...", "found": true/false}` |
| `counts_tifu_short.jsonl` | Same format |
| … one file per field … | |

---

### Step 3 — `analyze_contamination.py`: Computing rates and making figures

**What it does:** Reads all the `counts_*.jsonl` files, computes the contamination rate for each field, and generates comparison tables and figures.

**Contamination rate formula:**

$$\text{contamination rate} = \frac{\text{number of texts found in C4}}{\text{total texts sampled}}$$

For example: if we checked 300 TIFU-short titles and 62 were found in C4, the rate is 62/300 = 20.7%.

**Confidence interval:** Because we sample (e.g., 300 texts out of 79,740), there is uncertainty. We compute a 95% Wilson confidence interval — a range `[low%, high%]` that has a 95% chance of containing the true rate for the full dataset. The Wilson method is preferred over a simple ±margin because it works accurately even for very high or very low rates.

**Outputs** (written to `--counts_dir/analysis/`):

| File | Contents |
|---|---|
| `table2.md` | Markdown table: our rate [CI] vs paper rate, for each field |
| `table2.csv` | Same data, machine-readable |
| `fig_table2.png` | Bar chart of our rates with CI error bars, paper values shown as diamonds |
| `examples.md` | Up to 5 contaminated text examples per field (mimics paper's Table 5) |
| `positive_control.md` | Results of looking up the paper's 10 known-contaminated XSum summaries |

---

### Alternative Method: WIMBD (not yet run — API keys required)

The paper's own normalization (lowercase + strip punctuation before matching) can be replicated exactly using AI2's **WIMBD** Elasticsearch index. Unlike infini-gram's exact matching, WIMBD pre-normalizes all C4 text the same way the paper did. This would give our rates = the paper's rates rather than a lower bound.

**How to get access:** Fill in the form at https://forms.gle/yMz7uTFhd1dKNYTk7. You will receive an API key. Place it in a file called `es_config.yml` in this folder (already git-ignored).

**How to run it once you have keys:**
```bash
pip install wimbd
python scripts/query_wimbd.py --out_dir results/W1_wimbd_full --sample 0
python scripts/analyze_contamination.py --counts results/W1_wimbd_full
```

`--sample 0` means "use the full dataset, not a random sample." The output format is identical to the infini-gram output, so `analyze_contamination.py` works for both.

---

## 4. Results — What the Numbers Mean

### Our Run C1 (2026-10-23): 6 fields × 300 texts

**Setup:** Ran locally on a laptop. 1,170 API calls at 1 request/5 seconds. Total time: ~1.6 hours. Zero blocks, zero errors.

| Field | What is being checked | Our rate | 95% CI | Paper rate | Paper inside our CI? |
|---|---|---|---|---|---|
| QNLI sentence | A sentence from a Wikipedia paragraph | **41.3%** | [35.9, 47.0] | 53.6% | No — below, as expected for a lower bound |
| TIFU-short | A Reddit post title (~7 words) | **20.7%** | [16.5, 25.6] | 24.9% | **Yes** ✓ |
| XSum summary | A one-sentence BBC news summary | **6.3%** | [4.1, 9.7] | 15.5% | No — larger gap, see note below |
| LAMA T-REx | A factual sentence (e.g., "Einstein was born in Ulm") | **3.3%** | [1.8, 6.0] | 4.7% | **Yes** ✓ |
| TIFU-long | A Reddit TL;DR summary | **2.3%** | [1.1, 4.7] | 1.9% | **Yes** ✓ |
| QNLI question | A natural language question | **1.0%** | [0.3, 2.9] | 1.8% | **Yes** ✓ |

**Spearman rank correlation with the paper:** 1.0 (perfect — the ranking of fields is identical)

---

### Interpreting Each Row

#### QNLI sentence (41.3% vs paper's 53.6%)
Wikipedia text has been crawled heavily into C4. QNLI sentences are extracted directly from Wikipedia, so it is not surprising that roughly half appear in C4. Our rate (41.3%) is below the paper's (53.6%) because Wikipedia articles frequently use special formatting characters (dashes, quotation marks) that differ subtly between the Wikipedia source and C4's version. The paper's normalization removes these differences; ours does not.

**Implication:** Models trained on C4 and evaluated on QNLI have seen more than half the test sentences during training.

#### TIFU-short (20.7%; the paper's 24.9% is inside our CI ✓)
Reddit post titles are short and casual. About 1 in 4 titles made it into C4. Our rate of 20.7% is consistent with the paper's 24.9% — the slight difference is normal sampling variation.

**Implication:** Models have likely "seen" roughly 1 in 4 TIFU-short test titles during training.

#### XSum summary (6.3% vs paper's 15.5%)
The largest gap between our rates and the paper's. BBC journalism uses typographic conventions — curly quotation marks `" "`, em-dashes `—`, ellipses `…` — that differ from the plain ASCII characters C4 normally stores. The paper's normalization collapses these to the same character; exact matching treats them as different and misses them.

**Evidence:** The positive control (see below) found 8 of the paper's 10 known matches. The 2 it missed differ from C4's copy only in punctuation marks. This suggests our XSum rate is really closer to 6.3% / 0.8 ≈ **~7.9%** true coverage.

#### LAMA T-REx (3.3%; the paper's value is inside our CI ✓)
Factual probe sentences are short and structured. Our rate of 3.3% is consistent with the paper's 4.7%.

**Implication:** Knowledge probes that appear in C4 could be answered from memorization rather than reasoning.

#### TIFU-long (2.3%; the paper's 1.9% is inside our CI ✓)
Longer Reddit summaries (TL;DRs) are less likely to appear verbatim in C4. Our rate of 2.3% is consistent with the paper's 1.9%.

#### QNLI question (1.0%; the paper's 1.8% is inside our CI ✓)
Natural language questions rarely appear verbatim on the web. 1% is low but not zero.

---

### Positive Control: Finding the Paper's Own Examples

The paper's Table 5 lists 10 specific XSum summaries it found in C4. Our `analyze_contamination.py` automatically searches for all 10 and reports results in `positive_control.md`.

- **8 of 10 found** ✓ — our pipeline is working.
- **2 missed** — those 2 differ from C4's stored copy only in punctuation (e.g., a curly quote `'` in the benchmark vs a straight quote `'` in C4).

**Key takeaway:** on these 10 XSum summaries, exact matching found 8 of the paper's normalized matches. That is one
field and 10 texts, so it does not give a reliable correction factor for the other fields: scaling XSum by 1/0.8
still leaves 7.9% vs the paper's 15.5%. Treat our rates as lower bounds; the WIMBD index, which normalizes the way the
paper did, is the way to close the gap (`scripts/query_wimbd.py`, keys pending).

---

### Fields Not Yet Covered

The paper's Table 4 has 21 fields total. Our C1 run covered 6. The remaining 15 can be added by re-running the query script with more `--fields` arguments. All completed texts are already saved and will not be re-queried.

**Not tested, not reproducible:**
- **WikiBio:** stored lowercased + tokenized, incompatible with a case-sensitive index.
- **AMR-to-text:** requires LDC license (LDC2017T10), not freely available.

---

## 5. Execution Guide — How to Run Everything

### Option A: Kaggle (Recommended — No GPU Needed)

Kaggle is a free cloud computing platform. This notebook uses only the CPU (no GPU required) but needs Internet access to reach the infini-gram API.

#### Step-by-step:

**A1. Prepare the dataset zip**

Zip the `benchmark-contamination` folder. You can leave out the `data/` subdirectory (all benchmark texts are re-downloaded automatically).

```
benchmark-contamination.zip
└── benchmark-contamination/
    ├── scripts/
    │   ├── benchmarks.py
    │   ├── query_infinigram.py
    │   └── analyze_contamination.py
    ├── kaggle/
    │   └── f3_contamination_kaggle.ipynb
    ├── results/        (leave out — large, re-generated by the notebook)
    └── ...
```

**A2. Upload as a Kaggle dataset**

1. Go to https://www.kaggle.com/datasets
2. Click **New Dataset**
3. Upload `benchmark-contamination.zip`
4. Name the dataset: `benchmark-contamination`
5. Set visibility to **Private**, then click **Create**

**A3. Import the notebook**

1. Go to https://www.kaggle.com/code
2. Click **New Notebook**
3. Click **File → Import Notebook**
4. Upload `kaggle/f3_contamination_kaggle.ipynb`

**A4. Configure the notebook**

| Setting | Value |
|---|---|
| Accelerator | **None** (no GPU needed — this is API calls only) |
| Internet | **On** (required — the notebook contacts infini-gram) |
| Session type | **Save & Run All** (headless background run) |

**A5. Add the dataset**

1. In the right panel, click **Add Input → Datasets**
2. Search for your dataset `benchmark-contamination`
3. Click **Add**

**A6. Run**

Click **Save Version → Save & Run All (Commit)**. Close the browser tab — it continues running in the background.

**Estimated time:** ~1.5 hours for the 6 default fields × 300 texts each.

**A7. Download results**

When finished, go to the **Output** tab. Download `f3_results.zip`. Unzip into `results/` in this folder.

---

> **⚠️ Rate limit warning:** Never run two copies of the notebook simultaneously (on Kaggle or anywhere). Parallel requests will trigger an IP block (HTTP 403). If blocked, the script automatically waits 10 minutes and resumes.

---

### Option B: Google Colab (Free, CPU)

Google Colab is Google's free cloud notebook environment. CPU is sufficient; no GPU is needed.

> **Note:** If a `colab/` subfolder exists here, use the `.ipynb` file inside it. If not, follow the steps below to run the scripts manually.

**B1. Open Colab**

Go to https://colab.research.google.com → **New notebook**

**B2. Clone the repository**

```python
!git clone https://github.com/<your-org>/safl-work-test.git /content/repo
%cd /content/repo/reproduction/benchmark-contamination
```

Replace `<your-org>` with the actual GitHub organization or username.

**B3. Install dependencies**

```bash
!pip install datasets requests tqdm
```

**B4. Run the three scripts in sequence**

```bash
# Download benchmark texts (~5 minutes, ~250 MB)
!python scripts/benchmarks.py

# Query infini-gram — 6 key fields, 300 texts each (~1.5 hours)
# This is resumable: safe to re-run if interrupted
!python scripts/query_infinigram.py --out_dir results/C1_infinigram --fields tifu_short tifu_long xsum lama_trex qnli_sentence qnli_question --sample 300

# Analyse results and generate figures
!python scripts/analyze_contamination.py --counts results/C1_infinigram
```

**B5. Download results**

```python
from google.colab import files
import zipfile, os
zipfile.ZipFile('results.zip', 'w').write('results/')
files.download('results.zip')
```

> **Note:** Colab sessions have a runtime limit (12 hours for free tier). The query script is resumable — if the session times out, restart and re-run from Step B4; it will skip already-queried texts.

---

### Option C: Local Machine

**Prerequisites:**

- Python 3.8 or newer
- Internet access (to reach the infini-gram API at `api.infini-gram.io`)
- No GPU required

**C1. Install dependencies**

```bash
pip install datasets requests tqdm
```

**C2. Run the three scripts**

```bash
# From the benchmark-contamination/ directory:

# Step 1: Download benchmark texts (once only, ~250 MB)
python scripts/benchmarks.py

# Step 2: Query the infini-gram API
# --sample 300 : use 300 random texts per field (faster)
# --sample 0   : use all texts per field (full reproduction, ~12 hours)
# --fields     : which fields to run (default: all 21)
python scripts/query_infinigram.py \
    --out_dir results/C1_infinigram \
    --fields tifu_short tifu_long xsum lama_trex qnli_sentence qnli_question \
    --sample 300

# Step 3: Compute rates, generate figures
python scripts/analyze_contamination.py --counts results/C1_infinigram
```

**⏱️ Time estimates:**

| Configuration | Estimated time |
|---|---|
| 6 fields × 300 texts (default) | ~1.5 hours |
| 6 fields × 2,000 texts | ~9–10 hours |
| All 21 fields × 2,000 texts | ~30–40 hours |

> **Do not run multiple terminal windows** pointing at `query_infinigram.py` — the infini-gram API rate-limits by IP.

---

### Option D: Rebuild the Kaggle Notebook from Scratch

If you modify any of the scripts under `scripts/` and want the Kaggle notebook to reflect your changes, regenerate it:

```bash
# From the benchmark-contamination/ directory:
python kaggle/build_kaggle_notebook.py
```

This rewrites `kaggle/f3_contamination_kaggle.ipynb` by embedding the current contents of the three scripts directly into notebook cells. The updated notebook can then be re-imported into Kaggle.

---

### WIMBD Alternative (When API Keys Are Available)

If you have obtained WIMBD API keys (form: https://forms.gle/yMz7uTFhd1dKNYTk7):

1. Place the key file in `es_config.yml` (this file is already git-ignored — it will not be committed).
2. Run:

```bash
pip install wimbd
python scripts/query_wimbd.py --out_dir results/W1_wimbd_full --sample 0
python scripts/analyze_contamination.py --counts results/W1_wimbd_full
```

WIMBD normalizes capitalization and punctuation the same way the paper did. Rates from WIMBD should exactly match (or be very close to) the paper's Table 4 values — no "lower bound" caveat needed.

---

## 6. Standalone Checklist

| Item | Status | Notes |
|---|---|---|
| **Kaggle notebook runs standalone** | ✅ PASS | CPU + Internet ON. Downloads all data and scripts from attached dataset. ~1.5 hours. |
| **No manual setup needed on Kaggle** | ✅ PASS | All dependencies installed inside the notebook. |
| **Resumable after crash** | ✅ PASS | `query_infinigram.py` appends results immediately. Re-run skips done texts. |
| **`benchmarks.py` must run first** | ⚠️ Required first step | Must run before `query_infinigram.py`. Already handled by the Kaggle notebook. |
| **GPU required** | ❌ NOT NEEDED | All work is API calls over the network. CPU only. |
| **Internet required** | ✅ Required | Must reach `api.infini-gram.io`. Kaggle: set Internet ON. |
| **WIMBD path requires API keys** | ⚠️ Blocked | Keys from AI2 form. `query_wimbd.py` is written but never run. |
| **AMR-to-text** | ❌ Not reproducible | Requires LDC2017T10 license. Skip. |
| **WikiBio** | ⚠️ Not comparable | Data is lowercased/tokenized. Results are not comparable to paper. |
| **`data/benchmarks/` included in zip** | Optional | ~250 MB. Can be excluded — re-downloaded automatically. |

---

## 7. Redundant Data — What Should NOT Go on GitHub

The following files are large and re-generatable. **Do not commit them.** They are already listed in `.gitignore`.

| Path | Size | Why excluded |
|---|---|---|
| `data/benchmarks/*.jsonl.gz` | ~250 MB total | Re-downloaded by `benchmarks.py` from HuggingFace |
| `data/downloads/` | Varies | Raw source files (TIFU Reddit data, etc.) downloaded temporarily |
| `results/*/counts_*.jsonl` | Varies | Generated by `query_infinigram.py`; re-runnable |
| `results/*/analysis/` | Small (~5 MB) | Generated by `analyze_contamination.py`; re-runnable |
| `es_config.yml` | Tiny | API keys — must never be committed |
| `hf_cache/` | Varies | HuggingFace model/dataset cache; re-downloadable |
| `__pycache__/` | Tiny | Python bytecode cache; auto-generated |

**What SHOULD be committed:**
- All `.py` scripts in `scripts/`
- The Kaggle notebook `kaggle/f3_contamination_kaggle.ipynb`
- The notebook builder `kaggle/build_kaggle_notebook.py`
- `RESULTS.md` (the run log with our actual numbers)
- This `README.md`

---

*This README was written assuming the reader has no prior knowledge of NLP, machine learning, or the tools used. Every acronym is defined on first use. Run instructions assume a fresh machine with only Python installed.*
