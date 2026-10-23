# F3 reproduction: benchmark contamination (Dodge et al. 2021, §4.2, Table 2 / Table 4)

**Paper claim.** Benchmark test texts appear verbatim in C4.EN:
- 1.87–24.88% of generation targets (XSum, TIFU, WikiBio, AMR-to-text);
- 4.6% / 5.7% of LAMA T-REx / Google-RE probes;
- 1.8–53.6% of GLUE / BoolQ test inputs, with QNLI sentences the highest.

The paper matched "exact matches, normalized for capitalization and punctuation".

**Status (2026-10-23):** 6 headline fields reproduced in pattern; see [RESULTS.md](RESULTS.md). The ranking matches the
paper exactly, and the paper's value is inside our CI for 4 of 6 fields.

**Our method.** We query the free [infini-gram](https://infini-gram.io) API. Its `v4_c4train_llama` index covers
**all of C4.EN** (364,868,892 documents), so this is a full-corpus check, not a shard sample. A text counts as
contaminated if it occurs at least once.

## Differences from the paper (all ours, all stated)

- **Lower bounds.** infini-gram matches tokens exactly, so case and punctuation must agree; the paper normalized
  both. Two steps narrow the gap:
  - trailing punctuation is dropped before querying;
  - fully lowercased texts are also tried in sentence case.

  "Ours below the paper" is therefore expected; "ours above the paper" would be a real discrepancy.
- **Samples.** Each field is a seeded random sample (`--sample`, default 2,000; all texts for smaller sets), so
  every rate has a 95% Wilson CI. The paper reports no uncertainty.
- **Not built:**
  - **AMR-to-text** is licensed (LDC2017T10).
  - **WikiBio** exists only lowercased and tokenized, so a case-sensitive index cannot test it fairly. It is
    built only with `--wikibio` and reported as not comparable.
- **BoolQ.** The paper does not say which field was matched. We query the question (the paper's row) and the
  passage (`boolq_passage`, extra).
- **LAMA.** We use the paper's "template-generated sentences with a masked token that we fill in": LAMA's own
  templates filled with the subject and object labels.
- **TIFU.** As in the Hugging Face loader: the trimmed title (short) or the TL;DR (long), skipping posts with an empty
  body. The loader stores them lowercased, so each one is found in the original post and taken back in its original
  case. This works for 100% of rows.

## Dataset sizes vs Table 4

| Field | Ours | Paper |
|---|---|---|
| LAMA T-REx | 34,039 | 34,014 |
| LAMA Google-RE | 5,528 | 5,528 |
| XSum | 11,334 | 11,334 |
| TIFU-short / TIFU-long | 79,740 / 42,139 | 79,740 / 42,139 |
| BoolQ | 3,245 | 3,245 |
| CoLA, MNLI (mismatched test), MRPC, QNLI, RTE, SST-2, STS-B, WNLI | 1,063, 9,847, 1,725, 5,463, 3,000, 1,821, 1,379, 146 | the same |

## Positive control

The paper's Table 5 lists 10 XSum summaries that it found in C4. `analyze_contamination.py` finds each one in our
XSum test file and queries it. All 10 should be found; if they are not, the matching is broken.

## Preferred source once keys arrive: WIMBD

AI2's WIMBD Elasticsearch index (Elazar et al. 2023) holds C4 with a `subset` field. Its phrase search lowercases and
drops punctuation, which is the paper's own normalization, so it gives the paper's measure rather than a lower bound.
It also batches queries, so every field at full size takes minutes. The keys come from a form
(https://forms.gle/yMz7uTFhd1dKNYTk7). Put them in `es_config.yml` here (git-ignored), then:

```bash
pip install wimbd
python scripts/query_wimbd.py --out_dir results/W1_wimbd_full --sample 0
python scripts/analyze_contamination.py --counts results/W1_wimbd_full
```

`query_wimbd.py` has not been run yet, because we have no keys. Its output format matches the infini-gram script's.

## Running (infini-gram)

**Kaggle (recommended):** zip this folder (you can leave out `data/`, which is re-downloaded) and upload it as a
dataset. Import `kaggle/f3_contamination_kaggle.ipynb`, set Accelerator **None** and Internet **on**, then
Save & Run All. By default it queries the 6 fields behind the paper's headline claims (TIFU-short/long, XSum, LAMA
T-REx, QNLI sentence/question) at 300 texts each: about 1.5 hours. Add `--fields` to cover the rest later.

**Smoke test (C0, local, 30 texts per field for 4 fields):** no errors at 1 request per second. The positive control
found **8 of the paper's 10** Table 5 XSum summaries; the other two differ from C4's copy only in punctuation. So exact
matching recovers roughly 80% of the paper's normalized matches, and our rates should be read with that in mind.

**Locally:**

The API is a free shared service and blocks an IP (HTTP 403) that sends too many requests; a 4-way parallel smoke
test triggered this, and so did one request per second after ~50-65 requests. The client therefore sends one
request every 5 seconds by default, slows down further after any block, and tests texts in OR-groups (the API's
pure OR counts are exact; a group with count 0 resolves all its texts in one request). On the 6 key fields this
takes about 950 requests instead of ~1,800.
At that rate, 2,000 texts × 21 fields take about 12 hours. `--sample 500` takes about 3 hours.

```bash
python scripts/benchmarks.py                                  # downloads sources, writes data/benchmarks/ (~250 MB)
python scripts/query_infinigram.py --out_dir results/C1_infinigram --sample 2000     # resumable
python scripts/analyze_contamination.py --counts results/C1_infinigram
```

Outputs: `table2.csv/.md`, `fig_table2.png` (ours with CIs vs the paper's diamonds), `examples.md` (matched texts per
label field, as the paper's Table 5) and `positive_control.md`.
