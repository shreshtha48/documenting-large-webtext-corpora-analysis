# F3 results log (Dodge et al. 2021 §4.2, Table 2 / Table 4)

Every run, reproduced or not. Dates: 2026-10-23. Source: infini-gram API, index `v4_c4train_llama` (all of C4.EN,
364,868,892 documents). The matching is exact and case- and punctuation-sensitive, so every rate is a **lower bound**
on the paper's normalized matching.

| ID | What | Where | Outcome | Results |
|---|---|---|---|---|
| C0 | Smoke test: 30 texts × 4 fields, then the positive control | laptop, 1 request/s | Pipeline works. Positive control: **8 of 10** of the paper's Table 5 XSum summaries found (the 2 misses differ only in punctuation). An earlier 4-way parallel attempt got the IP blocked (HTTP 403) for a few minutes | `results/C0_smoke/` |
| — | First Kaggle run, 6 key fields × 300 | Kaggle CPU | **Stopped** after 217 TIFU-short texts: the client crashed on an unhandled network/response error (fixed afterwards: every failure retried). Partial TIFU-short 20.7% [15.9–26.6] | not kept (Kaggle output) |
| — | Local run at 1 request/s | laptop | Blocked every ~50–65 requests (10-min pauses), which projected ~5–6 h; stopped after 114 texts, which were kept and reused by C1 | merged into C1 |
| C1 | **6 headline fields × 300 texts**, OR-group testing, one request per 5 s | laptop | **Reproduces the pattern.** 1,170 requests (vs ~1,800 one by one), 0 blocks, 0 errors, ~1.6 h | `results/C1_infinigram_key6_300/` |

## C1 numbers

| Field | Ours (95% Wilson CI) | Paper (Table 4) | Paper inside our CI? |
|---|---|---|---|
| QNLI sentence (input) | 41.3% [35.9, 47.0] | 53.6% (2,931/5,463) | no (below) |
| TIFU-short (label) | 20.7% [16.5, 25.6] | 24.9% (19,843/79,740) | **yes** |
| XSum (label) | 6.3% [4.1, 9.7] | 15.5% (1,756/11,334) | no (below) |
| LAMA T-REx (label) | 3.3% [1.8, 6.0] | 4.7% (1,585/34,014) | **yes** |
| TIFU-long (label) | 2.3% [1.1, 4.7] | 1.9% (790/42,139) | **yes** |
| QNLI question (input) | 1.0% [0.3, 2.9] | 1.8% (97/5,463) | **yes** |

What this shows:
- **The headline claim reproduces.** The ranking of the six fields is identical to the paper's (Spearman 1.0), from
  QNLI sentences (about half found) down to TIFU-long and QNLI questions (about 1–2%).
- **The paper's value is inside our CI for 4 of 6 fields.** The other two (XSum, QNLI sentence) are below the paper,
  the direction a lower bound predicts. No field is above the paper.
- **XSum's gap is larger than the positive control suggests** (6.3% vs 15.5%, though 8 of 10 known matches were
  found). Likely cause: BBC summaries appear in C4 with small typographic differences (curly vs straight quotes,
  dashes) that the paper's normalization ignored and exact matching does not. QNLI's Wikipedia sentences likely have
  the same issue. The WIMBD index (`scripts/query_wimbd.py`, keys pending) would test this directly, because it
  normalizes the way the paper did.

Not covered: the other 15 Table 4 fields (BoolQ, CoLA, MNLI, MRPC, RTE, SST-2, STS-B, WNLI, Google-RE, WikiBio,
AMR-to-text). They can be added with `--fields`; finished texts are kept.
