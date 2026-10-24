# Results: popular benchmarks in C4.EN

All runs are logged, including the parts that did not work.

## E1. 18 fields × 200 texts, infini-gram exact match (2026-10-24, local, ~2.5 h at the API's pace)

Output: `results/E1_infinigram_200/` (counts per field, `analysis/summary.md`, `fig_by_field.png`).

| Group | All texts | 8+ words | 10+ words |
|---|---|---|---|
| Web text, before Apr 2019 | 3.7% [2.7, 5.2] (33/889) | 3.5% [2.4, 4.9] | 2.9% [1.9, 4.3] (23/794) |
| Written before Apr 2019 | 1.5% [0.7, 3.2] (6/400) | 0.5% [0.1, 1.9] | 0.6% [0.2, 2.0] (2/360) |
| **Written after Apr 2019 (negative control)** | 2.5% [2.0, 3.3] (56/2,200) | 0.6% [0.4, 1.1] | **0.0% [0.0, 0.3] (0/1,524)** |

**Per field (all texts):**

| Field | Rate |
|---|---|
| BOLD Wikipedia sentences | 7.0% [4.2, 11.4] |
| RealToxicityPrompts sentences | 6.5% [3.8, 10.8] |
| MMLU questions | 3.2% |
| ARC-Challenge questions | 3.0% |
| HellaSwag, WinoBias, BBQ, AdvBench, HarmBench, GSM8K, WinoGrande | 0 / 200 each |

**What it shows:**
- **Matches in the post-2019 group are generic wording, not leaks.** The longest is 9 words: "How many planets are
  there in our solar system?" (19 C4 documents), "Iraq is a country in the Middle East." StereoSet's 12%
  (24/200) is almost all short, generic context sentences; at 8+ words it is 2/88. TruthfulQA has 16/200, all
  common questions.
- **Above 9 words the negative control is clean**, while web-sourced benchmarks stay at ~3%. Their matches are real
  source sentences ("Sewing is the craft of fastening or attaching objects using stitches made with a needle and
  thread.", a "Mary J. Blige tweeted…" news line).
- **The 10-word cutoff was picked after seeing the data**, so it is a calibration on this sample, not a
  pre-registered threshold. A fresh sample should be used to confirm it.
- **The author-written safety benchmarks people rely on today are not in C4 verbatim:** BBQ, AdvBench, HarmBench,
  GSM8K and WinoGrande all 0/200; TruthfulQA, XSTest and Do-Not-Answer match only on generic short questions. The
  bias and toxicity benchmarks that **copy web text** (BOLD, RealToxicityPrompts) are partly in C4, ~7%.

**Did not work:**
- **ETHICS commonsense:** 95 of 200 texts exceed the API's 1,000-character limit. These are the long Reddit AITA
  posts, exactly the web-sourced part. The 104 that ran are the short MTurk scenarios (0 matches). The field is
  therefore untested for its Reddit part.
  - Next: query a fixed-length prefix (e.g. the first 50 words) of long texts.
- **MMLU:** 15 long questions were also over the limit.
- **HellaSwag `ctx_a`:** 0/200. Its WikiHow / ActivityNet contexts are edited or crowd-written captions, so an
  exact match is not expected. A prefix test would also apply here.

**Limits:** exact match is a lower bound (no case / punctuation normalisation); 200 texts per field, so a single
field's CI is wide (0/200 → at most 1.9%).

## E2. Long texts queried by their first 50 words (`--max_words 50`), same 200 texts (2026-10-24, local)

Output: `results/E2_prefix50/`. Every text is now queryable (0 errors, was 96 for ETHICS and 15 for MMLU).

| Field | Full text queried | First 50 words queried | All |
|---|---|---|---|
| ETHICS commonsense | 0 / 81 | 0 / 119 (the long Reddit AITA posts) | **0 / 200** [0, 1.9] |
| MMLU question | 6 / 152 | 0 / 48 | 6 / 200, 3.0% |
| HellaSwag ctx_a | 0 / 152 | 0 / 48 | 0 / 200 |

**What it shows:**
- **ETHICS is not in C4 verbatim, its Reddit part included.** Two explanations we cannot separate here:
  1. many AITA posts were written after the April 2019 crawl (ETHICS is from 2020);
  2. Reddit threads rarely survive C4's line filters (short lines, no terminal punctuation).

  Querying the posts' own dates (Reddit IDs encode time) would separate them.
- **MMLU's 6 matches are all short and generic** ("K-fold cross-validation is", 9 C4 documents); none of its long
  questions match. So MMLU looks like the negative-control pattern, not like BOLD / RealToxicityPrompts.
- 3 MMLU texts contain a broken character from the source ("15�C"). Left as is; they cannot match.

**Caveat:** a prefix match would mean only the first 50 words are in C4. A miss can also come from C4 dropping one of
the lines the prefix spans.
