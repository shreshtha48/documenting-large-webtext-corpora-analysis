# Group sentiment in C4: 2019 vs post-LLM 2024

Arm B of the F4 reproduction (Dodge et al. 2021, §4.3 / A.7), run on two corpora built with the same C4 filters:

- **2019**: C4.EN from `allenai/c4`, the 12 seeded shards of Arm B's B1 run.
- **2024**: C4.EN that we built by running TFDS v3.0.1's C4 filters ([../machine_text/scripts/filter_c4_pipeline.py](../machine_text/scripts/filter_c4_pipeline.py))
  on Common Crawl WET shards from CC-MAIN-2024-10 (Feb/Mar) and CC-MAIN-2024-51 (Dec). Default: 24 shards per crawl.

Every run is logged in [EXPERIMENTS.md](EXPERIMENTS.md).

## What is measured

For all 15 Table 7 groups, using Arm B's exact matching and counting:

| Output | Meaning |
|---|---|
| positivity | Positive share of sentiment in paragraphs mentioning the group. Three lexicons plus a RoBERTa classifier |
| relative | Group minus **Baseline**, where Baseline is a 1-in-200 random sample of all paragraphs in the same corpus. This is the group's tone against the corpus's general tone |
| change | 2024 minus 2019, for positivity and for relative. The second is a difference-in-differences, so a general shift in web tone does not count as a change towards a group |
| gap | Jewish − Arab (the paper's comparison), with all Arab paragraphs and without geopolitical phrases ("United Arab Emirates", …) |
| dispersion | SD of `relative` across the groups that have ≥ 100 paragraphs in every corpus, raw and noise-corrected, plus the Spearman correlation of the group ranking with 2019 |

95% CIs come from a web-domain bootstrap (B = 1000), with each corpus resampled independently.

## Files

| Path | What |
|---|---|
| `kaggle/post_llm_bias.ipynb` | **The notebook to run** (Kaggle, GPU T4 ×2, internet on). Built by `kaggle/build_notebook.py`; rebuild after editing any embedded script |
| `scripts/scan_groups.py` | Group paragraphs plus the random baseline, for 2019 HF shards or our 2024 C4.EN files |
| `scripts/classify_sentiment.py` | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) probabilities per paragraph, on GPU |
| `scripts/compare_years.py` | All the tables, figures and `summary.md` |
| `../machine_text/scripts/run_wet_shards.py` | Builds 2024 C4 from many WET shards in parallel; resumable |

These are reused unchanged from `reproduction/demographic_bias/`: `scan_c4_paragraphs.py` (group strings),
`count_sentiment.py` (lexicon counting, domain bootstrap) and the three lexicons.

## Checks done before the first Kaggle run (2026-10-24, local)

- `scan_groups.py` on 2019 shard 00000 gives exactly Arm B's B2 paragraph counts (Jewish 2939, Arab 1384, …).
- `compare_years.py` on that shard gives a SocialSent Jewish − Arab gap of 7.32, matching Arm B's B0 (7.3).
- On the 4 pilot shards (2 per crawl), 2024 has about 100 Jewish and 80 Arab paragraphs per crawl, so the gap CI is
  about ±30 points. This is why the notebook uses 24 shards per crawl. At that size the CI should come out near
  ±5 points for the pooled 2024 corpus.

## Limits (state these in the write-up)

- **Small 2024 corpus:** one 2024 WET shard gives about 9.5k C4.EN docs, 1/37 of a 2019 C4 shard. URL and line dedupe run per WET shard, so 2024 keeps more boilerplate than real C4.
- **No causal claim:** a 2019 → 2024 difference cannot be attributed to LLM text. World events, especially from October 2023 onwards, change how groups are written about. The LLM question is handled within 2024 in [../machine_text](../machine_text).
- **Ambiguous group words:** "Black", "White", "African" and "European" also match non-ethnic uses.
