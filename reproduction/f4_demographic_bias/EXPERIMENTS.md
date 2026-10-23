# Arm A experiment log (Dodge et al. 2021 §4.3 / A.7, Table 7)

Every run is recorded here, including the ones that did not reproduce the paper and the ones that
exposed bugs. All dates are 2026-10-23. Numbers come from `results/experiments_summary.csv` unless
marked otherwise.

**Paper target (Table 7):** Jewish 67.1%, Arab 37.0%, spread across the 15 groups 30.5 points.
The paper does not report CIs, the UnifiedQA size, the input format or how outputs were mapped to choices.

**Common setup:** UnQover ethnicity templates (`allenai/unqover` @ `3e47969`) → 147,000 contexts × 2 questions = 294,000,
exactly the paper's count. Each question is scored as 2-way multiple choice by the log-likelihood of each answer option
(`scripts/score_unifiedqa.py`). Input `question \n (a) X (b) Y \n context`, lowercased (UnifiedQA's own `tasks.py`
lowercases all inputs). Positivity follows A.7. CIs: bootstrap over the 50 attributes, B=1000.

| ID | What | Where it ran | Data | Outcome | Results |
|---|---|---|---|---|---|
| E1 | Smoke test: scorer, resume, analysis | laptop CPU | t5-small, first 300 contexts | Pipeline works. Greedy output agreed with the log-likelihood choice 128/128. Numbers meaningless (one pair, one order) | not kept |
| E2 | Strided-sample check | laptop CPU | t5-small, every 50th context, then every 47th | **Bug found:** a stride of 50 selects a single attribute (50 attributes cycle with period 50), giving zero-width CIs. Fixed by using a stride coprime with 50 and 14 (47, later 97). Stride 47: Spearman −0.21, same answer for both questions 82% | not kept |
| E3 | Full run, UnifiedQA-small | Kaggle 2×T4 | extended file, 266,000 contexts (contains the paper's 147,000) | **Did not reproduce.** Spearman −0.01; Jewish 47.4, Arab 47.4; spread 8.0. Answer flips with sentiment in only 18% of contexts | `results/E3_full_small_ext/` |
| E4 | Size test: small, base, large, 3B | Kaggle 2×T4 | 1-in-97 sample, 1,516 contexts (every pair, both orders, all 50 attributes) | **Only 3B reproduces.** Spearman: small 0.00, base 0.03, large 0.48, **3B 0.79** (Pearson 0.86). 3B: Jewish 65.1 [59.9–70.1], Arab 31.4 [27.6–35.7] | `results/E4_size_test_sample/` |
| E5 | Answer rule: summed vs per-token-mean log-prob | analysis only | E3, E4 | No change to any conclusion: the largest Spearman change is 0.08 (base, 0.03 → −0.05), and 3B moves 0.79 → 0.81. Rule `sum` is the default | `analysis_mean/` folders |
| E6 | Full run, UnifiedQA-3B | Kaggle 2×T4 | the paper's 147,000 contexts (294,000 questions) | **Reproduces the ranking and both focus groups.** Spearman 0.84 (Pearson 0.87). Jewish 65.7 [62.7–68.8] vs paper 67.1; Arab 34.6 [31.8–37.5] vs paper 37.0. All checks pass (join, full coverage, greedy agreement 99.98–100%). The paper lies outside our CI for 7 of 15 groups; the large gaps are Black 25.2 vs 39.3, Latino 56.0 vs 43.1 and European 68.6 vs 60.5. Head to head, the model puts the negative trait on the Arab person in 84% of 1,400 Jewish–Arab contexts | `results/E6_full_3b/`, figures in `results/figures_arm_a/` |
| A1 | Audit fixes (analysis only, no new model runs) | laptop CPU | E3, E4 re-analysed | See "Audit fixes" below. No reported E4 number changed. E3's diagnostics now use only its 147,000 Table 7 contexts (flip rate 17.8 → 18.1). **New finding:** 3B is over-dispersed. Its noise-corrected SD is 13.3 vs the paper's 9.8 (slope 1.20), so sampling noise does not explain the 46 vs 30.5 spread | `results/*/analysis_*/` |
| — | CPU run of base and large on the 1-in-97 sample | laptop CPU | — | Stopped before finishing and moved to GPU (became E4). No results | — |

## Why small, base and large fail (E3, E4)

1. **The scoring is not the problem.** In E3 the option chosen by log-likelihood matched the model's own greedy
   generation for 99.98% of 4,096 checked questions, so we are measuring what the model answers.
2. **The metric needs answers that depend on the question's sentiment.** Each context is asked twice: a negative
   question ("Who was a hateful extremist?") and its positive counterpart ("Who was a peaceful resident?"). A model that
   picks the same group for both gives each group one credit, which pulls every group to 50%. Small flips its answer in
   18% of contexts, base 26%, large 30%, 3B 50% (`fig_size_trend.png`). Only 3B produces the paper's spread.
3. **The paper never states the model size** (§4.3, A.7 and footnote 19 just say "UnifiedQA"). The size test is our
   evidence that the paper used 3B (11B could not be tested on Kaggle).

## Open discrepancies for 3B (E4 sample; E6 confirms them)

**E6 update:** on the full set, Black (25.2 vs 39.3), Latino (56.0 vs 43.1) and European (68.6 vs 60.5) remain far
outside our CIs, so sampling noise is ruled out. Our values are also more spread out (noise-corrected SD 12.1 vs 9.8).
Remaining candidate explanations: a different UnifiedQA size or version in the paper (11B is untested, because it
does not fit the free tier), or a different decision rule.

### Earlier note (E4 sample)

Outside our 95% CI on the sample: Latino 58.4 vs 43.1, Black 24.5 vs 39.3, European 70.8 vs 60.5, South-American 38.8
vs 44.4. Candidate reasons, untested: sampling noise at 1,516 contexts, an 11B model in the paper, or a different
output-to-choice mapping. E6 on the full set will separate the first from the others.

## Audit fixes (2026-10-23, see AUDIT_2026-10-23.md §1)

- **Join check (K3).** Scores are joined to questions by `id`. New chunks store `subj0|subj1|tid|attr` and the
  analysis compares it with the question file. Old chunks are checked through two fingerprints: each option string
  must always have the same token count, and greedy answers must name one of the row's options. All existing
  E3 and E4 chunks pass, so the old results were joined correctly. A shuffled join and a corrupted key both stop
  the run (tested). `generate_questions.py` now keeps existing question files unless `--force` is given.
- **Coverage check (K4).** The analysis refuses a run that lacks any attribute or ordered pair, or covers them
  unevenly (min/max below 0.5). A simulated "one GPU died" run (even ids only, 25 of 50 attributes) is refused.
  `--allow_partial` analyses it for monitoring only.
- **Diagnostics on the Table 7 set (K5)**, so E3 (extended file) compares like with like against the other runs.
- **Dispersion (K1)** is in `diagnostics.json` and the summary: spread, SD, noise-corrected SD, slope on the
  paper's values. On E4, 3B is about 1.2–1.4× more extreme than the paper. Open: E6 will show whether the full
  set keeps this. 11B, the other candidate explanation, does not fit the free Kaggle / Colab tiers (the
  checkpoint is ~45 GB), so it stays untested.
- **Small head-to-head flagged (K6).** The Jewish-vs-Arab head-to-head on the E4 sample has 15 contexts. It is now
  marked "illustrative only" in the figure and in `jewish_vs_arab.md`.
- **Per-attribute breakdown (K8):** `by_attribute.csv` and `fig_by_attribute.png`, plus `table7_no_negation.csv`.
  4 attributes have a negation-style positive question ("Who wasn't a drug dealer?"). Dropping them moves 3B's
  Spearman from 0.79 to 0.80, so they are not the problem.

## Per-ethnicity counts (all 15 groups, not only Jewish and Arab)

`counts_by_ethnicity.csv/.md` gives, for every group and every model, how often it was the model's answer on positive and on negative questions, its positive and negative associations, and the resulting Table 7 positivity. `fig_counts_by_ethnicity.png` shows the two answer rates per group, and `fig_opponents_all_groups.png` shows each group against every other group. The code checks that these counts reproduce the Table 7 positivity exactly. Example (E4, 3B): Black is the answer on 69% of negative questions and 18% of positive ones (positivity 24.5); European on 36% and 77% (70.8).

## Other checks recorded

- Independent recount: a separate script, written directly from A.7's wording and sharing no code with `analyze_arm_a.py`, reproduces E4-3B's positivity for all 15 groups exactly (max difference 0.0). Every one of the 3,032 answers is one of the two ethnicities in the question.
- Question generation is deterministic and reproduces the paper's 294,000 exactly (15 groups × 14 others × 14 templates × 50 attributes × 2 questions).
- The extended file's 147,000 original contexts are identical to the paper's file (checked by content).
- Upstream UnQover quirk kept for fidelity: its article rule writes "an European".
- Option tokens differ in length (`arab` 2 tokens, `jewish` 4, `middle-eastern` 6), which is why E5 compares rules.

---

# Arm B experiment log (C4 paragraph sentiment, §4.3 / A.7)

**Paper target:** in C4.EN paragraphs containing " Jewish " / " Arab ", SocialSent-derived positivity is 73.2% (3.4M
tokens) vs 65.7% (1.2M), a 7.5-point gap. The VADER gap is 2.0 and the UnQover-lexicon gap is 1.4. NYT: 74.0 vs 69.5;
Al Jazeera: 42.5 vs 42.8.

**Common setup:** paragraphs are the newline-separated lines of C4.EN documents (`allenai/c4`, `en`). A paragraph
matches a group when it contains the word with a space on each side, case-sensitive (A.7's fgrep matching). Tokens
are split on whitespace, stripped of punctuation and lowercased (lowercasing is our assumption). Lexicons come from
`scripts/build_lexicons.py`, with sources pinned in `data/lexicons/MANIFEST.json`. CIs: bootstrap over web domains, B=1000.

| ID | What | Data | Outcome | Results |
|---|---|---|---|---|
| B0 | One-shard test | shard 00000 (356,317 docs; 2,939 Jewish / 1,384 Arab paragraphs) | **Reproduces.** SocialSent 71.8 vs 64.5 (paper 73.2 vs 65.7); gap 7.3 [2.5–11.9] (paper 7.5). VADER gap 1.8 [−1.4–4.8] (paper 2.0); UnQover gap 2.6 [−3.5–8.5] (paper 1.4). NYT / Al Jazeera: too few tokens (1–17) to compare | `results/B0_one_shard_00000/` |
| B1 | 12 seeded shards (seed 0; listed in `shards.json`) | 12/1024 of C4.EN (4.28M docs, 18,893 domains; 32,562 Jewish / 17,175 Arab paragraphs) | **Reproduces; every paper value is inside our 95% CI.** SocialSent: Jewish 73.4 [72.4–74.3] (paper 73.2), Arab 66.2 [64.9–67.5] (65.7), gap 7.1 [5.6–8.7] (7.5). VADER gap 2.0 [0.9–2.9] (2.0). UnQover gap 3.0 [1.2–4.9] (1.4; our lexicon is a reconstruction). NYT 69.3 vs 67.4 (paper 74.0 vs 69.5) and Al Jazeera 46.7 vs 40.5 (42.5 vs 42.8), but only 30–101 tokens each, so these are not interpretable | `results/B1_c4_12_shards/` |

| B2 | **Extension** (not in the paper): all 15 Table 7 groups, matched by spelling variants | shard 00000 (19,581 domains) | Jewish/Arab point estimates identical to B0 (71.8 / 64.5); CIs shift slightly because the domain bootstrap now covers more sites. Across groups, SocialSent corpus positivity **does not track** the paper's Table 7 (Spearman −0.05). "Black" (83.1%) and "White" (81.7%) score as highly positive, consistent with the non-ethnic senses the paper warns about. One shard only | `results/B2_ext_all_groups_shard00000/` |
| BA1 | Audit fixes (analysis only; same paragraphs) | B0, B1, B2 re-analysed | The paper-setting SocialSent and VADER numbers are **unchanged**. UnQover lexicon v2 moves the B1 UnQover gap 3.0 → **5.7 [3.4–8.0]**, further from the paper's 1.4. New robustness table: see "Arm B audit fixes" below | `results/B*/analysis/robustness.csv` |
| B3 | Extension: all 15 groups on B1's 12 shards | 12/1024 | Not run yet (Colab notebook section 5) | `results/B3_ext_all_groups_12_shards/` |

| B4 | Larger sample: shards 0–255 (1/4 of C4.EN), for NYT / Al Jazeera | Not run yet (Colab section 6; free tier, several sessions, output on Drive) | `MyDrive/f4_B4_c4_256_shards` |

## Arm B audit fixes (2026-10-23, AUDIT_2026-10-23.md §2)

**UnQover lexicon v2 (C2).** v1 tokenised the UnQover phrases with `[a-z']+`, splitting hyphenated words. It
therefore counted "well" (from "well-behaved") as positive and "mean" (from "mean-spirited") as negative, and those
became the top tokens for both groups. v2 tokenises like the corpus (whitespace split, punctuation removed), and
drops "doing", "going" and "hard", which carry no sentiment on their own. Result: 33 positive and 25 negative
words (v1: 33 / 27). v1 is kept as `data/lexicons/unqover_v1_hyphen_split.tsv`. **Correction to the audit:** the
fix does not move the UnQover gap toward the paper (1.4). B1 goes from 3.0 to 5.7, so our UnQover lexicon remains a
reconstruction that does not match the paper's unreleased one.

**Robustness of the gap (C1, C3), B1, 12 shards** (each row changes one thing; 95% domain-bootstrap CI):

| Variant | SocialSent | VADER | UnQover v2 |
|---|---|---|---|
| As in paper (paper: 7.5 / 2.0 / 1.4) | 7.1 [5.6, 8.7] | 2.0 [0.9, 2.9] | 5.7 [3.4, 8.0] |
| "Arab" without "United Arab Emirates" paragraphs (−3,870 of 17,175) | 10.2 [8.6, 11.9] | 7.3 [6.2, 8.5] | 7.3 [4.7, 9.8] |
| "Arab" without geopolitical phrases (UAE, Arab League / Spring / world / countries / states / nations; −7,971) | 7.6 [5.8, 9.4] | 6.0 [4.7, 7.2] | 4.8 [2.2, 7.5] |
| Without 1-letter tokens and SocialSent noise words (man, mr, series, due, ...) | 7.5 [5.8, 9.1] | 2.0 [0.9, 2.9] | 5.7 [3.4, 8.0] |
| Without religious words (god, holy, faith, holiday, ...) | **3.4 [1.6, 5.2]** | **0.4 [−0.7, 1.4]** | 5.7 [3.4, 8.0] |

What this means:
- **The paper's VADER match (2.0) is produced by UAE text.** "United" is VADER-positive, and UAE paragraphs are mostly
  upbeat commercial text. Without them the VADER gap is 7.3. The paper's method would hit the same artefact.
- **About half of the SocialSent gap and nearly all of the VADER gap come from religious vocabulary**, which is
  more common in "Jewish" paragraphs. This is a word-mix effect, not necessarily a more favourable tone towards
  the group. It is the strongest caveat on reading A.7's gap as an ethnic bias.
- SocialSent noise words do not matter (7.1 → 7.5).
- B0 and B2 (one shard) show the same pattern with wider CIs.

**Scanner (C6 and free tier).** A finished shard is reused only if it was scanned with the same targets, and a
folder's `shards.json` now accumulates shards across sessions. Older list-style files are read correctly.
`--workers N` downloads and scans shards in parallel, and `--range A B` scans a block of shards, so B4 can run on
free Colab over several sessions into one Google Drive folder. The domain bootstrap now draws replicates in
batches, so a 256- or 1,024-shard analysis fits in free-tier RAM. The paper-setting CIs are unchanged.

**Why Arm B's reproduction covers only Jewish and Arab:** A.7 says the paper "selected 'Jewish' and 'Arab' as points of comparison for a corpus study ... as they are the ethnicities with the most extreme biases that were easy to find in C4.EN with simple scripts", noting that "African" is a substring of "African American" and that "Black" has common non-ethnic senses. B2 and B3 extend this to all groups and inherit those caveats.

**Scale check (B1, 12 shards):** SocialSent tokens 40,710 Jewish / 14,100 Arab vs the paper's 3.4M / 1.2M ÷ 1024 × 12 = 39,844 / 14,063. Words in matching paragraphs: 2.88M / 1.44M vs 2.93M / 1.58M. The matching and counting therefore track the paper's closely.

**Scale check (B0):** words and SocialSent tokens in matching paragraphs, against the paper's full-corpus totals ÷ 1024:
Jewish 262k words (paper ≈ 244k) and 3,751 tokens (≈ 3,320); Arab 106k words (≈ 132k) and 1,019 tokens (≈ 1,172).

**Deviations (ours, because the paper is silent or the resource is gone):**
- The 250 SocialSent subreddit lexicons come from the copy bundled in shifterator (`8d74d74`); the Stanford page is unreachable.
- The "overtly ethnic terms" removed from SocialSent are our list (31 found among the candidates; listed in MANIFEST.json).
  Non-ethnic identity words were left in, as the paper describes only an ethnic filter. Example: "gays" is in the top-250 negative words.
- The UnQover lexicon was not released. Ours takes content words from UnQover's stereotype file and keeps those whose
  sign agrees with VADER or the SocialSent aggregate (33 positive, 27 negative).
- The NYT / Al Jazeera comparison needs the full corpus. The paper itself had only 1.6k–7.9k tokens per domain, which
  is about 2–8 per shard.
