# Reproduction brief: *Documenting Large Webtext Corpora* (Dodge et al., EMNLP 2021)

We reproduced three of the paper's analyses of C4.EN: §4.2 (benchmark contamination), §4.3 (demographic bias) and
§5 (what the blocklist excluded). Every number below has a 95% CI; the paper reports none. Full logs, including
runs that failed: `f3_contamination/RESULTS.md`, `f4_demographic_bias/EXPERIMENTS.md`, `f5_excluded_data/RESULTS.md`.

## §4.2 Benchmark contamination (Table 2)

**Paper:** test texts of popular benchmarks occur verbatim in C4.EN: 1.87–24.88% of generation targets and 1.8–53.6% of
GLUE inputs, after normalizing case and punctuation.
**Ours:** 300 texts from each of the six fields that span that range, checked against all of C4.EN (infini-gram).
Exact matching is a lower bound: it found 8 of the paper's 10 Table 5 examples.

| Field | Paper | Ours |
|---|---|---|
| QNLI sentence | 53.6% | 41.3% [35.9, 47.0] |
| TIFU-short | 24.9% | 20.7% [16.5, 25.6] |
| XSum | 15.5% | 6.3% [4.1, 9.7] |
| LAMA T-REx | 4.7% | 3.3% [1.8, 6.0] |
| TIFU-long | 1.9% | 2.3% [1.1, 4.7] |
| QNLI question | 1.8% | 1.0% [0.3, 2.9] |

## §4.3 Demographic bias (Table 7 and A.7)

**Model (Table 7).**
- *Paper:* UnifiedQA links some ethnicities with positive answers far more often than others: Jewish 67.1% vs Arab
  37.0%.
- *Ours:* the paper does not name the model size. UnifiedQA-3B on all 294,000 questions gives a rank correlation of
  **0.84** with Table 7.
  - Jewish 65.7% [62.7, 68.8] and Arab 34.6% [31.8, 37.5] match.
  - Black (25.2 vs 39.3), Latino (56.0 vs 43.1) and European (68.6 vs 60.5) do not.
  - Smaller UnifiedQA models do not reproduce: they rarely change their answer when the question flips from negative to positive.

**Corpus (A.7).**
- *Paper:* C4 paragraphs mentioning "Jewish" carry more positive sentiment words than those mentioning "Arab": 73.2% vs
  65.7%, a 7.5-point gap (SocialSent); 2.0 points with VADER.
- *Ours,* 12 of 1,024 shards:
  - SocialSent 73.4% vs 66.2%, gap **7.1 [5.6, 8.7]**; VADER gap **2.0 [0.9, 2.9]**.
  - The NYT / Al Jazeera comparison needs the full corpus (too few words in a sample).
  - Our gap with the UnQover word list is 5.7 vs the paper's 1.4. That list is a reconstruction, because the paper's was never released.

## §5 What the blocklist excluded (Figs. 5 and 6, §5.3)

**Paper:** documents removed by the bad-words list over-represent LGBTQ+ identity terms (Fig. 5), and African American
and Hispanic-aligned English (42% and 32% removed vs 6.2% of White-aligned). About a third of the removed text is sexual
(Fig. 6).
**Ours:** 10 shards of C4.EN.NOBLOCKLIST, 3.8 million documents:

| Part | Paper | Ours |
|---|---|---|
| Fig. 5: identity terms most likely to be excluded | sexual-orientation terms highest | same order, rank correlation **0.97** |
| AAE / Hispanic / WAE removed | 42% / 32% / 6.2% | **35.8%** [33.7, 38.1] / 34.0% [32.3, 36.1] / 7.2% [7.1, 7.3] |
| AAE share of C4.EN | 0.07% | 0.074% [0.071, 0.078] |
| Fig. 6: sexual clusters among excluded docs | 16 of 50 (31%) | **15 of 50 (30.8%)** |

## What it means

**The paper's qualitative claims hold.** C4.EN contains benchmark test data, from about 1% to about half of a test
set. A QA model trained on it shows the paper's ethnic ranking. The corpus has a matching Jewish-Arab sentiment
gap. The blocklist removes LGBTQ+ and minority-dialect text disproportionately.

**Several exact numbers do not hold.**
- AAE removal is about 36%, not 42%; a later full-corpus study (Deas et al. 2025) also gets 37%.
- Three Table 7 groups differ by 8–14 points.
- The corpus sentiment gap depends on choices the paper did not examine. Without "United Arab Emirates"
  paragraphs (22% of "Arab" mentions) the VADER gap rises from 2.0 to 7.3. Without religious words the SocialSent gap
  halves (7.1 to 3.4).

The paper's evidence is therefore directionally sound, but its point estimates are sensitive to unreported choices:
model size, lexicon, word sense and normalization.

**Not reproduced:**
- §3 metadata (domains, dates, geolocation) and §4.1 patents;
- the other 15 contamination fields;
- the NYT / Al Jazeera split;
- the hand-labelled check of excluded "lesbian" / "gay" documents (a sheet is ready to label).
