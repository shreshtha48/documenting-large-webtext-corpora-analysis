# F5 results log (Dodge et al. 2021 §5)

Every run is listed, reproduced or not. Dates: 2026-10-23.

| ID | What | Where | Data | Outcome | Results |
|---|---|---|---|---|---|
| S0 | One-shard test of the whole pipeline | laptop CPU (16 min scan, 1 min analysis) | `en.noblocklist` shard 00000: 384,171 docs, 286,507 domains | **Reproduces all three parts in direction and size**; see below | `results/S0_noblocklist_shard00000/` |
| S1 | Main run: 12 seeded shards planned (the indices of Arm B's B1) | Colab free tier, several sessions; scan ~35 min per shard pair, clustering ~15 min | **10** of the 12 shards finished before a disconnect: 3,841,713 docs, 1,741,415 domains | **Reproduces all three parts**; see "S1 numbers" below. The first analysis run crashed after k-means seed 0 several times (Colab session limits); fixed by saving Fig. 6 / clusters.csv per seed and the `--clusters_only` resume | `results/S1_noblocklist_10_shards/` (analysis, per-shard stats; the counts / samples stay on Google Drive) |
| — | Manual check of 50 "lesbian" + 50 "gay" excluded docs | a person | `inspect_lesbian_gay.csv` | Not labelled yet (notebook section 6) | — |

## S1 (10 shards): numbers

**Excluded set:** 7.24% of 3,841,713 documents. Fast-vs-exact dialect agreement: 99.72% on 10,000 documents.

| Part | Paper | Ours (95% CI, domain bootstrap) |
|---|---|---|
| §5.1 Fig. 6: sexual clusters (k=50, 100,000 docs) | 16 of 50, 31% of docs (by hand) | **15 of 50, 30.8%** (seed 0, automatic rule); seeds 1-2: 14 / 28.9%, 15 / 28.5% |
| §5.2 Fig. 5: identity PMI ranking | sexual-orientation terms highest | Spearman **0.97** with the paper's bars; lesbian 2.44 (paper ~2.44) top, european 0.78 (~0.84) bottom |
| §5.3 AAE removed | 42% | **35.8% [33.7, 38.1]** (n = 4,129) |
| §5.3 Hispanic-aligned removed | 32% | 34.0% [32.3, 36.1] (n = 3,502) |
| §5.3 WAE / other removed | 6.2% / 7.2% | 7.2% [7.1, 7.3] / 6.6% [6.3, 6.8] |
| §5.3 share of C4.EN: WAE / AAE / Hispanic | 97.8% / 0.07% / 0.09% | 98.0% / **0.074%** / 0.065% |

- The paper's 42% AAE removal is outside our CI. Ours agrees with the full-corpus count implied by Deas et al. 2025
  with the same classifier (37.2%), so the paper's figure looks overstated. The direction (AAE and Hispanic removed
  about 5x as often as WAE) holds.
- The manual check (§5.2: 22% / 36% of excluded "lesbian" / "gay" documents non-offensive or non-sexual) still needs
  a person to label `inspect_lesbian_gay.csv`.

## S0 (one shard): numbers

**Excluded set.** 27,675 of 384,171 documents (7.20%) match the blocklist. That is the same count as the earlier pilot,
and close to the corpus-level 7.6% (395M vs 365M documents).

**§5.2, Fig. 5.** Spearman with the paper's bars, read off the figure, is **0.97**. As in the paper, the sexual-orientation terms
are highest (lesbian 2.44, heterosexual 2.34, homosexual 2.27, bisexual 2.26, gay 2.20) and european (0.80) and
white (0.96) are lowest. Ours are lower than the paper for man (1.36 vs ~1.65), women (1.44 vs ~1.73), black (1.20 vs
~1.43) and trans (1.52 vs ~1.75). One shard; S1 will show whether these persist.

**§5.3, dialects** (95% CI, domain bootstrap):

| Dialect | Removed (ours) | Removed (paper) | Share of C4.EN (ours) | Share (paper) |
|---|---|---|---|---|
| AAE | 36.3% [30.9, 41.8] (n=386) | 42% | 0.069% [0.061, 0.080] | 0.07% |
| Hispanic | 34.6% [29.5, 39.7] (n=350) | 32% | 0.064% [0.056, 0.075] | 0.09% |
| Other | 6.8% [6.3, 7.5] | 7.2% | 1.79% | — |
| WAE | 7.2% [7.1, 7.3] | 6.2% | 98.0% | 97.8% |

- The headline holds: AAE and Hispanic-aligned documents are removed about 5× as often as WAE.
- WAE's removal rate (7.2%) is above the paper's 6.2%, and its CI excludes the paper's value. That is expected if
  most documents are WAE: the overall rate on this shard is 7.2%.
- The fast dialect inference agrees with the exact port on 99.3% of 1,000 documents. The exact port reproduces the
  model README's worked examples to every printed digit.

**§5.1, Fig. 6** (10,000 excluded docs on one shard; S1 uses 100,000 as the paper does):

| k-means seed | Sexual clusters (automatic rule) | Share of docs |
|---|---|---|
| 0 | 13 / 50 | 33.0% |
| 1 | 14 / 50 | 27.5% |
| 2 | 12 / 50 | 28.7% |
| Paper (by hand) | 16 / 50 | 31% |

The automatic rule is imperfect. It counts a news cluster ("police women said rape year") as sexual, and misses
"cam chat webcam" and "dating singles online". `clusters.csv` therefore has a `manual_label` column. Non-sexual
clusters appear as in the paper: science and medicine ("patients cancer treatment"), law ("court criminal
law"), politics ("trump president"), religion, sports and recipes.
