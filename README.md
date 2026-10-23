# Reproducing *Documenting Large Webtext Corpora* (Dodge et al., EMNLP 2021)

A reproduction of the C4 audit's bias, contamination and excluded-data findings: **§4.2** (benchmark contamination),
**§4.3** (demographic bias) and **§5** (what the blocklist excluded). One-page summary:
[reproduction/REPRODUCTION_BRIEF.md](reproduction/REPRODUCTION_BRIEF.md).

| Part | Paper claim | Our result |
|---|---|---|
| §4.2 contamination | Benchmark texts appear verbatim in C4.EN: 1.87–24.88% of generation targets, 1.8–53.6% of GLUE inputs (Table 2) | 6 headline fields, all of C4.EN (infini-gram, exact match = lower bound): identical ranking; the paper's value is inside our CI for 4 of 6 |
| §4.3 A: model probe | UnifiedQA favours some ethnicities: Jewish 67.1% vs Arab 37.0% positive (Table 7) | UnifiedQA-3B on all 294,000 questions: Spearman 0.84; Jewish 65.7, Arab 34.6 (both inside CI); Black, Latino and European off by 8–14 points |
| §4.3 B: corpus | C4 paragraphs about "Jewish" are more positive than those about "Arab": gap 7.5 (SocialSent) | 12 of 1,024 shards: gap 7.1 [5.6–8.7]; VADER gap 2.0 vs 2.0. The gap is sensitive to "United Arab Emirates" text and religious words |
| §5 excluded data | The blocklist removes LGBTQ+ identity mentions most (Fig. 5); AAE / Hispanic-aligned English 42% / 32% removed vs 6.2% WAE; 16 of 50 excluded-doc clusters sexual, 31% (Fig. 6) | 10 shards: Fig. 5 Spearman 0.97; AAE 35.8% / Hispanic 34.0% / WAE 7.2% removed; 15 of 50 clusters sexual, 30.8% |

Every run, including the ones that failed, is logged in each folder's
[EXPERIMENTS.md](reproduction/f4_demographic_bias/EXPERIMENTS.md) /
[RESULTS.md](reproduction/f5_excluded_data/RESULTS.md) /
[RESULTS.md](reproduction/f3_contamination/RESULTS.md). Audit of the original notebooks:
[AUDIT_2026-10-23.md](reproduction/f4_demographic_bias/AUDIT_2026-10-23.md).

F4's run outputs are in git; F3's and F5's `results/` folders are git-ignored (see `.gitignore`).

## Repository map

| Path | What |
|---|---|
| `reproduction/PROPOSAL_DRAFT.md` | Research questions, expectations, and what would change our mind |
| `reproduction/f4_demographic_bias/` | §4.3 reproduction: code, data, Kaggle (Arm A) and Colab (Arm B) notebooks. Start with its README |
| `reproduction/f5_excluded_data/` | §5 reproduction: code, Kaggle and Colab notebooks. Start with its README |
| `reproduction/f3_contamination/` | §4.2 reproduction (benchmark contamination): code, Kaggle notebook |
| `reproduction/REPRODUCTION_BRIEF.md` | **One-page summary**: what we reproduced, the paper's numbers vs ours, what it means |
| `reproduction/f4_demographic_bias/ARM_A_REPORT.md` | Write-up of Arm A |
| `papers/implementation-ideas/` | Follow-up work since 2021 and extension ideas |
| `papers/citing-papers/citing_papers.csv` | 829 papers citing Dodge et al. (Semantic Scholar + OpenAlex) |
| `research-empirical-methods/` | Method notes per analysis, with reference code (TFDS C4, WIMBD, TwitterAAE) |
| `pilots/` | Early one-shard pilots (blocklist check, identity PMI, filters on machine-generated text) |

## Quick start

```bash
pip install -r requirements.txt
cd reproduction/f4_demographic_bias
python scripts/generate_questions.py                                   # Arm A questions (seconds)
python scripts/build_lexicons.py                                       # Arm B lexicons
python scripts/scan_c4_paragraphs.py --n 12 --out_dir results/B1_c4_12_shards
python scripts/count_sentiment.py --paragraphs results/B1_c4_12_shards --out_dir results/B1_c4_12_shards/analysis
```
Arm A scoring needs a GPU: use `reproduction/f4_demographic_bias/kaggle/f4_arm_a_end_to_end.ipynb` on Kaggle (2×T4).
Arm B runs on CPU: use `reproduction/f4_demographic_bias/colab/f4_arm_b_colab.ipynb` on Google Colab, or the commands above.
§5 runs on CPU: `reproduction/f5_excluded_data/kaggle/f5_excluded_kaggle.ipynb` (Kaggle) or `colab/f5_excluded_colab.ipynb`.

## Third-party material and licences

| Material | Source | Licence |
|---|---|---|
| `reproduction/f4_demographic_bias/third_party/unqover/` | allenai/unqover @ 3e47969 | Apache-2.0 (LICENSE included) |
| `research-empirical-methods/reference-code/tfds_c4/*.py` | tensorflow/datasets | Apache-2.0 (headers in files) |
| `research-empirical-methods/reference-code/tfds_c4/ldnoobw_en_25e679f.txt` | LDNOOBW word list | CC BY 4.0 |
| `research-empirical-methods/reference-code/wimbd/` | allenai/wimbd @ 18fbc62 | Apache-2.0 (LICENSE included) |
| `research-empirical-methods/reference-code/twitteraae/` | slanglab/twitteraae | MIT (LICENSE included) |
| `data/lexicons/socialsent_raw.json.gz` | SocialSent subreddit lexicons (Hamilton et al. 2016), via shifterator @ 8d74d74 (Apache-2.0) | cite Hamilton et al. 2016 |
| `data/lexicons/vader.tsv` (derived) | cjhutto/vaderSentiment @ 44fc044 | MIT |
| `results/B*/paragraphs_*.jsonl.gz` | Paragraphs extracted from C4 (allenai/c4) | ODC-BY; also subject to Common Crawl terms of use |
| `reproduction/f5_excluded_data/data/ldnoobw_en_25e679f.txt` | LDNOOBW word list (copy of the reference-code file) | CC BY 4.0 |
| TwitterAAE model files (downloaded at run time, not in git) | slanglab/twitteraae @ db4d6d1 | MIT |
