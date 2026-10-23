# Reproducing *Documenting Large Webtext Corpora* (Dodge et al., EMNLP 2021)

A reproduction and extension of the C4 audit: **§4.3** (demographic bias) and **§5** (what the blocklist excluded).

| Part | Paper claim | Status |
|---|---|---|
| A: model probe | UnifiedQA favours some ethnicities: Jewish 67.1% vs Arab 37.0% positive (Table 7) | Reproduced with UnifiedQA-3B on a 1,516-context sample (Spearman 0.79); smaller models do not reproduce. The full run is pending |
| B: corpus | C4 paragraphs about "Jewish" are more positive than those about "Arab": 73.2% vs 65.7%, gap 7.5 (SocialSent) | Reproduced on 12 of 1,024 C4 shards: 73.4% vs 66.2%, gap 7.1 [5.6–8.7]; VADER gap 2.0 vs 2.0. The gap is sensitive to "United Arab Emirates" text and religious words (EXPERIMENTS.md) |
| §5: excluded data | The blocklist disproportionately removes text about LGBTQ+ identities (Fig. 5) and in AAE / Hispanic-aligned English (42% / 32% removed vs 6.2% WAE); 31% of excluded docs are sexual (Fig. 6) | One shard reproduces all three: Fig. 5 Spearman 0.97; AAE 36% / Hispanic 35% / WAE 7.2% removed; 12–14 of 50 sexual clusters (28–33%). 12-shard run pending |

Every run, including the ones that failed, is logged in
[reproduction/f4_demographic_bias/EXPERIMENTS.md](reproduction/f4_demographic_bias/EXPERIMENTS.md) and
[reproduction/f5_excluded_data/RESULTS.md](reproduction/f5_excluded_data/RESULTS.md). Audit of the notebooks:
[AUDIT_2026-10-23.md](reproduction/f4_demographic_bias/AUDIT_2026-10-23.md).

Run outputs (`results/` folders, plots) are not in git yet; see `.gitignore`.

## Repository map

| Path | What |
|---|---|
| `reproduction/PROPOSAL_DRAFT.md` | Research questions, expectations, and what would change our mind |
| `reproduction/f4_demographic_bias/` | §4.3 reproduction: code, data, Kaggle (Arm A) and Colab (Arm B) notebooks. Start with its README |
| `reproduction/f5_excluded_data/` | §5 reproduction: code, Kaggle and Colab notebooks. Start with its README |
| `reproduction/f4_demographic_bias/ARM_A_REPORT.md` | Write-up of Arm A |
| `reproduction/paper_figures/` | Replication of the paper's Figure 5 (§5.2) |
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
