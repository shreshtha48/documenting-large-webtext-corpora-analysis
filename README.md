# Reproducing *Documenting Large Webtext Corpora* (Dodge et al., EMNLP 2021)

Dodge et al. audited **C4**, the web corpus T5 was trained on. We reproduced three of its analyses and then asked
whether the findings still hold for the post-LLM web (2024):

- **Benchmark contamination** (§4.2): are test sets inside the training data?
- **Demographic bias** (§4.3): is the corpus, and a model trained on it, biased by ethnicity?
- **The blocklist filter** (§5): whose text does the "bad words" filter throw away?

Every number has a 95% CI (the paper reports none). Every run is logged, including smoke tests and failed runs, in each
folder's `EXPERIMENTS.md` / `RESULTS.md`. One-page summary of the reproduction:
[reproduction/REPRODUCTION_BRIEF.md](reproduction/REPRODUCTION_BRIEF.md).

## The questions and how many we answered

Our proposal ([reproduction/PROPOSAL_DRAFT.md](reproduction/PROPOSAL_DRAFT.md)) set out three claims to test, plus
three follow-up questions. We added the contamination claim later.

| # | Question | Status |
|---|---|---|
| 1 | Does ethnic bias in C4 carry into a model trained on it? (§4.3) | **Answered: yes** |
| 2 | Does the blocklist exclude some demographics unfairly? (§5.2–5.3) | **Answered: yes** |
| 3 | Whose English is in C4? (97.8% White-aligned English, 51.3% US-hosted) | **Partly**: dialect share reproduced, US hosting not checked |
| 4 | Is C4 contaminated with benchmark test sets? (§4.2) | **Answered: yes** (6 of 21 fields) |
| 5 | Does a sentiment classifier, rather than word lists, change the picture? | **Answered**: same picture, and it separates groups most clearly |
| 6 | Does the bias differ by type of document (news, forums, domains)? | **Partly**: only the geopolitical / non-geopolitical split |
| 7 | Do LLM-generated documents amplify the bias? | **Partly**: machine-like text measured, too few group mentions to compare bias |

That makes 4 claims and 1 follow-up answered in full, and 3 answered in part.

## Key findings: reproduction

**Overall, the paper's claims hold in direction, but several of its exact numbers depend on choices the paper does not
report** (model size, word list, word sense).

- **Contamination (§4.2).** Benchmark texts do occur verbatim in C4. We checked six fields that span the paper's range
  against all of C4.EN. They come out in the same order as in the paper, and 4 of the 6 paper values fall inside our
  CI (e.g. TIFU-long 2.3% vs 1.9%; QNLI sentences, 41% vs 54%, is one of the two outside).
- **Bias in a model (§4.3).** We ran UnifiedQA-3B on all 294,000 questions. Its ranking of ethnicities correlates
  0.84 with the paper's Table 7. Jewish (65.7%) and Arab (34.6%) match the paper; Black, Latino and European are off
  by 8–14 points. Smaller UnifiedQA models do not reproduce the result.
- **Bias in the corpus (§4.3).** In 12 C4 shards, "Jewish" paragraphs are more positive than "Arab" ones by
  7.1 [5.6, 8.7] points (paper: 7.5). The gap is fragile:
  - dropping "United Arab Emirates" text moves the VADER gap from 2.0 to 7.3;
  - dropping religious words halves the SocialSent gap.
- **Blocklist (§5).** We ran 10 shards (3.8 M documents):
  - LGBTQ+ terms are the most excluded (rank correlation 0.97 with Fig. 5; 83% of "lesbian" documents removed).
  - African American English loses 35.8% of its documents (paper: 42%), Hispanic-aligned English 34.0% (paper: 32%)
    and White-aligned English 7.2% (paper: 6.2%).
  - 15 of 50 clusters of excluded documents are sexual content (paper: 16).

## Extensions: what we did beyond the paper

| Extension | Question | Key finding |
|---|---|---|
| **Post-LLM C4** ([extension/machine_text](extension/machine_text), [post-llm-crawl](extension/post-llm-crawl)) | Can we rebuild C4 from a 2024 crawl? | Yes: our filters give output identical to the TFDS v3.0.1 reference. We built 48 shards of Common Crawl from Feb and Dec 2024; the blocklist removes ~5% of documents vs 7.6% in 2019 |
| **Post-LLM bias** ([extension/post_llm_bias](extension/post_llm_bias)) | Has the web's tone towards ethnic groups changed since 2019? | The Jewish–Arab gap shrinks from 7.1 to 0.3 points (SocialSent, 2019 vs 2024). In Feb 2024, Arab paragraphs gained 14 points beyond the web-wide shift (RoBERTa). Groups mostly keep their 2019 order |
| **Machine-generated text** ([extension/machine_text](extension/machine_text)) | How much of the 2024 web looks machine-written? | Documents with LLM "marker words" (*delve, tapestry…*) rose from 4.6% → 7.6% → 9.7% (2019 → Feb → Dec 2024), while ordinary words stayed flat. That implies at least 5.4% of Dec 2024 documents were LLM-processed. The Binoculars detector shows a smaller rise (1.0% → 1.3%) |
| **Gender & age** ([extension/gender_age](extension/gender_age)) | Who does the 2019-C4 model (UnifiedQA-3B) cast in a positive light? | Women are picked for the positive answer 62–70% of the time across the four probes, and more often than men in every age group and every ethnicity (65–76%). Teens are cast most negatively (35%) and middle-aged people most positively (66%) |
| **Modern benchmark contamination** ([extension/eval_contamination](extension/eval_contamination)) | Are today's safety / bias benchmarks in C4? | Benchmarks written by their authors (BBQ, AdvBench, HarmBench, GSM8K, WinoGrande) are absent: 0/200 each; so is ETHICS commonsense (0/200, including its Reddit posts). Benchmarks that copy web text (BOLD, RealToxicityPrompts) are ~7% inside C4 |

Figures and tables for the extensions: [extension/results/](extension/results), produced by
[extension/extension_results.ipynb](extension/extension_results.ipynb) from the result zips.

## Repository map

| Path | What |
|---|---|
| [reproduction/REPRODUCTION_BRIEF.md](reproduction/REPRODUCTION_BRIEF.md) | One-page summary: the paper's numbers vs ours |
| [reproduction/PROPOSAL_DRAFT.md](reproduction/PROPOSAL_DRAFT.md) | Research questions, expectations, what would change our mind |
| [reproduction/benchmark-contamination/](reproduction/benchmark-contamination) | §4.2: scripts, benchmark data, Kaggle notebook, results (incl. the C0 smoke test) |
| [reproduction/demographic_bias/](reproduction/demographic_bias) | §4.3: scripts, lexicons, questions, Kaggle (model) and Colab (corpus) notebooks, results |
| [reproduction/excluded_data/](reproduction/excluded_data) | §5: scripts, Kaggle and Colab notebooks, results (S0 one-shard test, S1 10 shards) |
| [reproduction/results/](reproduction/results) | Summary figures for the brief |
| [extension/](extension) | The five extensions above. Each has a README and an experiment log |
| [paper-ideas-extension.md](paper-ideas-extension.md), [papers/implementation-ideas/](papers/implementation-ideas) | Extension ideas and how we scoped them |
| [papers/citing-papers/citing_papers.csv](papers/citing-papers/citing_papers.csv) | 829 papers citing Dodge et al. |
| [research-empirical-methods/](research-empirical-methods) | Method notes, with reference code (TFDS C4, WIMBD, TwitterAAE) |
| [pilots/](pilots) | Early one-shard pilots |

**Notebooks.** Each notebook holds its scripts as `%%writefile` cells and runs end to end on Kaggle or Colab. We
generated them with small builder scripts that are not in the repo, so the README of each folder may still mention a
`build_*notebook*.py`.

**Large files.** Three files over 100 MB are stored with Git LFS: the 2024 C4 shards
(`extension/post-llm-crawl/post_llm_c4_2024.zip`), the post-LLM bias outputs
(`extension/post_llm_bias/post_llm_bias_results.zip`) and the TIFU source data. Run `git lfs install` before cloning.
The TwitterAAE model files are downloaded at run time.

## Quick start

```bash
pip install -r requirements.txt
cd reproduction/demographic_bias
python scripts/generate_questions.py                                   # model-probe questions (seconds)
python scripts/build_lexicons.py                                       # corpus-sentiment lexicons
python scripts/scan_c4_paragraphs.py --n 12 --out_dir results/B1_c4_12_shards
python scripts/count_sentiment.py --paragraphs results/B1_c4_12_shards --out_dir results/B1_c4_12_shards/analysis
```

- **GPU runs:** model scoring uses `reproduction/demographic_bias/kaggle/f4_arm_a_end_to_end.ipynb`; the extensions
  use the `kaggle/*.ipynb` notebooks in each extension folder (Kaggle, T4 ×2).
- **CPU runs:** `reproduction/demographic_bias/colab/f4_arm_b_colab.ipynb` and
  `reproduction/excluded_data/{kaggle,colab}/f5_excluded_*.ipynb`.
- **Extension figures:** open `extension/extension_results.ipynb` from the `extension/` folder and run all.

## Third-party material and licences

| Material | Source | Licence |
|---|---|---|
| `reproduction/demographic_bias/third_party/unqover/`, `extension/gender_age/third_party/unqover/` | allenai/unqover @ 3e47969 | Apache-2.0 (LICENSE included) |
| `research-empirical-methods/reference-code/tfds_c4/*.py` | tensorflow/datasets | Apache-2.0 (headers in files) |
| `ldnoobw_en_25e679f.txt` (reference code and `reproduction/excluded_data/data/`) | LDNOOBW word list | CC BY 4.0 |
| `research-empirical-methods/reference-code/wimbd/` | allenai/wimbd @ 18fbc62 | Apache-2.0 (LICENSE included) |
| `research-empirical-methods/reference-code/twitteraae/` | slanglab/twitteraae | MIT (LICENSE included) |
| `reproduction/demographic_bias/data/lexicons/socialsent_raw.json.gz` | SocialSent (Hamilton et al. 2016), via shifterator @ 8d74d74 (Apache-2.0) | cite Hamilton et al. 2016 |
| `reproduction/demographic_bias/data/lexicons/vader.tsv` (derived) | cjhutto/vaderSentiment @ 44fc044 | MIT |
| Benchmark files in `reproduction/benchmark-contamination/data/` and `extension/eval_contamination/data/` | Each benchmark's public release (see the folder READMEs) | Each benchmark's own licence |
| C4 paragraphs and shards (`results/`, `extension/post-llm-crawl/`, result zips) | allenai/c4 and Common Crawl | ODC-BY; Common Crawl terms of use |
| TwitterAAE model files (downloaded at run time, not in git) | slanglab/twitteraae @ db4d6d1 | MIT |
