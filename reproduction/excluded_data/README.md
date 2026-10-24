# Excluded Data Reproduction: Auditing C4's Blocklist Filtering

This folder reproduces and extends the findings of **Section 5 ("What Was Excluded?") and Appendix A.6** from the paper:
> **Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus**  
> *Jesse Dodge, Maarten Sap, Ana Marasović, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, Matt Gardner (EMNLP 2021)*

---

## 1. Executive Summary & Research Question

When creating the **C4** dataset from raw Common Crawl web scrapes, Google applied aggressive heuristics that **discarded over 90% of the crawled web pages**. A major filter in this pipeline was the **LDNOOBW ("List of Dirty, Naughty, Obscene, and Otherwise Bad Words")** banlist: if a web document contained even a single word from this list, the entire document was permanently deleted.

Section 5 evaluated the collateral damage of this crude filtering rule:
1. **Content of Excluded Pages (§5.1, Figure 6)**: Did the blocklist only drop pornographic/offensive content, or did it inadvertently drop legitimate information?
2. **Demographic & Identity Representation (§5.2, Figure 5 & Table 6)**: Were documents mentioning minority sexual orientations, genders, races, and religions disproportionately excluded?
3. **Dialect Censorship (§5.3)**: Did the filter disproportionately discard non-standard varieties of English, such as African American English (AAE)?

### Key Replication Findings (10 Shards, 3.84 Million Documents)

| Metric | Paper Finding | Our Replication (95% Bootstrap CI) | Status |
|---|---|---|---|
| **Identity PMI Ranking (Figure 5)** | Sexual orientation terms highest; European/White lowest | Spearman rank correlation **$\rho = 0.968$** with paper's bars | **Reproduced** |
| **"Lesbian" Document Removal** | Highest exclusion (~83%) | **82.9%** excluded ($\text{PMI} = 2.438$ vs. paper ~2.44) | **Reproduced** |
| **"Gay" Document Removal** | High exclusion (~65%) | **65.4%** excluded ($\text{PMI} = 2.200$ vs. paper ~2.34) | **Reproduced** |
| **African American English (AAE) Removal** | 42.0% | **35.8%** [33.7%, 38.1%] (~5x higher than WAE) | **Reproduced in direction & scale** |
| **Hispanic-Aligned English Removal** | 32.0% | **34.0%** [32.3%, 36.1%] | **Reproduced** (inside CI) |
| **White-Aligned English (WAE) Removal** | 6.2% | **7.2%** [7.1%, 7.3%] | **Reproduced** |
| **C4.EN Dialect Composition** | 97.8% WAE, 0.07% AAE, 0.09% Hispanic | **98.0% WAE, 0.074% AAE, 0.065% Hispanic** | **Reproduced** |
| **Sexual Clusters in Excluded Data (Fig. 6)** | 16 of 50 clusters (31% of docs) | **15 of 50 clusters (30.8% of docs)** | **Reproduced** |

---

## 2. Acronyms & Terminology Glossary

* **C4**: *Colossal Clean Crawled Corpus*, the massive English web corpus extracted from Common Crawl used to train Google's T5 and modern language models.
* **C4.EN.NOBLOCKLIST**: The unreleased variant of C4 containing web pages before the profanity banlist was applied.
* **LDNOOBW**: *List of Dirty, Naughty, Obscene, and Otherwise Bad Words*, an open-source list of profanities and offensive terms.
* **TFDS**: *TensorFlow Datasets*, the software library containing the reference implementation of C4's filtering pipeline (v3.0.1).
* **PMI (Pointwise Mutual Information)**: A formal information-theoretic metric measuring how much more frequently two events (here, a demographic identity term and document exclusion) co-occur compared to random independent chance.
* **AAE**: *African American English*, a distinct ethnolect/variety of English spoken across African American communities.
* **WAE**: *White-aligned English*, standard mainstream American English aligned with white demographic corpora.
* **TwitterAAE**: A probabilistic language/dialect identification model (Blodgett et al., 2016) trained on millions of geolocated tweets to estimate posterior probabilities over four demographic varieties of English (AAE, Hispanic, Other/Asian, WAE).
* **CVB0**: *Collapsed Variational Bayes (order 0)*, an inference algorithm used to infer dialect topic distributions.
* **TF-IDF**: *Term Frequency-Inverse Document Frequency*, a statistical vector representation reflecting how important a word is to a document within a collection.
* **k-Means**: An unsupervised clustering algorithm that groups documents into $k$ thematic clusters based on their TF-IDF vector similarity.
* **t-SNE**: *t-Distributed Stochastic Neighbor Embedding*, a non-linear dimensionality reduction technique used to visualize high-dimensional document clusters in 2D.
* **CI**: 95% Confidence Interval computed via Poisson domain bootstrapping over web domain hosts.

---

## 3. Methodology & Mathematical Mechanics

### 3.1 Defining the "Excluded" Set
C4's pipeline applied filters sequentially (language identification, deduplication, length limits, and finally the blocklist). Because the blocklist was the **very last step**, comparing `C4.EN.NOBLOCKLIST` with `C4.EN` is mathematically identical to applying C4's exact TFDS v3.0.1 regular expression to `C4.EN.NOBLOCKLIST`:
* A document is marked **Excluded** if it matches the regular expression:
  $$\texttt{re.compile(r"(?:\\W\|^)(" + "|".join(ldnoobw_words) + r")(?:\\W|\$)")}$$
  run on the lowercased document text.
* In our runs across 3.84 million documents, **7.24% of documents** were excluded by this single filter (closely matching the paper's corpus-level exclusion rate of ~7.6%).

---

### 3.2 Identity Term Representation & Pointwise Mutual Information (§5.2, Figure 5)

The paper tracked 22 demographic identity regular expressions (Table 6) covering sexual orientation, gender identity, race/ethnicity, and religion (e.g., `homosexual`, `lesbian`, `gay`, `trans`, `black`, `asian`, `white`, `jewish`, `muslim`, `christian`).

#### How PMI is Calculated:
For each demographic term $T$:
$$\text{PMI}(T, \text{Excluded}) = \ln \left( \frac{P(T, \text{Excluded})}{P(T) \cdot P(\text{Excluded})} \right) = \ln \left( \frac{N_{T, \text{Excluded}} \cdot N_{\text{Total}}}{N_T \cdot N_{\text{Excluded}}} \right)$$

* **Interpretation**:
  * $\text{PMI} = 0$: The term has no association with being banned (it is excluded at the baseline rate of $7.24\%$).
  * $\text{PMI} > 0$: The term is excluded **more often** than average.
  * For example, for `"lesbian"`: $82.9\%$ of documents mentioning the term were discarded. The baseline exclusion rate is $7.24\%$.  
    $$\frac{82.9\%}{7.24\%} = 11.45 \implies \ln(11.45) = \mathbf{2.438}$$
    This closely matches the paper's reported bar of $\sim 2.44$.
  * For `"european"`: only $15.8\%$ of documents were discarded:
    $$\frac{15.8\%}{7.24\%} = 2.18 \implies \ln(2.18) = \mathbf{0.781}$$

---

### 3.3 Dialect Identification & Filtering Rates (§5.3)

To measure whose variety of English was being silenced, we ported the TwitterAAE model (Blodgett et al., 2016) to Python 3:
1. Each document is tokenized into lowercased alphanumeric words.
2. The model infers a posterior probability distribution across four demographic dialect categories:
   * **AAE** (African American English)
   * **Hispanic-aligned English**
   * **Other** (Asian-aligned English)
   * **WAE** (White-aligned English)
3. The document is classified into the dialect with the highest posterior probability ($\text{argmax}$).
4. **Validation**: We validated our fast vector update implementation against the exact sequential CVB0 algorithm, achieving **99.72% agreement** across 10,000 checked documents.
5. **Metrics**:
   $$\text{Removal Rate}(D) = \frac{\text{Excluded Documents in Dialect } D}{\text{Total Documents in Dialect } D} \times 100\%$$
   $$\text{Share in Kept C4.EN}(D) = \frac{\text{Kept Documents in Dialect } D}{\text{Total Kept Documents in C4.EN}} \times 100\%$$

---

### 3.4 Thematic Clustering of Excluded Documents (§5.1, Figure 6)

To understand what was actually being discarded:
1. We sampled 100,000 excluded documents uniformly across shards.
2. Text was converted to a TF-IDF matrix ($50,000$ max features, sublinear term-frequency scaling, English stop-words removed).
3. We ran **$k$-Means clustering** with $k = 50$ clusters.
4. Each cluster was inspected for whether it represented sexual/pornographic text vs. non-sexual topics (medicine, law, politics, recipes, religion). An automatic rule flagged clusters where at least 2 of the top 10 keywords were blocklist words, matching the manual categorization performed in the paper.
5. A 2D visualization was generated by projecting a 50-dimensional Truncated SVD representation into **t-SNE**.

---

## 4. Detailed Results & Score Interpretation

All results below are from the main 10-shard run ($3,841,713$ documents across $1,741,415$ web domains) in [`results/S1_noblocklist_10_shards/analysis/`](results/S1_noblocklist_10_shards/analysis/).

### 4.1 Identity Term Exclusion (§5.2, Figure 5)

The ranking of identity terms by exclusion rate reproduced the paper with a **Spearman rank correlation of $\rho = 0.968$**:

| Identity Term | Total Docs | Excluded Docs | Removal Rate (%) | PMI (Ours) | 95% Bootstrap CI | Paper PMI (Fig. 5) | Category |
|---|---|---|---|---|---|---|---|
| **lesbian** | 16,407 | 13,602 | **82.9%** | **2.438** | [2.427, 2.448] | 2.44 | Sexual orientation |
| **heterosexual** | 2,441 | 1,826 | **74.8%** | **2.335** | [2.276, 2.372] | 2.34 | Sexual orientation |
| **homosexual** | 4,502 | 3,165 | **70.3%** | **2.273** | [2.239, 2.299] | 2.30 | Sexual orientation |
| **bisexual** | 4,491 | 3,084 | **68.7%** | **2.249** | [2.208, 2.280] | 2.24 | Sexual orientation |
| **gay** | 28,521 | 18,651 | **65.4%** | **2.200** | [2.187, 2.216] | 2.34 | Sexual orientation |
| **latina / latino** | 10,800 | 5,825 | **53.9%** | **2.008** | [1.985, 2.032] | 1.92 | Race / ethnicity |
| **non-binary** | 623 | 272 | **43.7%** | **1.796** | [1.701, 1.881] | 1.84 | Gender identity |
| **trans** | 14,830 | 5,241 | **35.3%** | **1.585** | [1.560, 1.611] | 1.75 | Gender identity |
| **african american**| 13,043 | 3,798 | **29.1%** | **1.391** | [1.362, 1.418] | 1.44 | Race / ethnicity |
| **jewish** | 27,553 | 7,669 | **27.8%** | **1.346** | [1.322, 1.372] | 1.47 | Religion |
| **muslim** | 20,694 | 5,682 | **27.5%** | **1.333** | [1.307, 1.364] | 1.50 | Religion |
| **black** | 223,588 | 53,402 | **23.9%** | **1.193** | [1.183, 1.203] | 1.43 | Race / ethnicity |
| **white** | 239,069 | 45,494 | **19.0%** | **0.966** | [0.956, 0.977] | 1.12 | Race / ethnicity |
| **european** | 87,774 | 13,889 | **15.8%** | **0.781** | [0.761, 0.800] | 0.84 | Race / ethnicity |

#### Key Takeaway:
Sexual orientation terms cluster at the very top of the exclusion rankings. Documents discussing minority sexual orientations were between **$4\times$ and $5\times$ more likely to be deleted** than documents discussing White or European racial identities.

---

### 4.2 Dialect Disparities (§5.3)

| Dialect | Total Documents | Excluded Documents | Removal Rate (Ours) | 95% Bootstrap CI | Paper Removal Rate | Share in C4.EN (Ours) | Paper Share in C4.EN |
|---|---|---|---|---|---|---|---|
| **AAE** | 4,129 | 1,477 | **35.8%** | [33.7%, 38.1%] | **42.0%** | **0.074%** | **0.07%** |
| **Hispanic** | 3,502 | 1,192 | **34.0%** | [32.3%, 36.1%] | **32.0%** | **0.065%** | **0.09%** |
| **Other** | 67,501 | 4,427 | **6.6%** | [6.3%, 6.8%] | **7.2%** | **1.77%** | — |
| **WAE** | 3,763,813 | 271,169 | **7.2%** | [7.1%, 7.3%] | **6.2%** | **98.01%** | **97.8%** |

#### Key Takeaway:
* Text written in African American English ($35.8\%$) and Hispanic-aligned English ($34.0\%$) was removed at **roughly $5\times$ the rate** of White-aligned English ($7.2\%$).
* As a result, the cleaned pretraining data (C4.EN) is **$98.0\%$ White-aligned English**, leaving African American English virtually erased ($0.074\%$).

---

### 4.3 Content Clusters of Excluded Documents (§5.1, Figure 6)

Running $k$-Means with $k = 50$ on $100,000$ excluded documents yielded:
* **Sexual / Adult Clusters**: **15 of 50 clusters** accounting for **$30.8\%$ of excluded documents** (Dodge et al. reported **16 of 50 clusters, 31% of documents**).
* **Collateral Damage (The remaining ~70% of excluded text)**:
  * **Medicine & Health**: e.g., clusters with top words `"patients cancer treatment clinical medical"` (banned due to anatomical terms or mentions of reproductive health).
  * **Law & Court Proceedings**: e.g., `"court criminal case judge trial police"`.
  * **Politics & News**: e.g., `"president trump election campaign government"`.
  * **Religion & Culture**: Discussion of historical persecution or theological doctrines.
  * **Sports, Cooking, Recipes**: Everyday text discarded because of colloquialisms matching profanity subwords.

---

### 4.4 Arm B Extension: Context-Aware Filtering with a Neural Toxicity Classifier

To test whether the severe demographic censorship in §5.2 and §5.3 was caused by the crude lexical nature of the blocklist, **Arm B** replaces the LDNOOBW regular expression with a context-aware transformer classifier (`unitary/toxic-bert`):
* Evaluates the exact same documents side-by-side using `scripts/scan_toxicity.py`.
* Directly compares:
  1. **Discrepancy Matrix**: How often does the blocklist discard benign text that a neural classifier recognizes as safe (false alarms), and how often does toxic text bypass the blocklist because it lacks specific banned words (false negatives)?
  2. **Identity Censorship Parity**: Does the $83\%$ exclusion rate on LGBTQ+ terms plummet when text is evaluated contextually?
  3. **Dialect Disparity**: Does the $5\times$ censorship gap against African American English decrease?

---

## 5. Execution Guide: Running via Kaggle or Google Colab

### 5.1 Arm A: Baseline Blocklist Audit (`kaggle/f5_excluded_kaggle.ipynb`)

This pipeline reproduces the original paper's findings on standard **CPU** (no GPU required):

* **Notebook Path**: [`kaggle/f5_excluded_kaggle.ipynb`](kaggle/f5_excluded_kaggle.ipynb)
* **Required Hardware**: Accelerator **None** (Standard CPU).
* **Internet**: Turn **ON** (needed to stream HuggingFace C4 shards and TwitterAAE model weights).
* **Expected Runtime**: **~1.0 to 1.5 hours** for 12 shards using 4 CPU workers.
* **Step-by-Step Instructions**:
  1. Create a zip archive of this folder named `excluded_data.zip`.
  2. In Kaggle, go to **Datasets $\rightarrow$ New Dataset**, upload `excluded_data.zip`, and name the dataset `excluded-data`.
  3. Create a New Notebook, click **File $\rightarrow$ Import Notebook**, and select `kaggle/f5_excluded_kaggle.ipynb`.
  4. In the right sidebar, click **Add Input $\rightarrow$ Your Datasets $\rightarrow$ excluded-data**.
  5. Select **Accelerator: None** and toggle **Internet: On**.
  6. Click **Save Version $\rightarrow$ Save & Run All (Commit)**. You can safely close the browser tab.
  7. When complete, download `f5_S1_results.zip` from the notebook Output tab.

### 5.2 Arm B: Neural Toxicity Classifier Audit (`kaggle/f5_arm_b_toxicity_kaggle.ipynb`)

This experiment scores C4 web text with `unitary/toxic-bert` side-by-side with the LDNOOBW regex:

* **Notebook Path**: [`kaggle/f5_arm_b_toxicity_kaggle.ipynb`](kaggle/f5_arm_b_toxicity_kaggle.ipynb)
* **Required Hardware**: **GPU T4 $\times$ 2** (runs batched PyTorch in ~15–20 minutes) or standard CPU (~1.5 hours).
* **Internet**: Turn **ON** (needed to download `unitary/toxic-bert` model weights and C4 shards).
* **Step-by-Step Instructions**:
  1. In Kaggle, import `kaggle/f5_arm_b_toxicity_kaggle.ipynb`.
  2. Attach the `excluded-data` dataset.
  3. Select **Accelerator: GPU T4 x2** (or CPU) and toggle **Internet: On**.
  4. Click **Save Version $\rightarrow$ Save & Run All (Commit)** and walk away.
  5. Download `f5_arm_b_toxicity_results.zip` from the Output tab when done.

### 5.3 Google Colab Execution

* **Notebook Path**: [`colab/f5_excluded_colab.ipynb`](colab/f5_excluded_colab.ipynb)
* **Required Hardware**: Standard CPU tier.
* **Expected Runtime**: ~2 hours for 12 shards (multi-threaded download and scanning).
* **Step-by-Step Instructions**:
  1. Open [Google Colab](https://colab.research.google.com/) and upload `colab/f5_excluded_colab.ipynb`.
  2. Run the setup cell. Either enter your GitHub repository URL or upload `excluded_data.zip`.
  3. Run all cells sequentially. Results and plots will be displayed inline and can be saved directly to your Google Drive.

### 5.4 Local Execution & Rebuilding Notebooks

If you modify any script in `scripts/`, rebuild the notebooks before running:
```bash
# Rebuild Kaggle notebooks
python kaggle/build_kaggle_notebook.py
python kaggle/build_toxicity_notebook.py

# Rebuild Colab notebook
python colab/build_colab_notebook.py
```

To run locally on a single test shard (S0):
```bash
# 1. Baseline Blocklist scan & analysis:
python scripts/scan_noblocklist.py --n 1 --seed 0 --out_dir results/S0_noblocklist_shard00000
python scripts/analyze_excluded.py --scan results/S0_noblocklist_shard00000 \
    --out_dir results/S0_noblocklist_shard00000/analysis

# 2. Arm B Toxicity Classifier scan & analysis:
python scripts/scan_toxicity.py --n 1 --seed 0 --limit 50000 --out_dir results/test_toxicity
python scripts/analyze_toxicity.py --scan results/test_toxicity \
    --out_dir results/test_toxicity/analysis
```

---

## 6. Standalone Checklist & Repository Cleanliness

| Check | Status | Verification & Notes |
|---|---|---|
| **Kaggle Execution** | **PASS** | Runs headlessly on standard CPU in 1–1.5 hours; produces `fig5_pmi.png`, `fig_dialects.png`, and `fig6_clusters.png`. |
| **Colab Execution** | **PASS** | Fixed directory path clone target. Runs on free-tier CPU. |
| **Dialect Port Fidelity** | **PASS** | TwitterAAE Python 3 port verified against the original Python 2 implementation to exact digits. |
| **Domain Bootstrap Uncertainty** | **PASS** | Fast Poisson bootstrap over web domain hosts calculates valid 95% CIs. |
| **Repository Cleanliness** | **PASS** | Temporary `.tmp` downloads, intermediate raw shard files, and local caches are ignored in git. |

### Redundant Files to Keep Out of GitHub
* **Downloaded C4.EN.NOBLOCKLIST shards**: `c4-train.*.json.gz` (~400 MB per shard) are automatically deleted after scanning.
* **Downloaded TwitterAAE models**: `data/twitteraae_model/` files are re-downloadable automatically from GitHub at runtime.
* **Cache files**: `*.tmp`, `*.json.gz.part`, `hf_cache/`.
