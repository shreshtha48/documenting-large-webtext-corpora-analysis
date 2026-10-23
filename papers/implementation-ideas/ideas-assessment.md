# Assessment of extension directions

**What this file is.** My assessment of your four proposed directions, plus three angles I think are technically stronger. Each one is scored against the trial brief: reproducible cheaply, extends a Dodge et al. finding, has a control, has quantified uncertainty, and has an alternative explanation.

**How it was checked.** Every external claim is tagged with how I verified it:
- **[PDF✓]**: I downloaded and read the paper.
- **[ABS]**: abstract only.
- **[RUN]**: I ran it myself in this session. The code is in [../../pilots/](../../pilots/).

Numbers labelled *pilot* come from small samples and are evidence that a direction is worth pursuing. They are not results.

---

## Bottom line

| Direction | Verdict | Why |
|---|---|---|
| 1. Misinformation × growth of machine-generated text | **Reshape** | The *growth* of LLM text in Common Crawl is already measured (DeGenTWeb). "Misinformation" has no validated, cheap measure. The sharper question, which my pilot supports, is whether **C4's own filters preferentially keep LLM text**. |
| 2. Bias originates in C4; controlled study; women and teens over time | **Keep, re-scoped** | The fine-tuning study isn't needed, because AllenAI's DataDecide suite already varies only the pretraining corpus. Women and teens can reuse Dodge's co-occurrence method. Teen representation in a *corpus* looks unmeasured. |
| 3. Popular safety eval sets built from C4 | **Keep. Strongest evidence, cheapest to run** | Pilot: author-written sets (TruthfulQA, BBQ, ToxiGen) show 0 long-item matches; web-sourced sets (BOLD, RealToxicityPrompts) show about 10% verbatim presence in C4. I found no prior measurement for these sets. |
| 4. A C4-trained model writes an eval set; check overlap with C4 | **Fold into 3** | Feasible, but alone it mostly re-measures memorisation. It becomes useful as the "how does overlap change measured behaviour?" half of direction 3. |

**My recommendation:** do **direction 3 with the DataDecide models** as the main extension: provenance, then behaviour. Keep the filter-enrichment result (angle A) as a second, short experiment if time allows. Both:
- run on a laptop or free Colab;
- have built-in negative controls;
- produce numbers anyone can re-check through public APIs.

---

## 1. Misinformation and machine-generated text over time

### What already exists

- **DeGenTWeb: A First Look at LLM-dominant Websites** (He, Ardi, Govindan, Madhyastha; arXiv:2605.00087, Apr 2026) **[PDF✓]**.
  - Classifies about 100k sites archived by Common Crawl.
  - Finds 6.0% (5,643/94,908) LLM-dominant. The fraction is "up from 2.1% in late 2022 to 29.4% in early 2025."
  - Uses the Binoculars detector on an H100 GPU. The false-positive rate on sites from before ChatGPT is 0.30% with their page filtering and 12.9% without it.
  - It **does not measure misinformation**. It does not cite Dodge et al.
- **Population-level estimators that need no per-document detector:**
  - Liang et al. 2024 (arXiv:2403.07183) **[PDF✓]**: maximum-likelihood mixture over adjective frequencies, using known-human and known-AI reference texts.
  - Kobak et al. (arXiv:2406.07016) **[PDF✓]**: "excess vocabulary" against a counterfactual linearly extrapolated from 2021–2022. They report that "from 2013 until 2019, no single word has ever shown excess frequency gap > 0", which is a built-in control.
- **Detector bias:** Liang et al. 2023 (arXiv:2304.02819) **[PDF✓]**. Seven detectors gave an average false-positive rate of 61% on non-native English (TOEFL) essays.
  - **This matters directly for you.** A per-document detector run on web text would partly measure "non-Western English", which is exactly the group Dodge et al. found under-represented.

### Problems with the direction as stated

- **"Older versions of C4" do not exist.** There is one C4, from the April 2019 crawl (see [../../research-empirical-methods/README.md](../../research-empirical-methods/README.md) §9). "Then vs now" means applying the C4 recipe to a newer crawl, for example `CC-MAIN-2025-30` with 100,000 WET files of about 73 MB each **[RUN: checked the file list]**. The 2019 side is free: it is C4 itself.
- **The growth is already shown.** Re-measuring "LLM text has increased" would replicate DeGenTWeb.
- **Misinformation has no cheap, validated, scalable measure.**
  - DeGenTWeb cites NewsGuard's tracker of "1,254 'Unreliable AI-Generated News' Websites". I have **not** checked whether that domain list is public.
  - A claim-level factuality classifier on web text would introduce its own errors that you couldn't separate from the effect.

### The sharper version: do C4's filters let LLM text through more easily?

C4's rules (ends in punctuation, at least 5 words per line, at least 3 sentences, no blocklist words) reward well-formed, clean prose, which is exactly what instruction-tuned models produce.

**Pilot [RUN]** ([../../pilots/pilot_mage_filters.py](../../pilots/pilot_mage_filters.py)):
- I ported TFDS C4 v3.0.1 `clean_page` plus the blocklist (no dedup, no langdetect).
- I ran it on the MAGE test split (yaful/MAGE): 56,819 texts, balanced human vs machine across 10 domains.

| Group | n | Kept by C4 filters | 95% Wilson CI |
|---|---|---|---|
| Human | 28,741 | **71.8%** | 71.3–72.4 |
| `gpt-3.5-turbo` + `text-davinci-003` | 4,472 | **90.6%** | 89.7–91.4 |
| All machine text (includes old small models such as T5, OPT) | 28,078 | 61.1% | 60.5–61.7 |

- **Within length bands, human vs modern model:** under 100 words 62.5% vs 81.1%; 100–250 words 87.1% vs 96.3%; 250+ words 69.8% vs 97.4%.
- **The blocklist alone** removes 6.7% of human texts (1,933) but 1.9% of the modern-model texts (85).

**Caveats:**
- MAGE is benchmark text, not raw web pages.
- Its "modern" models date from 2022–23.
- The human and machine texts are matched by domain, not item by item.
- This is grounds for a hypothesis, not a finding.

**A full design would:**
- run the same filters on RAID (liamdugan/raid; 2.27M train rows across domains and generators, verified on Hugging Face) and on real pages from a recent crawl, scored with a population-level estimator;
- **Control:** pages from `CC-MAIN-2019-18`, before ChatGPT, where the estimated LLM share should be about 0;
- **Alternative explanation:** register. LLM text resembles edited prose, so the filter may simply prefer *edited* text of any origin. To separate the two, compare against professionally edited human text such as news, which MAGE's `xsum` domain approximates.

**Why it extends Dodge et al.:** they showed the filter **removes** minority voices. This tests whether the same filter **enriches** machine text, its unmeasured mirror image.

---

## 2. Bias that originates in C4: a controlled study, and women and teens over time

### The controlled study does not require training

- **DataDecide** (AllenAI) **[RUN: checked the model card and repos]**: "a suite of models we pretrain on 25 corpora … over 14 different model sizes ranging from 4M parameters up to 1B", "runs over 3 random seeds", "token to parameter ratio of 100". There are C4 models at every size (`allenai/DataDecide-c4-4M` … `-1B`) alongside Dolma, DCLM, FineWeb-Edu, Falcon/RefinedWeb and others. The C4-1B repo has 104 checkpoint/seed branches. Models load with `pip install ai2-olmo`.
- **Paloma baselines** **[RUN: repos exist]**: `allenai/paloma-1b-baseline-{c4, mc4, dolma, pile, redpajama, falcon-refinedweb}`.
- **So:** your "same model, with vs without C4" study becomes **inference-only**: the C4 model vs other-corpus models of the same size, with 3 seeds giving the noise floor.
  - This is cleaner than fine-tuning, which mixes the original pretraining with the new data.
  - Note: the model card describes the suite but does not list every hyperparameter, so check the DataDecide paper for exactly what is held constant.

### Race and ethnicity representation over time

- Reuse Dodge's A.7 method (lexicon positivity in paragraphs mentioning a group; see methods dictionary §6) on:
  - C4 itself (2019);
  - the same C4 recipe applied to a recent crawl sample.
- **Already done across corpora, not over time:** WIMBD **[PDF✓]** measured average sentiment for gender, racial and religious terms across 10 corpora with TextBlob. It reports weak positivity overall, "with the exception of those including 'black' being uniformly more negative across all corpora", and no substantial variation between datasets.

### Women

- The UnQover repo (`allenai/unqover`, which Dodge et al. used) ships **gender** templates alongside ethnicity, nationality and religion (repo README **[RUN: read]**). So the paper's model-probe-plus-corpus pairing can be repeated for gender directly.
- WIMBD's code also has male/female term lists ([../../research-empirical-methods/reference-code/wimbd/wimbd/sentiment_coocurrence/demographic_terms.json](../../research-empirical-methods/reference-code/wimbd/wimbd/sentiment_coocurrence/demographic_terms.json)).

### Teens

This is a novel angle, as far as I could find.

- **Wolfe et al. 2024**, "Representation Bias of Adolescents in AI" (arXiv:2408.01961) **[PDF✓ for abstract]**:
  - "more than 50% of the 1,000 words most associated with teenagers in the pretrained GloVe SWE reflect such problems" (societal problems);
  - "30% of outputs from GPT2-XL and 29% from LLaMA-2-7B … discuss societal problems, most commonly violence, but also drug use, mental illness, and sexual taboo."
  - They measured **embeddings and models, not a pretraining corpus**, and the paper does not cite Dodge et al. (its two "Dodge" mentions are K. A. Dodge, a psychologist).
- **Gap:** measure teen-term co-occurrence with Wolfe's problem categories **inside C4**, compare it with "adult/adults", and check how it shifts in a recent-crawl sample.
- **Control:** the same analysis for a neutral age group such as "adults", or "seniors" as a second age group.
- **Caveat:** "teen" terms are ambiguous ("teen" in porn spam), and the blocklist already removed much of that from C4. That interaction is itself worth reporting: the blocklist may *improve* teen representation while hurting LGBTQ+ representation.

### Feasibility

- Corpus co-occurrence counts on a few shards: laptop.
- DataDecide 1B inference: free Colab GPU (my estimate; not tested).

---

## 3. Popular safety and bias eval sets built from C4

### Provenance, from each benchmark's own paper **[PDF✓ or ABS]**

| Benchmark | Where the items come from |
|---|---|
| TruthfulQA | "We crafted questions …" (817, written by the authors) |
| BBQ | "constructed by the authors" |
| CrowS-Pairs | Crowdsourced |
| ToxiGen | Machine-generated |
| BOLD | "Our LM prompts are extracted from English Wikipedia articles" |
| RealToxicityPrompts | "We select our prompts from the OPENWEBTEXT CORPUS" |

- **AuditBench** exists (arXiv:2602.22755, "Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors"). It is about models with hidden behaviours, not a text dataset that could come from C4, so it doesn't fit this question.
- **Prior coverage:** WIMBD's contamination table (82 PromptSource datasets) includes **CrowS-Pairs: 0.00% in C4** **[PDF✓]**. I found no measurement for BOLD, RealToxicityPrompts, TruthfulQA, BBQ or ToxiGen in C4. These arXiv searches returned nothing relevant: "RealToxicityPrompts contamination", "safety benchmark contamination pretraining", "BOLD contamination pretraining corpus", "TruthfulQA contamination pretraining data".

### Pilot [RUN]

Setup ([../../pilots/pilot_c4_overlap.py](../../pilots/pilot_c4_overlap.py)):
- Exact matches in all of C4.EN train via the Infini-gram API (`v4_c4train_llama`).
- 100 items per set, sampled from 5 random 100-row blocks. This is clustered sampling; a full run should sample uniformly.
- Matching is case-sensitive, so these are lower bounds.
- RealToxicityPrompts' mis-encoded text was repaired before querying.

| Set | Items ≥ 8 words found | Notes |
|---|---|---|
| **Positive control:** Dodge Table 5 XSum summaries | **7/8** | The method works |
| TruthfulQA questions | **0/82** | All 3 hits were short generic questions ("Who invented the Internet?": 30) |
| BBQ contexts (race/ethnicity) | **0/100** | |
| ToxiGen | **0/93** | Only hit: "existence does not exist" |
| StereoSet intersentence contexts | 1/39 | Hits are short generic sentences |
| **BOLD source sentences** | **10/100** (all ≥ 8 words; Wilson CI about 5–17%) | e.g. "Scott Joplin was an American composer and pianist." |
| **RealToxicityPrompts full sentences** | **9/100**; 7/85 among ≥ 15 words | Real web sentences present verbatim in C4 |

**Reading:**
- The benchmarks you named (TruthfulQA-style, written by the authors) are **not** built from C4. That part of the hypothesis is answered *no*, cleanly.
- **Web-sourced** bias and toxicity benchmarks partly **are** inside C4. Data poisoning is the wrong frame here: this is provenance overlap, the same phenomenon as Dodge's §4.2, extended to safety evals.

### Full design: provenance, then behaviour

1. **Provenance.** Run all items of BOLD (7,201 rows) and RealToxicityPrompts (99,442; sample 10k) through the Infini-gram API, querying a few casing and punctuation variants to approximate Dodge's normalisation. Report rates with Wilson CIs, by item length.
2. **Behaviour.** Compare `DataDecide-c4-1B` against the same-size Dolma, FineWeb-Edu and DCLM models on in-C4 and not-in-C4 prompts, using a difference-in-differences design.
   - **Outcome:** continuation toxicity (RealToxicityPrompts) or regard and sentiment (BOLD), plus **verbatim continuation**, meaning the model reproduces the original next words.
   - **Control:** the non-C4 models on the same prompt split. Matching on prompt length handles length.
   - **Uncertainty:** 3 seeds per corpus, plus bootstrap over prompts.
   - **Alternative explanation:** in-C4 prompts may differ in topic or toxicity from out-of-C4 prompts. The difference-in-differences cancels any content difference shared across models. Report the residual imbalance too.

This is where your **direction 4** fits. Instead of asking a C4 model to write a new eval set, measure whether a C4 model **recites** the in-C4 benchmark items. That is the concrete version of "can we reliably use it to test any other model?"

---

## 4. A C4-trained model generates an eval set

- **Feasible.** The DataDecide or Paloma C4 1B models can generate items, and Infini-gram can check their n-gram overlap with C4.
- **Weakness as a standalone project:**
  - The likely result is "some generated items are memorised C4 text", which is already established in general. Yang et al. (arXiv:2311.04850) **[PDF✓]** found "such contamination in synthetic dataset generated by GPT-3.5/4".
  - A 1B base model also writes poor eval items.
- **Verdict:** fold it into direction 3 as described above.

---

## 5. My ranking

| Rank | Direction | Novelty (from my searches) | Laptop/Colab | Control built in | Time to first result |
|---|---|---|---|---|---|
| 1 | **3 + 4: safety-eval provenance → behaviour** | No prior measurement found for BOLD/RealToxicityPrompts in C4. WIMBD covers only CrowS-Pairs | Yes: API calls plus 1B inference | Yes: author-written sets, and non-C4 models | Hours. The pilot already runs |
| 2 | **A: C4 filters keep LLM text** (reshaped direction 1) | No paper found testing this | Yes: CPU only | Yes: pre-ChatGPT crawl, and edited-human text | Pilot done |
| 3 | **2: women and teens in C4 vs a recent crawl** | Corpus-level teen measurement not found | Yes | Neutral age group; DataDecide seeds | About a day |
| 4 | Dialect × length (from paper-ideas-extension.md, A) | — | Yes | Fake blocklist | About a day |

The brief warns against trying to do too much: "one question answered well". So pick **one** of ranks 1–2 as the extension, and keep the other as a paragraph of "next steps".

---

## 6. Sources used in this file

**Not in the 829 citing papers:** DeGenTWeb, Liang et al. 2023 and 2024, Kobak et al., Wolfe et al., MAGE, RAID and AuditBench do not cite Dodge et al. (checked in the PDFs where downloaded). They are related work.

**Code and data verified to exist:**
- Hugging Face datasets: `yaful/MAGE`, `liamdugan/raid`, `truthfulqa/truthful_qa`, `AmazonScience/bold`, `allenai/real-toxicity-prompts`, `heegyu/bbq`, `toxigen/toxigen-data`, `McGill-NLP/stereoset`.
- Models: `allenai/DataDecide-c4-*`, `allenai/paloma-1b-baseline-c4`.
- `allenai/unqover`.
- Common Crawl crawl list: 129 crawls, the newest `CC-MAIN-2026-43`.

**Not verified:**
- whether NewsGuard's site list is public;
- DataDecide's full list of held-constant hyperparameters;
- Colab runtimes for 1B inference.
