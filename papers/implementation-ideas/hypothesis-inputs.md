# Hypothesis inputs (NOT the scoping note)

> **Status: input material written with AI help, for you to argue with.** The trial brief says the scoping note and the hypothesis must be written by you, in your own words, without AI. Do not paste from this file. Use it to decide what *you* will claim and predict, then write the scoping note yourself.

---

## 1. Critique of the current framing

**Framing under review:** "Reproduce the ethnicity bias in the text the blocklist filters out, confirm Dodge et al.'s claims are true, then split the text into machine-generated vs human and compare bias."

### 1.1 It mixes two separate findings

- **F5, the blocklist (§5).** The highest PMI with being filtered is for **sexual-orientation** terms. Dodge's Fig. 5 ranks ethnicity terms lower. Dialect removal: AAE 42%, Hispanic-aligned 32%, WAE 6.2%.
- **F4, ethnicity sentiment (§4.3, A.7).** The Jewish vs Arab positivity gap is measured in the text C4 **kept**. It has nothing to do with the blocklist.

**Decide which you are reproducing:** F5, F4 or both. Name it explicitly.

### 1.2 "Rule hierarchy" and "semanticity" are not in the paper

TF-IDF appears only in the descriptive clustering of removed documents (§5.1: k-means with k=50, 16/50 clusters "largely sexual", 31% of excluded documents). It is not a bias claim, so don't make it the central frame.

### 1.3 Don't presuppose the outcome

"We discovered that all the claims are true" can't be written before running anything. The brief rewards reporting discrepancies, and two are already documented in [../../findings.md](../../findings.md):
- Dodge §2 swaps the C4 thresholds: it says 3 words and 5 sentences, where the code and the T5 paper say 5 words per line and 3 sentences.
- Blocklist removal of AAE: **37.2%**, computed from Deas et al. 2025's counts, vs Dodge's **42%**.

### 1.4 Five problems with splitting C4 into machine-generated vs human text

1. **What "machine-generated" means in C4.** C4 is the April 2019 crawl. Its machine text is **machine translation and OCR**, mainly patents (Dodge §4.1), not LLM output.
2. **Domain confound.** Patents contain almost no ethnicity or gender mentions. A machine-vs-human bias comparison inside C4 becomes "patents vs everything else".
3. **Detector bias.** Liang et al. 2023 (arXiv:2304.02819) report an average 61% false-positive rate on non-native English (TOEFL) essays across seven detectors. On a dialect or ethnicity comparison, the detector itself would create a bias difference.
4. **Hardware.** Binoculars (`ahans30/Binoculars`) sets its threshold "using Falcon-7B and Falcon-7B-Instruct". Running two 7B models probably exceeds a free Colab T4's 16 GB. This is my estimate; I haven't tested it.
5. **"Increased over time" needs a second crawl**, because there is only one C4. DeGenTWeb (arXiv:2605.00087) already reports LLM-dominant Common Crawl sites rising "from 2.1% in late 2022 to 29.4% in early 2025".

---

## 2. A framing that holds up better

### 2.1 Reproduction (F5, plus F4 if time allows)

Run on k shards of `allenai/c4`:

1. **Blocklist check.** Verify that `en` = `en.noblocklist` minus the documents matching the LDNOOBW regex (TFDS v3.0.1), compared by URL. Script: [../../pilots/shard_check.py](../../pilots/shard_check.py). A one-shard run is in progress; its output goes to `pilots/shard_check_output.txt`.
2. **Identity-term PMI ranking** with Dodge's Table 6 regexes. Dodge predicts sexual-orientation terms rank highest.
3. **Dialect removal rates** with the Blodgett et al. classifier (argmax). Compare with 42% / 32% / 6.2%, and explain the 37.2% vs 42% gap.
4. **F4 positivity gap with VADER.** Dodge reports a 2.0-point gap with VADER.
   - The SocialSent subreddit lexicons, needed for the headline 7.5-point gap, are hosted at `nlp.stanford.edu/projects/socialsent`. That site was unreachable on every attempt in this session.
   - If it stays unavailable, say F4 can't be reproduced as stated, and why.

### 2.1a One-shard pilot result (shard `00000-of-01024`)

Full output: [../../pilots/shard_check_output.txt](../../pilots/shard_check_output.txt).

**Blocklist check:**
- `noblocklist` has 384,171 documents. The regex flags 27,675 (7.20%).
- **All 27,675 flagged documents are absent from `en`.** No flagged document appears in `en`.
- **2,302 documents (0.60%) pass the regex but are also missing from `en`.**
- Agreement with the released `en` shard: 99.40%.
- **The 2,302 are unexplained so far.** Candidate causes I have *not* tested:
  - differences in shard assignment between the two variants;
  - the 3-sentence deduplication interacting with neighbouring documents;
  - some other difference in how the `en` shard was generated.

  This needs checking before you claim the blocklist is applied "exactly".

**Identity PMI** (natural log, document level, bootstrap 95% CI over documents with B=200; Dodge Table 6 regexes):
- Highest: lesbian 2.44, heterosexual 2.34, homosexual 2.27, bisexual 2.26, gay 2.20.
- Lowest: christian 1.14, white 0.96, european 0.80.
- **Sexual-orientation terms ranking highest, and the 0–2.5 range, match Dodge Fig. 5 qualitatively.**
- Ethnicity terms sit in the middle: latinx 2.01, asian 1.59, african american 1.37, jew 1.34, muslim 1.32, black 1.20.

**Terms outside Dodge's list (for the gender and teen angle):**

| Term | PMI | Documents removed |
|---|---|---|
| teen | 1.73 | 40.7% |
| girl | 1.73 | 40.5% |
| female | 1.71 | — |
| woman | 1.44 | 30.4% |
| man | 1.36 | 28.2% |

The blocklist removes "teen" and "girl" documents at very high rates, very likely because of porn spam. That is relevant if you study the representation of teens and women in what *remains*.

**Counts that matter for statistical power:**
- In the documents the blocklist keeps, one shard has **2,923 " Jewish " paragraphs** and **1,381 " Arab " paragraphs**, matched as in Dodge A.7.
- Only **105 patents.google.com documents** per shard.
- This **confirms the domain concern in §1.4**: C4's machine-translated text is too small, and too empty of identity mentions, to compare bias against human text inside C4. That is a reason to use the matched-generation design below.

### 2.2 Extension: same measurements, machine vs human text, with matched contexts

Create the machine text yourself instead of trying to detect it in the corpus:

1. Take human C4 paragraphs mentioning a target group: women/men, an ethnicity, an LGBTQ+ term, or AAE-classified text.
2. Have 2–3 open LLMs **continue or rewrite** each one from the same prefix.
3. Run identical measurements on the human original and each machine version:
   - blocklist removal rate per group;
   - group-sentiment positivity, the F4 method;
   - dialect-classifier distribution.

**Why this design:**
- It avoids detector bias, because labels are known by construction.
- It avoids the patents-vs-everything domain confound, because contexts are matched.
- It links to the increase in LLM text through the filter pilot ([../../pilots/pilot_mage_results.json](../../pilots/pilot_mage_results.json)). C4 v3.0.1's filters kept **90.6%** of `gpt-3.5-turbo` / `text-davinci-003` texts vs **71.8%** of human texts on MAGE. LLM text is therefore more likely to survive into C4-style corpora.
  - Caveat: MAGE is benchmark text, not web pages.

### 2.3 Candidate hypotheses (you pick direction and wording)

- **H1, exclusion.** Machine continuations of LGBTQ+ or AAE contexts trip the blocklist less often than the human originals, so identity-exclusion PMI is lower in machine text.
- **H2, dialect erasure.** LLM rewrites of AAE-classified text are reclassified as White-aligned English. Related work in the citing set: *Standard Language Ideology in AI-Generated Language* (arXiv:2406.08726).
- **H3, sentiment.** The Jewish vs Arab, or women vs men, positivity gap differs between machine and human continuations. **Commit to a predicted sign before running.**

### 2.4 Design checklist, mapped to the brief

| Brief requirement | How to meet it |
|---|---|
| Hypothesis stated before running | Your scoping note, with the predicted direction for H1–H3 |
| Control or baseline | Neutral, non-identity prefixes, to measure the generator's own shift. Also: more than one generator; multiple sampling seeds; outputs length-matched to the human original |
| Quantified uncertainty | Bootstrap over prefixes, grouped by site or domain; Wilson CIs for rates; a paired test (human original vs machine version of the same prefix) |
| Alternative explanations | **Length:** longer texts are more likely to contain any blocklist word. **Register:** LLM output resembles edited prose. **Decoding:** settings such as temperature. **Prompt template.** Say what result would separate each one from your explanation |

### 2.5 Optional real-web arm (stretch)

Apply the C4 recipe ([../../pilots/c4_filters_v301.py](../../pilots/c4_filters_v301.py)) to a sample of WET files from `CC-MAIN-2019-18` and a 2025 crawl, and compare.
- Only do this if the detector problem in §1.4 is handled.
- Options: a population-level estimator (Liang et al. 2024, arXiv:2403.07183), or DeGenTWeb-style site-level aggregation.
