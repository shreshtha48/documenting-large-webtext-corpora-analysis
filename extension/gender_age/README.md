# Extension: gender and age in C4 (building on §4.3 and §5)

The reproduction measured bias by **ethnicity** (§4.3) and exclusion by **identity term and dialect** (§5). This
extension asks the same questions **by gender first, then zooms in on race within each gender**, and adds age.
Every run, including failed ones, goes in [EXPERIMENTS.md](EXPERIMENTS.md).

## Questions

1. **Model probe (primary).** Does UnifiedQA-3B (T5, pretrained on C4) link women or men, or an age group, with
   negative answers more often? Within each gender, which races stand out, e.g. is "Black woman vs Black man"
   different from the gap for other races?
2. **Corpus sentiment (secondary).** Are C4.EN paragraphs about women more or less positive than those about
   men, overall and per race ("Black women" vs "Black men")?
3. **Exclusion.** Does the blocklist remove women's or men's mentions more, and which women (by race and dialect)?
4. **Time: on hold.** C4's timestamps are all April 2019 (the crawl date), and no later C4 exists. The URL year is
   still computed by the scan; it is not part of the results.

Write your own predictions before running (the brief requires the hypothesis in your words).

## Where the question format comes from (checked against the sources)

| Piece | Source | Ours? |
|---|---|---|
| Question generator | UnQover's script (Li et al. 2020), **run unchanged**, as Dodge §4.3 did ("we used their script") | no |
| 50 negative / positive attribute pairs | UnQover `biased_ethnicity`, the list Dodge used | no |
| Mixed-gender templates | UnQover `gender_noact` (4 templates) | 2 edits, below |
| Names | UnQover female / male name lists, via UnQover's own `mixed_gender` pairing | no |
| Scoring and positivity | the reproduction's E6 (Dodge A.7 credit rule, summed log-likelihood) | no |
| Applying ethnicity attributes to gender and age | — | **yes** |
| Age and gender × age nouns, noun counterparts | — | **yes** (subject lists) |

- **What Dodge et al. and UnQover cover.** Dodge et al. ran UnQover for ethnicity only. UnQover tests gender only as
  names × occupations, which gives no positive / negative score. So a gender or age probe is necessarily an
  extension.
- **Our two template edits.** The mixed-gender templates are used with:
  - `[p_act]` → `[p_act-single]`: UnQover's own singular form of the same attributes, because `[p_act]` is plural
    ("Who were barbarians?");
  - an article before each noun ("A teenager…"). UnQover's own a / an smoothing then fixes the article.

  Both are in `third_party/unqover/word_lists/slots/*_single`, marked NOT UPSTREAM. Every vendored UnQover file is
  byte-identical to upstream.
- **Validation against a published benchmark.** We also run **BBQ** (Parrish et al., ACL Findings 2022), which
  covers gender identity, age and race × gender with author-written, literature-sourced stereotypes. BBQ's paper
  evaluated UnifiedQA (11B, "RACE-style" input), so we score its data with the same model family, using BBQ's own
  metric.
- **Checking our BBQ metric code.** It is a port of the official R script. On the authors' published 11B
  predictions it reproduces their Figure 3 exactly for 6 categories (Age 29.8 ambiguous / 1.2 disambiguated,
  Disability, Nationality, Religion, Sexual orientation, SES), within 0.7 for Physical appearance, but not for
  Gender identity. BBQ's script notes data versions differ by 16 examples; see EXPERIMENTS.md X6.

## Method

### 1. Probe (UnQover-style; `generate_questions.py`, `analyze_probe.py`)

| List | People | Contexts | Account |
|---|---|---|---|
| `gender_names` | 10 female × 10 male UnQover names, both orders | 40,000 | 1 |
| `gender_names_ctrl` | **control**: 5 names per gender against the same gender | 8,000 | 1 |
| `gender_nouns` | woman/man, lady/gentleman, mother/father, wife/husband, sister/brother, daughter/son | 2,400 | 1 |
| `age` | 4 age groups × 2 wordings, all pairs; **control**: same group, other wording | 11,200 | 2 |
| `gender_age` | teenage girl/boy, young woman/man, middle-aged woman/man, elderly woman/man | 11,200 | 2 |
| `gender_ethnicity` | "X woman" vs "X man" for the 15 Table 7 ethnicities | 6,000 | 2 |

**Gender first, then race:**
- `gender_ethnicity` gives women's positivity against men of the same race, for each race. It is compared with the
  pooled gap, and an ethnicity is flagged as an anomaly when the bootstrap difference survives Holm correction.
- Race *within* each gender is the reproduction's E6 split by template gender. Its templates come in man and woman
  versions, so no new run is needed.
- **Clean contrasts** throughout: each axis is compared only on pairs that differ on that axis alone.
- **Controls:**
  - same-gender name pairs and same-age-group pairs (noise floor);
  - every result also without the 4 gendered attributes;
  - position bias (share of option (a));
  - same answer to the negative and the positive question.
- **CIs:** bootstrap over the 50 attributes.

### 2. Probe on BBQ (`prepare_bbq.py`, `score_bbq.py`, `analyze_bbq.py`; account 2)

- **Data:** Age 3,680, Gender_identity 5,656, Race_x_gender 15,960 examples, pinned to nyu-mll/BBQ @ bea11bd.
- **Scoring:** UnifiedQA-3B, RACE-style input. Both answer rules: BBQ's (generation matched to an option) and ours
  (log-likelihood).
- **Metrics:** BBQ's bias scores (ambiguous scaled by 1 − accuracy). Race_x_gender is split by BBQ's own conditions:
  - same race, different gender (gender first);
  - same gender, different race (race within gender);
  - the Dodge-style credit on the two people, with anomaly tests.
- **CIs:** bootstrap over BBQ templates.
- **Reference:** the authors' published 11B predictions, analysed the same way.
- **Two data facts:**
  - BBQ's same-gender, different-race items are mostly about women: men appear in only 1–6 templates per race. So
    BBQ cannot support race-within-men comparisons for most races, and our UnQover lists and E6 are balanced instead.
  - 344 Race_x_gender items have an empty group label (a BBQ release defect). They are kept in the bias score and
    left out of the group analyses.
- Groups seen in fewer than 5 templates or 50 trials are listed, not tested.

### 3. Corpus (`scan_corpus.py`, `analyze_corpus.py`, `classify_paragraphs.py`)

12 `en.noblocklist` shards (the reproduction's indices), one pass:
- the blocklist decision;
- 80 terms: gender nouns, pronouns, age groups, gender × age, and gender × 12 ethnicities ("Black women");
- dialect;
- lexicon sentiment of the paragraphs that mention a term, in kept documents (= C4.EN) and in excluded ones.

Outputs:
- **Gender first, then race:** "<race> women" minus "<race> men", for exclusion rate and for sentiment, each
  compared with the pooled gap (Holm).
- **Mechanical exclusions:** LDNOOBW contains `girl on`, `big black`, `sultry women`, `wrapping men`,
  `white power`... A page saying "a girl on the bus" is removed because of that phrase. PMI is reported with and
  without exclusions caused only by entries that contain the term's own words.
- **Term words are removed before sentiment counting**, because SocialSent scores "man" as +2.1.
- **Positive controls:** paper-style " Jewish " / " Arab " (Arm B's 7.1 gap); women / men / female / male PMIs vs
  Fig. 5.
- **Dialect × gender:** removal of AAE documents that mention women vs men.
- **Classifier:** RoBERTa sentiment on sampled paragraphs.
- **CIs:** bootstrap over web domains.

### Alternative explanations to address in the write-up

- **Topic, not attitude:** negative paragraphs about women often report harm done *to* women. The paragraph-level
  methods can't separate this; the probe isn't affected.
- **Polysemy:** "man" (Iron Man), "white" / "black" as colours, "seniors" (students).
- **Gendered attributes**, and names that carry ethnicity and age.
- **Attribute transfer:** the 50 attributes were written for ethnicity stereotypes. BBQ's gender and age
  stereotypes are the check on that.

## Results so far

**E6 split by template gender:**
- Same ethnicity ranking for men and women (rank correlation 1.0).
- Every group moves by 2.4 points or less.
- Jewish − Arab gap: 64.9 (men) vs 65.7 (women).

**BBQ, published UnifiedQA-11B predictions (re-analysed):**
- Women's positivity against a man of the same race: 59.9 [46.5, 71.3].
- Arab is the one race where it is lower than the pooled value (Holm p < .05).

## How to run (Kaggle; scripts and data are embedded in each notebook)

| Notebook | Account | Accelerator | Time |
|---|---|---|---|
| `kaggle/ga_probe_account1.ipynb` (names, control, nouns) | 1 | GPU T4 ×2 | ~1.5 h |
| `kaggle/ga_probe_account2.ipynb` (age, gender × age, gender × ethnicity, BBQ) | 2 | GPU T4 ×2 | ~2 h |
| `kaggle/ga_corpus.ipynb` | either, at the same time | CPU | ~1.5–2 h |

- Use GPU T4 ×2, not TPU: the scorer is PyTorch/CUDA.
- Put each zip's `results/` into `extension/gender_age/results/`.
- After editing a script, run `python kaggle/build_notebooks.py`.
- E6 split, locally on CPU: `python scripts/analyze_probe.py --e6 --out_dir results/E6_gender_split`.
