# Extension scoping: your four ideas, checked for feasibility

> Input material written with AI help, not your scoping note. The brief asks you to write the hypothesis yourself.

**Constraint from the brief:** about 5 hours for the extension, and "one question answered well". Each design must
have a hypothesis stated in advance, a control, quantified uncertainty and one alternative explanation. Your scoping
note (`reproduction/PROPOSAL_DRAFT.md`) already commits you to two predictions: that machine-translated or LLM text
amplifies bias, and that a sentiment classifier may change the lexicon picture. Whatever you pick will be read
against those predictions.

## Summary

| # | Idea | Data available? | Novelty | Cost | Verdict |
|---|---|---|---|---|---|
| 1 | Safety-eval contamination in C4 | Yes. All sets on HF; infini-gram indexes all of C4.EN; pilot done | No prior measurement found except CrowS-Pairs (WIMBD: 0%) | ~2–3 h, CPU + API | **Primary** |
| 2 | Gender / age bias (F4 method on a new axis) | Yes. UnQover ships gender templates + 69 male / 69 female names; BBQ has `Age` and `Gender_identity`; WIMBD has gender term lists | Gender sentiment in corpora partly done (WIMBD, Pile §6.4). Teens in a corpus: not found | ~3 h, reuses F4 scripts + 1 Kaggle GPU run | **Secondary, if time** |
| 3 | Machine-translated vs human text: which is more biased? | No off-the-shelf English MT detector found | Open | High, and the measurement is weak | **One paragraph in the write-up, not run** |
| 4 | More AI-generated text, more bias over time | No later C4. FineWeb has 110 per-dump subsets (2013-20 to 2025-26) | Partly done (DeGenTWeb measures LLM growth) | High | **"Next week" item** |

## 1. Safety-eval contamination (primary)

**Question.** What share of popular safety / bias benchmark items appears verbatim in C4.EN, and does it depend on
how the benchmark was built?

**Sets, grouped by how they were built (this is the design):**

| Group | Sets | Expected |
|---|---|---|
| Web-sourced, text pre-dates April 2019 | BOLD (Wikipedia), RealToxicityPrompts (OpenWebText), ETHICS commonsense (Reddit AITA, check post dates) | > 0 |
| Author-written or crowdsourced, pre-2019 | CrowS-Pairs, StereoSet, WinoBias / Winogender, BBQ | ≈ 0 |
| Written after April 2019 (**negative control**: cannot be in C4) | TruthfulQA, ToxiGen, XSTest, HarmBench, AdvBench, Do-Not-Answer | 0; any hit = false-positive rate |
| **Positive control** | XSum targets (Dodge Table 5), already done | ~6% (our F3 number) |

AuditBench is not a text dataset (it is a set of models with hidden behaviours), so it can't be checked against C4.

**Pilot already run** (`pilots/pilot_c4_overlap.py`, 100 items each): TruthfulQA 0/82, BBQ 0/100, ToxiGen 0/93,
BOLD 10/100, RealToxicityPrompts 9/100.

**Design.**
- Reuse `reproduction/benchmark-contamination/scripts/query_infinigram.py` (pacing, OR-groups, case variants).
- 300 items per set, sampled uniformly (the pilot sampled in clusters). Wilson 95% CIs.
- Report by item length bucket: short generic items ("Who invented the Internet?") match by chance.
- **Alternative explanation:** a hit on a web-sourced set may mean C4 and the benchmark both copied the same
  Wikipedia sentence, not that the test set leaked. That is the point of the grouping; say so.
- **Stretch (behaviour):** do in-C4 items change model behaviour? Compare `allenai/DataDecide-c4-1B` with a same-size
  non-C4 model on in-C4 vs not-in-C4 RealToxicityPrompts prompts (difference in differences). Only if time is left.

## 2. Gender and age bias (secondary)

**Model probe (Table 7 on a new axis).** UnQover's gender templates pair male and female *names* with occupations,
not with positive / negative attributes. The ethnicity attributes (`biased_ethnicity`, 206 lines) can be crossed with
gendered subjects instead: either the 69 + 69 names, or nouns ("man"/"woman", "teenager"/"adult"/"elderly person").
Score with UnifiedQA-3B on Kaggle, as in Arm A. Note it is a new probe, not the paper's.

**Corpus (A.7 on a new axis).** Re-run `scan_c4_paragraphs.py` with gender and age terms on the same 12 shards;
count SocialSent / VADER positivity.

**Controls and confounds you must handle:**
- Pronouns ("he"/"she") are everywhere; use nouns. "Man" has other senses; "teen" and "girl" are porn-spam heavy (the
  pilot found 40% of "teen" docs removed by the blocklist), so what remains in C4 is already filtered.
- A neutral control pair ("adult"/"person") for the generic sentiment level.
- Your scoping note asks whether a sentiment **classifier** changes the lexicon picture. This is the place to test it
  (e.g. a RoBERTa sentiment model on the same paragraphs, on GPU).

## 3. Machine-translated text (why not to run it)

- **No detector.** I found no released English translationese / MT detector checkpoint, only research classifiers
  without weights. Training one is beyond 5 hours.
- **Detector bias.** AI-text detectors flag non-native English at a 61% false-positive rate (Liang et al. 2023). An MT
  detector would likely do the same, and non-Western English is exactly the group your comparison is about.
- **Too little overlap.** C4's machine-translated text is mostly patents (~105 patents.google.com docs per shard), which
  almost never mention ethnicities.

In the write-up, report it honestly: you predicted MT text amplifies bias, and explain why C4 doesn't let you test it
cheaply. That is a legitimate "what would I do next" item.

## 4. AI-generated text over time

There is only one C4 (April 2019 crawl). Two ways to get a "later" corpus:
1. Run the TFDS C4 v3.0.1 filters (`pilots/c4_filters_v301.py`) on WET files from a 2025 crawl.
2. Use FineWeb's per-dump subsets (110 dumps, verified on Hugging Face). Same pipeline across dumps, so a within-FineWeb
   trend is fair, though FineWeb's filters are not C4's.

**The blocking confound:** a change in the Jewish/Arab sentiment gap between 2019 and 2025 is far more likely due to
world events (October 2023 onwards) than to LLM text. To attribute it to LLM text you need an LLM-share estimate per
dump (Kobak et al. excess vocabulary, Liang et al. 2024 mixture estimator) and a topic control. That is a project of its
own, so put it in "next steps".

## Sources checked in this pass
- UnQover word lists in `reproduction/demographic_bias/third_party/unqover/word_lists/` (gender slots, male/female
  names, no gender-attribute list).
- `huggingface.co/api/datasets/heegyu/bbq`: `Age.jsonl`, `Gender_identity.jsonl` present.
- `huggingface.co/api/datasets/HuggingFaceFW/fineweb`: 110 dumps, `CC-MAIN-2013-20` to `CC-MAIN-2025-26`.
- WIMBD (arXiv:2310.20707) and the Pile (arXiv:2101.00027 §6.4): prior gender sentiment analyses of corpora.
