# Experiment log: gender / age extension

Every run, including smoke tests and dead ends, with what it showed.

## X0. Can C4 timestamps give a time axis? (2026-10-24, local)

- **Run:** streamed the first 50,000 documents of `en/c4-train.00000-of-01024` and counted the months in `timestamp`.
- **Result:** 50,000 / 50,000 are 2019-04. The timestamp is the crawl date.
- **Decision:** picking shards by timestamp is not possible. We use the URL year as a proxy instead (19% coverage on
  20,000 docs).

## X1. E6 split by template gender (local, CPU, 14 s)

- **Run:** `analyze_probe.py --e6` on the reproduction's UnifiedQA-3B scores (147,000 contexts).
- **Result:**
  - Same ranking for "man" and "woman" templates (rank correlation 1.0); every group moves by 2.4 points or less.
  - Jewish − Arab gap: 64.9 (men) vs 65.7 (women).
  - Middle-Eastern moves the most (+2.4 [1.6, 3.2] in woman templates).
- **Output:** `results/E6_gender_split/`.

## X2. Probe pipeline smoke test (local, CPU, UnifiedQA-small)

- **Run:** 1-in-40 stride of gender_nouns / gender_names / gender_age.
- **Result:** the pipeline runs end to end.
- **Bug found:** a stride of 40 with 50 attributes samples only 5 attributes (the reproduction's E2 trap), so
  "without gendered attributes" equalled "all".
  - It does not affect full runs.
  - For any sampled check, use a stride coprime with 50 and 7 (e.g. 97).
- Small-model numbers are not results (E4 showed small does not reproduce).

## X3. Corpus scan smoke test (local, 20,000 docs of noblocklist shard 0)

- **First version:** one big case-insensitive regex alternation took 210 s per 20k docs (about 1 h per shard), 55%
  of it in matching.
- **Fix:** word-sequence lookup, skipping words that can't start a term. 53 s per 20k docs with dialect, 38 s
  without, so about 17 min per full shard.
  - Checked that matching does not cross punctuation ("black, women" is not "Black women").
- **Found:** LDNOOBW entries that contain term words (`girl on`, `big black`, `sultry women`, ...).
  - Added the non-mechanical exclusion count.
- **Found:** SocialSent's "man" (+2.1). Term words are now removed before counting.
- 20k docs are too few for results: the Jewish/Arab control is −13.9 [−30.8, 1.4] with only ~100 paragraphs.

## X4. Notebook dry run (local)

- **Run:** extracted every `%%writefile` cell of `ga_probe_account1` and `ga_corpus` into the Kaggle layout, then
  ran generate → score (small, 64 contexts) → analyze, and scan (1,500 docs) → analyze.
- **Result:** all steps ran without errors.

## X5. Question format checked against the sources; generator rebuilt on UnQover (2026-10-24)

- **Dodge §4.3:** "we used their script", ethnicity only.
- **UnQover:** gender = names × occupations (no positive / negative score); 4 mixed-gender templates
  (`gender_noact`).
- **BBQ:** the published benchmark for gender, age and race × gender. It evaluated UnifiedQA-11B with RACE-style
  input and is scored with its own bias score.
- **Change:** the first generator used our own templates. It now runs UnQover's script unchanged (vendored files
  byte-identical to upstream @ 3e47969) with added subject lists and two slot files. The only template edits are
  `[p_act]` → `[p_act-single]` and an article before nouns.
- **Lists changed:**
  - gender × ethnicity is now same-race cross-gender for all 15 Table 7 groups; race-within-gender comes from E6;
  - the name control is its own list.
- **Total:** 78,800 contexts (was 158,800).

## X6. BBQ metric code checked against the paper (local, CPU)

- **Run:** `analyze_bbq.py --published_11b` on the authors' UnifiedQA-11B predictions for all 11 categories.
- **First reading:** it looked wrong (Age 29.8 vs 11.8). Cause: in the PDF's text layer Figure 3's columns follow
  the plot's alphabetical model order, so UnifiedQA (RACE) is the 4th column, not the 6th.
- **Against the right column:**
  - Age, Disability, Nationality, Religion, Sexual orientation and SES match exactly (ambiguous and disambiguated).
  - Physical appearance is within 0.7 (ambiguous) / 2.0 (disambiguated).
  - Race label / names rows match with the two rows swapped.
  - Gender identity does not match (ours 14.9 / 33.0 vs 32.3 / 2.4). Likely a data-version difference: BBQ's R
    script says versions differ by 16 examples, and 16 Gender_identity examples lack metadata.
- **Found:**
  - 344 Race_x_gender items have an empty group label (BBQ defect); they are excluded from the group analyses only.
  - BBQ's same-gender, different-race items are mostly female: men appear in only 1–6 templates per race.
  - A template bootstrap with 1–2 templates is degenerate (a 50–50 "CI" for Caucasian was flagged as an anomaly).
    Groups below 5 templates or 50 trials are no longer tested.
- **11B result:** female positivity against a same-race man is 59.9 [46.5, 71.3]; Arab is the anomaly (lower,
  Holm p < .05).

## X7. Time experiment put on hold

- C4 has no dated snapshots (X0), so it is on hold.
- The URL year is still computed by the scan, but it is not part of the reported results.

## Not yet run (left for the user)

- Probe on the new lists, and BBQ on UnifiedQA-3B (Kaggle).
- Re-running the E6 split with the new anomaly columns: `analyze_probe.py --e6`.
- Corpus scan with the race-zoom tables.

## Kaggle runs (2026-10-24)

| ID | Notebook | What | Outcome |
|---|---|---|---|
| P1a | ga_probe_account1 | UnifiedQA-3B, gender_names (+ control) + gender_nouns | Done (`ga_probe_account1.zip`). Women picked for the positive attribute 62.0% (names; same-gender control 50.0) and 65.9% (nouns) |
| P1b | ga_probe_account2 | UnifiedQA-3B, age + gender_age + gender_ethnicity | Done (`ga_probe_account2.zip`). Women 69.5% (with age), 70.1% (with ethnicity). Age: teens 35.4%, young adults 47.2%, elderly 51.4%, middle-aged 66.0%. Among women, African-American (+6.2) and African (+4.5) sit above the average woman, White (−5.1) and Jewish (−3.1) below (Holm p < .05) |
| P2 | ga_probe_account2 | BBQ on UnifiedQA-3B | Incomplete: scoring stopped at 8,000 / 12,648 items (`logs_bbq_*.txt`); no analysis. Partial scores kept in the zip |
| C1 | ga_corpus | 12 noblocklist shards: exclusion, dialect × gender, sentiment, URL year, classifier | Not run |

Figures and tables: `../results/gender_age/` (from `../extension_results.ipynb`).
