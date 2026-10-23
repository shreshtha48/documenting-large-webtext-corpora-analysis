# Core findings of Dodge et al. (2021), and how to reproduce each one

Paper: *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*, Dodge, Sap, Marasović, Agnew, Ilharco, Groeneveld, Mitchell, Gardner. arXiv:2104.08758v2 (EMNLP 2021). Local copy: [papers/Documenting-large-webtext-corpora.pdf](papers/Documenting-large-webtext-corpora.pdf).

Every number below was read from that PDF. Where I recomputed something from the paper's own counts, I say so. Section numbers (§) are the paper's.

Status labels used below:
- **Verified**: checked against a primary source (the paper, the original T5 paper, or the code).
- **Discrepancy**: two primary sources disagree.
- **Not verified**: I could not confirm it. It is stated as such.

---

## 0. The setup (§2, Table 1)

C4 comes from the **April 2019 Common Crawl snapshot**. The paper hosts three versions:

| Version | Definition (paper's words, paraphrased) | Docs | Tokens (SpaCy) | Size |
|---|---|---|---|---|
| `C4.EN.NOCLEAN` | Snapshot identified as English, no other filters | 1.1B | 1.4T | 2.3 TB |
| `C4.EN.NOBLOCKLIST` | Same as `C4.EN` but without the bad-words blocklist | 395M | 198B | 380 GB |
| `C4.EN` | All filters applied | 365M | 156B | 305 GB |

### Discrepancy: the filter thresholds

Dodge et al. §2 say C4 removes lines "which don't end in a terminal punctuation mark or have fewer than **three** words" and documents "with less than **five** sentences".

The original T5 paper (Raffel et al., arXiv:1910.10683v4, §2.2) says the opposite: "We discarded any page with fewer than **3 sentences** and only retained lines that contained at least **5 words**."

The code agrees with T5. TensorFlow Datasets (TFDS) [`c4_utils.py`](research-empirical-methods/reference-code/tfds_c4/c4_utils.py) has `_MIN_WORDS_PER_LINE = 5` and `_MIN_NUM_SENTENCES = 3`.

**So Dodge et al. swapped the two numbers.** This is a documentation error. It does not affect the data they analysed. Report it in the reproduction notes.

### Full filter list

Taken from T5 §2.2, and the same as the TFDS code:
- Keep only lines that end in `.`, `?`, `!` or `"`. Lines ending in `...` are dropped.
- Keep only lines with at least 5 words.
- Drop pages that have fewer than 3 sentences after line filtering.
- Drop any page containing a word from LDNOOBW (the "List of Dirty, Naughty, Obscene or Otherwise Bad Words").
- Drop lines containing "javascript".
- Drop pages containing "lorem ipsum".
- Drop pages containing `{`.
- Remove citation markers such as `[1]` and `[citation needed]`.
- Drop lines containing policy and cookie strings.
- Deduplicate any 3-sentence span that occurs more than once.
- Keep only pages that `langdetect` classifies as English with probability at least 0.99.

### Pipeline order (from TFDS `c4.py`)

1. Parse the WET files (Common Crawl's extracted plain text).
2. Length filter.
3. URL dedup.
4. `clean_page`.
5. 3-sentence-span dedup.
6. `langdetect` English check.
7. **Bad-words filter, last, applied to the cleaned text.**

Consequence: `C4.EN` = `C4.EN.NOBLOCKLIST` minus the pages whose cleaned text matches the LDNOOBW regex. **You can rebuild the excluded set from a sample of `noblocklist` alone.**

**Not verified:** which TFDS version AI2 used to produce `allenai/c4`. TFDS later changed things. Version 3.1.0 picks the newest timestamp when deduplicating by URL, and escapes the bad-word regexes. Check a sample before assuming exact equality.

---

## F1. Where the text comes from: metadata (§3)

| Claim | Number in paper | Method |
|---|---|---|
| Top top-level domains (TLDs) by token count | `.com`, `.org`, `.co.uk`, `.net`, … (Fig. 2 left, 25 TLDs) | `tldextract` on URLs; SpaCy tokens |
| `.mil` (US military) | **33,874,654 tokens from 58,394 URLs** | same |
| `.mod.uk` (UK Ministry of Defence) | **1,224,576 tokens from 2,873 URLs** | same |
| Single biggest website | **patents.google.com**; patents.com is in the top 10 | Fig. 2 right, top 25 sites by tokens |
| Other big sites | en.wikipedia.org, en.m.wikipedia.org, nytimes, latimes, theguardian, journals.plos.org, forbes, huffpost, scribd, … | Fig. 2 right |
| Date the text was written (§3.2) | **92%** of 1,000,000 sampled URLs were first indexed between 2011 and 2019; long tail back 10–20 years | Proxy: first Internet Archive capture. Only **~65%** of `C4.EN` URLs are in the Internet Archive |
| Geolocation (§3.3) | **51.3%** of pages hosted in the US; India **3.4%**, Pakistan **0.06%**, Nigeria **0.03%**, Philippines **0.1%** | 175,000 sampled URLs, IP2Location LITE IP-country database |

**Reproducibility:**
- Domains: **easy**. It only needs the URL field and a token count, so it runs on any sample.
- Dates: **hard**. AI2 got the Internet Archive data directly from the Archive (see the paper's acknowledgements). You can query the public Wayback CDX API, but it is slow at scale.
- Geolocation: **doable**. The IP2Location LITE database is free but needs registration. DNS resolution today will differ from 2021, so that is a source of drift to report.

## F2. Machine-generated text in patents (§4.1, Table 3)

patents.google.com machine-translates non-English patents into English. Older patents are OCR'd, and some go through OCR and then machine translation.

Table 3 lists patents in `C4.EN` by patent office. I paired each count with its office by reading word positions in the PDF.

| Office | Count | Required language |
|---|---|---|
| US | 70,489 | English |
| EP | 4,583 | English, French or German |
| JP | 4,554 | Japanese |
| CN | 2,283 | Chinese (Simplified) |
| WO | 2,154 | Various |
| KR | 1,554 | Korean |
| CA | 1,417 | English |
| AU | 982 | English |
| GB | 747 | English |
| DE | 338 | German |
| TW | 332 | Traditional Chinese |
| FR | 271 | French |
| MX | 138 | Spanish |
| SE | 118 | Swedish |
| Other (48 offices, each under 100) | 711 | Various |

**My recomputation:** the total is 90,671. JP+CN+KR+DE+TW+FR+MX+SE = 9,588, which is **10.57%**. That matches the paper's "more than ten percent", which it calls an **upper bound** on translated documents. The US share is 77.7%.

The paper **does not measure** how much of C4 overall is machine-generated. It says "our findings motivate further work".

## F3. Benchmark contamination (§4.2, Table 2 / appendix Table 4)

**Method:** exact match, after normalising capitalisation and punctuation. They explicitly chose this instead of GPT-3's 8–13-gram overlap. The normalisation code is **not public**; see [research-empirical-methods/](research-empirical-methods/).

Every row's percentage in Table 4 equals count ÷ size. I recomputed all of them and they agree.

**Input-and-label contamination** (the target text of the test set appears in C4):

| Dataset | Matched / size | % |
|---|---|---|
| XSum | 1,756 / 11,334 | 15.49 |
| TIFU-short | 19,843 / 79,740 | 24.88 |
| TIFU-long | 790 / 42,139 | 1.87 |
| WikiBio | 2,712 / 72,831 | 3.72 |
| AMR-to-text | 143 / 1,371 | 10.43 |
| LAMA T-REx | 1,585 / 34,014 | 4.6 |
| LAMA Google-RE | 314 / 5,528 | 5.7 |

**Input contamination** (test inputs only, GLUE plus BoolQ):

| Dataset | Matched / size | % |
|---|---|---|
| BoolQ | 79 / 3,245 | 2.4 |
| CoLA | 153 / 1,063 | 14.4 |
| MNLI hypothesis | 1,402 / 9,847 | 14.2 |
| MNLI premise | 1,494 / 9,847 | 15.2 |
| MRPC sentence 1 | 46 / 1,725 | 2.7 |
| MRPC sentence 2 | 46 / 1,725 | 2.7 |
| QNLI sentence | 2,931 / 5,463 | **53.6** |
| QNLI question | 97 / 5,463 | **1.8** |
| RTE sentence 1 | 179 / 3,000 | 6.0 |
| RTE sentence 2 | 325 / 3,000 | 10.8 |
| SST-2 | 200 / 1,821 | 11.0 |
| STS-B sentence 1 | 253 / 1,379 | 18.3 |
| STS-B sentence 2 | 256 / 1,379 | 18.6 |
| WNLI sentence 1 | 7 / 146 | 4.8 |
| WNLI sentence 2 | 3 / 146 | 2.1 |

Two notes on the tables:
- The text-extracted copy of the main-text Table 2 has its rows misaligned. Use appendix Table 4.
- Table 4 lists SST-2 twice. That is a typo in the paper.

**Classification label contamination (appendix A.5):** many GLUE and SuperGLUE test files are on GitHub. The authors searched for phrases from them in the *unfiltered* Common Crawl and checked any matches by hand. They found **no** case of input-and-label contamination.

## F4. Representational bias: Jewish vs Arab (§4.3, A.7, Table 7)

1. **Model probe:** UnifiedQA, a T5 model pre-trained on C4, run as multiple choice on **294,000** UnQover questions about **15 ethnicities** (from Li et al. 2020). Each question names two ethnicities and comes in a positive and a negative version. They measure the share of times each ethnicity is linked with the positive sentiment (Table 7):
   - Highest: Jewish **67.1%**, Asian 60.6%, Caucasian 60.5%, European 60.5%.
   - Lowest: Black 39.3%, **Arab 37.0%**, African 36.6%.
   - **Not stated in the paper:** which UnifiedQA model size was used.
2. **Corpus check:** they found every paragraph in `C4.EN` containing "Jewish" or "Arab", matched with spaces on both sides. They tokenised on whitespace, removed punctuation, and counted sentiment words from a lexicon:
   - **SocialSent lexicon** (the one the paper treats as primary): Jewish has **73.2% positive of 3.4M** sentiment tokens and Arab has **65.7% of 1.2M**. The gap is **7.5 points**.
   - VADER lexicon: the gap is 2.0 points.
   - A small manual lexicon built from the UnQover question words: the gap is 1.4 points.
3. **By domain:**
   - NYT: 74.0% vs 69.5%, a **4.5-point** gap.
   - Al Jazeera: 42.5% vs 42.8%, **no gap**.
   - These domains have far fewer sentiment tokens (1.6k–7.9k).

How the SocialSent lexicon was built:
- Average each word's score over the subreddit lexicons, keeping only words that appear in **at least 40** of them.
- Remove overtly ethnic terms by hand.
- Take the **top 250 positive and top 250 negative** words.

For VADER, a word counts as positive if its score is above 1.0 and negative if below −1.0.

The authors themselves say they "have not shown a causal link".

## F5. What the blocklist removes (§5)

- **Clustering (§5.1, A.6):** they sampled 100,000 excluded documents, built TF-IDF vectors, and ran k-means with k=50. Only **16 of 50 clusters** are mostly sexual, covering **31%** of the excluded documents. Other clusters cover science, medicine, health, legal, political, religion, gaming and infant care.
- **Identity mentions (§5.2, Table 6, Fig. 5):** they matched 22 regex patterns and computed PMI (pointwise mutual information) between each identity mention and the document being filtered. Sexual-orientation terms have the **highest** PMI: lesbian, gay, homosexual, bisexual and heterosexual.
  - In a manual check of 50 sampled documents each, **22%** of the excluded "lesbian" documents and **36%** of the excluded "gay" documents were neither offensive nor sexual.
  - The exact PMI values are only in Fig. 5. The x-axis runs 0–2.5.
- **Dialects (§5.3):** they ran the Blodgett et al. (2016) Twitter dialect model on each document and assigned the dialect with the highest proportion.

  | Dialect | Share removed by the blocklist |
  |---|---|
  | AAE (African American English) | **42%** |
  | Hispanic-aligned | **32%** |
  | WAE (White-aligned English) | **6.2%** |
  | Other | **7.2%** |

  Of the documents in `C4.EN`: **97.8%** are WAE, **0.07%** AAE and **0.09%** Hispanic-aligned.
- **Recommendation:** "We recommend against using blockilst filtering" (sic; the typo is in the paper).

### Construct caveat (verified from the model's own README)

The README of `slanglab/twitteraae` says:
- The outputs are "**not** confidence values about the probability of the author's demographics", but "the proportion of words in the text that come from a demographically-associated language/dialect".
- "we only consider the AA and White categories to be reliable … conducted linguistic validation only for those two."
- The output order is AA, Hispanic, **Asian**, White. The paper relabels Asian as "other".
- `predict()` returns `None` when fewer than 20% of the tokens are in the model's vocabulary. The paper does not say how many documents this affected.

That makes the paper's Hispanic number (32%) rest on a category that Blodgett et al. did not validate. The model was also trained on tweets and is being applied to web pages. **This is a ready-made construct-validity extension.**

## F6. Tokenizer side effect (A.1)

The T5 SentencePiece vocabulary was built from cleaned C4, so it lacks blocklist tokens. For example, "sex" is tokenised as "s" + "ex".

---

## Independent cross-checks from later papers (PDFs read)

| Dodge finding | Later measurement | Agrees? |
|---|---|---|
| AAL is 0.07% of C4.EN | Deas et al. 2025 (arXiv:2503.10789): **280,604 / 365M (0.07%)** in C4.EN; **447,032 / 395M (0.11%)** in noblocklist, using the same Blodgett classifier and argmax | **Yes**, for the share |
| Blocklist removes 42% of AAE documents | From Deas et al.'s counts (my arithmetic): 1 − 280,604/447,032 = **37.2%** | **No.** Gap of about 5 points, not explained by either paper. Good reproduction target |
| Blocklist removes LGBTQ+ identity mentions most | Hong et al. 2024 (arXiv:2405.08209): same regexes, applied to CLIP filtering in DataComp. LGBTQ+ terms are again excluded most | Same pattern, different filter |
| Low toxicity remains after the blocklist | WIMBD (arXiv:2310.20707): 0.01% of C4 toxic by lexicon, **5.75%** by classifier | Lexicon filters miss classifier-detected toxicity |
| Contamination in C4 | WIMBD: different method (tests "whether all input fields are present in a single document"), 82 PromptSource test sets, appendix Table 16 | Not directly comparable |
| Old text in recent crawls (§3.2) | Cheng et al. 2024, "Dated Data" (arXiv:2403.12958): CC dumps contain "non-trivial amounts of old data" | Yes |

## Which findings to reproduce

Here is how each finding scores on cost and on how well it extends:

| Finding | Laptop cost | Data needed | Extension potential |
|---|---|---|---|
| F5 blocklist and dialect | Low: stream about 100k `noblocklist` docs and apply the regex | `allenai/c4` `en.noblocklist` | **High**: construct validity (the dialect model), newer dialect classifiers, newer crawls |
| F3 contamination | Low to medium: hash-based exact match on a shard sample | Shards plus benchmark test sets | **High**: post-2021 benchmarks, safety evals, n-gram vs exact match |
| F1 domains | Low | URL field | Medium: compare against a 2024+ crawl run through the same filters |
| F4 Jewish/Arab | Medium: full-corpus paragraph scan, or a sample | `C4.EN`; lexicons | Medium: lexicon sensitivity is already visible (7.5 vs 2.0 vs 1.4 points) |
| F2 patents | Low | URL field | Low to medium |

My recommendation is to reproduce **F5** as the core result, using a sample. It is cheap, the excluded set can be rebuilt exactly, and it has a documented construct weakness to push on.

**Note on sampling:** each shard is about 1/1024 of the corpus. 1000 shard files are listed for `en` and `en.noblocklist` on Hugging Face; the listing may be paginated. A sample-based reproduction should report confidence intervals, not point estimates, as the trial brief requires.
