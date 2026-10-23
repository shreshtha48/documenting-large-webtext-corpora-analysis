# Empirical methods dictionary: C4 audit (Dodge et al. 2021)

This file lists, for each analysis in the paper:
- **what the paper says it did**, quoted from or pointing to the section;
- **the code or tool that implements it**, and whether that code is official or a later re-implementation;
- **the exact formula or parameters**;
- **what is not specified**, so you know where you will be making assumptions.

How to read the tags:
- **[PAPER]**: the paper itself.
- **[CODE]**: a file in [reference-code/](reference-code/) that I downloaded and read.
- **[VERIFIED]**: I ran it or checked it against a primary source.
- **[UNSPECIFIED]**: the paper does not say. Any choice here is ours and must be reported as an assumption.

---

## 0. Is there official code?

| Source | What it contains | Verified how |
|---|---|---|
| `github.com/allenai/c4-documentation` | **README only.** 6 commits, all 13–18 Apr 2021. Its discussions page has 2 threads, both from 2023 and both user questions. **No analysis code was ever released.** | GitHub API: commits, contents, discussions |
| `huggingface.co/datasets/allenai/c4` | The data: `en`, `en.noclean`, `en.noblocklist`, `realnewslike`, `multilingual`. Commit history: "Adds the en.withbadwords dataset" (2021-04-16), then "We can't say badwords" (2021-04-17), then "Added docs for the `noblocklist` data" (2021-06-15). This reads like a rename, but that is my inference from the commit titles. | HF API: tree, commits, dataset card |
| AI2 download thread (`allenai/allennlp` discussion #5056, 2021-03-16) | States the data was produced with **TFDS C4 version 3.0.1**. The bucket path is `gs://allennlp-tensorflow-datasets/c4/en/3.0.1/`. | GitHub GraphQL |
| `tensorflow/datasets` C4 builder | **The filter code that produced C4.** Saved here twice: v3.0.1 (commit `f411fae`, 2021-01-05, the version AI2 used) and v3.1.0 (current). | [reference-code/tfds_c4/](reference-code/tfds_c4/) |
| `allenai/wimbd` (Elazar et al. 2023, "What's In My Big Data?", arXiv:2310.20707) | AI2's **later** toolkit. Dodge is a co-author. It re-implements domain counts, date extraction, sentiment co-occurrence, profanity, contamination and PII for C4 and other corpora. **It is not the 2021 code, and its methods differ** (see each section). | [reference-code/wimbd/](reference-code/wimbd/), pinned to commit `18fbc62` |
| AI2 Elasticsearch index over all three C4 subsets | Documented in WIMBD `ELASTIC.md`. Access needs keys from a form in the WIMBD README. Jesse Dodge's personal GitHub page has a read-only key posted publicly. **I did not test it, and I have not copied it here.** | Read the docs only |
| **Infini-gram API**, index `v4_c4train_llama` | Exact-match counts over **all of C4.EN train**: 364,868,892 docs, 198,079,554,945 Llama-2 tokens. Free, no key needed. | **[VERIFIED]** I ran a query (see §3) |

**Bottom line:** no one has published a reproduction repo for this paper. I searched GitHub repo names and code for `en.noblocklist`, `c4-search.apps.allenai.org`, the paper title and `twitteraae c4`. Hits only used C4 for training, or were WIMBD. **Your reproduction would be the first public one I can find.**

---

## 1. Building C4, and the blocklist-excluded set (F5 prerequisite)

**[CODE]** [c4_utils_v3.0.1_f411fae.py](reference-code/tfds_c4/c4_utils_v3.0.1_f411fae.py), [c4_v3.0.1_f411fae.py](reference-code/tfds_c4/c4_v3.0.1_f411fae.py)

**Pipeline order** (v3.0.1, `_get_page_content`):
1. Split WET files.
2. `is_valid_length`: page length at most 1.9e5 characters.
3. Normalise URL and dedupe by URL.
4. `clean_page` (en only).
5. `remove_duplicate_text`: dedupe 3-sentence spans.
6. `detect_english` (langdetect, `min_probability=0.99`, seed 0).
7. `badwords_filter`.

`clean_page` constants **[CODE]**:
```
_MIN_WORDS_PER_LINE = 5      # paper (Dodge) says 3: wrong, see findings.md
_MIN_NUM_SENTENCES  = 3      # paper (Dodge) says 5: wrong
_MAX_WORD_LENGTH    = 1000   # lines with a longer "word" are dropped
_END_MARKS = (".", "?", "!", '"');  lines ending "..." dropped
citation_regex = r"\[\d*\]|\[edit\]|\[citation needed\]"
drop PAGE if: "lorem ipsum" in line.lower(); "{" in line
drop LINE if: "javascript" in line.lower(); any policy substring
sentence count = nltk punkt sentence tokenizer over kept lines
```

**Bad-words filter (exact)** **[CODE]**:
```python
words = open("ldnoobw_en_25e679f.txt").read().splitlines()   # LDNOOBW commit 25e679f…, 377 lines
regex = re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))   # v3.0.1: NOT re.escape'd
excluded = regex.search(cleaned_text.lower()) is not None
```
- The list is saved at [reference-code/tfds_c4/ldnoobw_en_25e679f.txt](reference-code/tfds_c4/ldnoobw_en_25e679f.txt).
- **[VERIFIED]** No entry contains a regex metacharacter, so the v3.1.0 change ("escape badword regexes") makes no difference for English.
- The match runs on **cleaned** text, after line filtering, because the filter is the last step.

**So the excluded set is:** take documents from `en.noblocklist` and apply the regex. The ones that match are exactly the documents removed by the blocklist. You do not need to diff `en` against `noblocklist`.

**Sanity check to run first:** stream shard `c4-train.00000-of-01024` of both `en` and `en.noblocklist`. Then confirm:
- `noblocklist` docs minus regex matches ≈ the `en` docs, matched by URL;
- 365M / 395M ≈ 92.4% survive, so about 7.6% should match.

**[UNSPECIFIED]** Whether AI2's dedupe (step 5) ran before or after the blocklist would change the documents next to a removed one. The code says before, so the result should be exact. Check it on one shard.

---

## 2. Metadata: domains, dates, geolocation (F1)

### 2a. TLDs and websites

**[PAPER] §3.1:** tokens per TLD and per website, using the SpaCy English tokenizer. TLDs are parsed with `tldextract`.

**[CODE] WIMBD:** [high_level_domain_counts/map_count.py](reference-code/wimbd/wimbd/high_level_domain_counts/map_count.py) uses `tldextract.extract(url).suffix`. [url_per_tok_counts/map_count.py](reference-code/wimbd/wimbd/url_per_tok_counts/map_count.py) uses `urlparse(url).netloc` plus `uniseg` word-break counts.

**Difference to report:** WIMBD counts words with `uniseg`, but the paper used SpaCy. The rankings should agree; absolute counts will not.

**[UNSPECIFIED]** Whether "website" means `netloc` including subdomains. Fig. 2 lists `en.wikipedia.org` and `en.m.wikipedia.org` separately, which implies full netloc.

### 2b. Utterance date

**[PAPER] §3.2:** the first date the Internet Archive indexed the URL, for 1,000,000 sampled URLs. The data came privately from the Internet Archive.

**Public substitute [VERIFIED]:** the Wayback CDX API returns captures in ascending time order, so the first row is the earliest. Example query:
```
https://web.archive.org/cdx/search/cdx?url=<url>&limit=1&output=json&fl=timestamp,original
```
It is slow, so use about 1–5k URLs and bootstrap a confidence interval.

**Not the same thing:** WIMBD [utterance_dates/map_count.py](reference-code/wimbd/wimbd/utterance_dates/map_count.py) reads the C4 `timestamp` field. That is the **crawl** time, not when the page was first archived.

### 2c. Geolocation

**[PAPER] §3.3:** IP2Location LITE IP-country database, 175,000 URLs.

**[UNSPECIFIED]** When and how hostnames were resolved to IPs. Resolving DNS today will drift from 2021.

**Not verified:** the IP2Location site (`lite.ip2location.com`) timed out when I tried it.

---

## 3. Benchmark contamination (F3)

**[PAPER] §4.2:** "we measure exact matches, normalized for capitalization and punctuation."

**[UNSPECIFIED]** All of the following:
- the normalisation function;
- whether a match means the example is a **substring** of a document or equal to a whole line;
- which field was matched for each dataset;
- which split ("test sets"; for GLUE the test inputs).

Later papers describe this method only in the same words. For example, Li et al. 2023 (arXiv:2310.17589) say "detect exact matches … after normalising for capitalisation and punctuation". None of them give code. A 2026 paper (Braun, arXiv:2608.12652) notes that the GPT-3 and C4 analyses "use different n-gram lengths and normalisation and are not directly comparable."

**Proposed normalisation.** This is **ours**, and must be stated as an assumption:
```python
import re, unicodedata
def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)      # strip punctuation
    return re.sub(r"\s+", " ", s).strip()
contaminated = norm(example) in norm(document)   # substring match
```

**Three ways to run it:**

1. **Infini-gram API, full corpus, free [VERIFIED]:**
   ```
   POST https://api.infini-gram.io/
   {"index":"v4_c4train_llama","query_type":"count","query":"<text>"}
   ```
   It matches tokens exactly and is **case- and punctuation-sensitive**. I tested the paper's Table 5 XSum summary "Grimsby fought back … play-off hopes.":
   - original casing: `count = 1`;
   - lowercased with the final period removed: `count = 0`.

   So the API gives an **un-normalised lower bound**. Querying several casings of each example narrows the gap; it cannot close it.
2. **Stream shards and use Aho-Corasick** (`pyahocorasick`) over the normalised text. This is exact to the paper's stated method, but a **k-shard sample only gives a lower bound** on the full-corpus rate. The rate is *not* scale-invariant.
3. **WIMBD Elasticsearch** `match_phrase`. The standard analyser lowercases and drops punctuation, which is effectively the paper's normalisation. This needs access keys. Code: [es/__init__.py](reference-code/wimbd/wimbd/es/__init__.py) `count_documents_for_each_phrase`, and [es/corpus_contamination.py](reference-code/wimbd/wimbd/es/corpus_contamination.py), where `percentage = count(phrases with ≥1 hit) / n`.

   It is plausible that the original analysis ran on AI2's own index, given the paper hosts a search index, but **that is my inference, not stated**.

**Datasets and fields** (from paper Table 4, appendix A.4):
- Generation targets: XSum summaries; TIFU-short and TIFU-long (Reddit TIFU summaries); WikiBio; AMR-to-text (LDC2017T10; **licensed, not freely available**).
- LAMA: T-REx and Google-RE.
- GLUE test inputs: CoLA, MNLI hypothesis and premise, MRPC sentence 1 and 2, QNLI sentence and question, RTE, SST-2, STS-B, WNLI.
- BoolQ.

The denominators in Table 4 are checkable. For example, XSum is 11,334, which matches the XSum test split size.

---

## 4. Identity mentions vs the blocklist (F5, §5.2)

**[PAPER]** 22 regexes (Table 6) and the PMI "between the likelihood of an identity mention occurring versus being filtered out by the blocklist".

The regexes as printed in Table 6, read from the PDF:
```
homosexuals?  gays?  non[ -]?binary  trans(|\+|gender)  lesbians?  blacks?
african[ -]americans?  latin[oax]s?  asian([ -]american)?s?  muslims?  jew(|s|ish)?
wom[ae]n  females?  m[ae]n  males?  straights?  heterosexuals?  bi-?sexuals?
whites?  caucasians?  european([ -]american)?s?  christians?
```

**Formula** (standard PMI, Church & Hanks 1990, at the document level):
```
PMI(id; filt) = log [ P(id ∧ filt) / ( P(id) · P(filt) ) ]
  computed over D = en.noblocklist documents
  id   = regex matches the document
  filt = document matched the blocklist
```

**[UNSPECIFIED]**:
- log base;
- case-insensitive matching (almost certainly yes);
- word boundaries around the regex;
- document vs token unit;
- smoothing.

Fig. 5's x-axis runs 0 to 2.5. If you use the natural log, PMI = 2.5 means the co-occurrence is about 12× what independence would predict. Report your choice.

**Uncertainty:** bootstrap over documents, or over shards, which is safer because of within-domain clustering.

Later papers that reuse this method:
- Hong et al. 2024 (arXiv:2405.08209) state they follow "the same methodology as Dodge et al." for identity keyword mentions.
- The LAION-Aesthetics audit ("The Algorithmic Gaze") says: "Similar to Dodge et al.'s prior audits of generative AI dataset filtering [20], we computed the pointwise mutual information (PMI) [16] of a regular expression (regex) appearing in the 6.5+ subset of images in LAD." (Quoted from the citing sentence Semantic Scholar returned; I did not read the PDF.)
- Sound Check (arXiv:2410.13114) reuses the identity keywords.

---

## 5. Dialect classification (F5, §5.3)

**[CODE]** [reference-code/twitteraae/predict.py](reference-code/twitteraae/predict.py), from `slanglab/twitteraae`, last pushed 2019-11-09. **Python 2.7.** To port it:
- `xrange` → `range`;
- `unicode` → `str`;
- remove `.decode("utf-8")`.

The model files (`model/model_count_table.txt`, 4.1 MB, and `model/model_vocab.txt`, 2.3 MB) are in that repo. I did not copy them here; fetch them from the repo.

**Algorithm:** collapsed variational Bayes (CVB0) over 4 topics, `alpha=1`, `numpasses=5`. It returns `None` if fewer than 20% of tokens are in the vocabulary (`thresh2=0.2`), or if no tokens are (`thresh1=1`).

**Output order:** [African-American, Hispanic, **Asian**, White]. Dodge et al. call the third one "other".

**Assignment:** argmax over the 4 proportions, as the paper says.

**[UNSPECIFIED]**:
- tokenisation (the repo used `twokenize.py`, built for tweets);
- whether the whole document or each paragraph was scored;
- how `None` outputs were handled.

**Construct warning (from the repo README):** the outputs are "not confidence values … rather … the proportion of words … from a demographically-associated language/dialect", and only the AA and White categories were validated.

**A later replication with code exists:** Deas et al. 2025, "Data Caricatures" (arXiv:2503.10789), use this same classifier across 12 corpora. See [../paper-ideas-extension.md](../paper-ideas-extension.md).

---

## 6. Ethnicity sentiment co-occurrence (F4, A.7)

**[PAPER]** procedure:
1. Find paragraphs in C4.EN containing " Jewish " or " Arab " (the word surrounded by spaces), using `fgrep`.
2. Split on whitespace, remove all punctuation, and count co-occurring words.
3. For each lexicon, compute positivity = pos / (pos + neg) over sentiment-bearing tokens in those paragraphs.

Lexicons:
- **SocialSent (primary):** average each word's score across the subreddit lexicons, keeping words present in **at least 40** of them. Remove ethnic terms by hand. Take the **top 250** positive and **top 250** negative words.
- **VADER:** positive if the score is above 1.0, negative if below −1.0.
- **Manual lexicon:** built from the UnQover question words. **Not released.**

Results: SocialSent 73.2% vs 65.7%, a gap of 7.5; VADER gap 2.0; manual gap 1.4.

Code and data:
- `williamleif/socialsent` holds the induction code. The **250 subreddit lexicons are on the project site** (`nlp.stanford.edu/projects/socialsent`), which **timed out when I tried it, so they are not verified downloadable today**.
- `cjhutto/vaderSentiment` contains `vader_lexicon.txt`, where the mean score is the second column.

**[UNSPECIFIED]**:
- the exact list of 250 + 250 words;
- which words were removed by hand;
- the paragraph delimiter (presumably `\n` in C4 JSON);
- whether "Jewish" matching was case-sensitive (`fgrep` is, by default).

**Different method in WIMBD:** [sentiment_coocurrence/map_count.py](reference-code/wimbd/wimbd/sentiment_coocurrence/map_count.py) scores **sentences** with TextBlob polarity, not a lexicon count over paragraphs. Its numbers are not comparable to F4.

**UnifiedQA probe:**
- `allenai/unqover` has the question templates and the generation scripts, `scripts/generate_questions.sh`.
- Models on Hugging Face: `allenai/unifiedqa-t5-{small,base,large,3b,11b}` **[VERIFIED: they exist]**.
- **[UNSPECIFIED]** which size the paper used.

---

## 7. Clustering of excluded documents (F5, §5.1, A.6)

**[PAPER]** 100,000 sampled excluded documents → TF-IDF → k-means with k=50 → tSNE/PCA projection for plotting. Clusters were labelled "largely sexual" **by hand**: 16 of 50, covering 31% of documents.

**[UNSPECIFIED]**:
- TF-IDF settings (vocabulary size, stop words, min_df);
- the k-means seed;
- the labelling rubric.

**Suggestion:** use scikit-learn `TfidfVectorizer(max_features=50k, stop_words="english")` and `KMeans(n_clusters=50, n_init=10, random_state=s)` over several seeds. Report how stable "16/50" is across seeds. That is a cheap robustness check the paper did not do.

---

## 8. Later tools and methods worth reusing (all repos verified to exist)

| Need | Tool / method | Source | Caveat |
|---|---|---|---|
| C4 filters on a new crawl | TFDS `clean_page` v3.0.1 (saved here) | Google | Use these files. **Do not** use datatrove `C4QualityFilter` defaults: `min_num_sentences=5, min_words_per_line=3` is the reverse of TFDS and T5. If you use datatrove, pass `min_num_sentences=3, min_words_per_line=5`. |
| AAL detection across corpora | `NickDeas/DataCaricatures` | Deas et al. 2025 | Uses argmax AAL, and also posterior > 0.3. Corpus sampling: SHA-256 of file URL, take files in hash order up to 250 GB. |
| langID / quality filters vs geography | `lucy3/whos_filtered` | Lucy et al. 2024 | Includes a reproduction of C4's langdetect (cutoff 0.99) built on Dolma's toolkit. |
| Identity regexes on another filter | `hongrachel/clip-filtering-bias` | Hong et al. 2024 | Reuses Dodge Table 6 regexes verbatim (their Table 1). |
| GPT-3-style quality filter | `kernelmachine/quality-filter` | Gururangan et al. 2022 | To run the same identity analysis on a different filter. |
| Paraphrase-robust contamination | `lm-sys/llm-decontaminator` | Yang et al. 2023 | LLM-judged, so it costs API calls. |
| Fuzzy search at trillion-token scale | `softmatcha/softmatcha2` | 2026 | Has an online demo. |
| Exact counts over full C4 | Infini-gram API `v4_c4train_llama` | Liu et al. 2024 | Case- and punctuation-sensitive (§3). |
| Contamination over 82 datasets | WIMBD `es/corpus_contamination.py` | Elazar et al. 2024 | Tests "whether all input fields are present in a single document"; the authors call this an "upper bound of exact-match dataset contamination". |
| Robots.txt / ToS over time for C4 domains | `Data-Provenance-Initiative/Data-Provenance-Collection` | Longpre et al. 2024 | Uses the Wayback Machine. |

## 9. Are there dated snapshots of C4? (verified)

**Short answer: no.** There is only one C4, built from `CC-MAIN-2019-18` (April 2019).
- `allenai/c4` variants (`en`, `noclean`, `noblocklist`, `realnewslike`) are **filter variants of that same crawl**, not dates.
- TFDS versions 2.2.0–3.1.0 are **code** versions, not time snapshots.

Ways to get a "then vs now" comparison:

| Option | What it gives | Verified |
|---|---|---|
| TFDS C4 builder with `cc_versions=[...]` | The **exact C4 recipe applied to any Common Crawl dump**. `ALL_CC_VERSIONS` in the code lists dumps 2013-20 → 2022-21. The URL template `crawl-data/CC-MAIN-{cc_version}/wet.paths.gz` suggests newer dumps would work too, but **I have not tested a post-2022 dump.** A full build needs Apache Beam and a lot of compute, so for a laptop, run `clean_page` and the blocklist on a handful of WET files instead. | Code read |
| `HuggingFaceFW/fineweb` per-dump configs | 110 dump configs, `CC-MAIN-2013-20` … `CC-MAIN-2025-26`, plus samples. **Different pipeline:** Trafilatura extraction from WARC (not WET), fastText langID (not langdetect), and C4 filters minus terminal punctuation. A proxy for the web over time, **not** for C4 over time. | HF datasets-server API (114 configs) |
| "Practical Datasets for Analyzing LLM Corpora Derived from Common Crawl" (ICWSM 2025) | Domain statistics across **96 CC snapshots**, plus URL lists for C4, RefinedWeb and CulturaX | Abstract only. **I have not located the download.** |
| Infini-gram indices | Exact counts in later corpora: Dolma, DCLM-baseline, RedPajama, Pile, OLMo mixes. Lets the same query run against C4 (2019) and newer corpora. | API docs read; C4 index queried |
| mC4 (`allenai/c4` `multilingual`) | Built from many dumps (TFDS config description: "generated from 86 Common Crawl dumps"), but it is non-English-centric and uses different filters | Code read |

## 10. Uncertainty: what to report everywhere

The paper reports **no error bars**. Use these defaults:
- **Proportions** (contamination %, removal % per dialect): Wilson 95% CI. Use `statsmodels.stats.proportion.proportion_confint(k, n, method="wilson")`.
- **Comparing two proportions** (AAE removal vs WAE removal): two-proportion z-test, or bootstrap the difference.
- **Corpus statistics** (PMI, positivity gaps): bootstrap with resampling by **shard or domain**, not by document. C4 documents are clustered by site, so a document-level bootstrap gives intervals that are too narrow.
- **Sampling scale:** each HF shard is about 1/1024 of the corpus. For rare categories such as AAE, which is 0.07% of C4.EN, you need many shards. At 356k docs per shard and 0.07%, expect about 250 AAE-argmax documents per shard.
