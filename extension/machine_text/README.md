# Machine-Generated Text & C4 Filtering Extension

This module provides tools to stream C4 / Common Crawl shards, apply the exact C4 cleaning hierarchy from TensorFlow Datasets (TFDS v3.0.1, commit `f411fae`), and partition web documents into:
1. **`C4.EN`**: Documents that satisfy all heuristic quality filters and contain no terms from the LDNOOBW blocklist.
2. **`C4.BLOCKLIST`**: Documents that satisfy quality/length/language heuristics but are rejected solely by the LDNOOBW bad-words blocklist.
3. **Rejected Heuristics** (in raw mode): Documents dropped due to terminal punctuation, minimum words, curly brackets, lorem ipsum, sentence count, or language detection.

## Important Context on C4 Shards & Modern Crawls

- **C4 Timing**: The official `allenai/c4` dataset on Hugging Face was derived exclusively from the **April 2019 Common Crawl snapshot** (`CC-MAIN-2019-18`). Shards `00000` through `01023` are arbitrary partitions of that single 2019 crawl—higher shard numbers are **not** later in time.
- **Machine-Generated Text in 2019 C4**: As shown in Dodge et al. §4.1, machine-generated text in 2019 C4 is concentrated in patents (`patents.google.com` machine-translated non-English patents), OCR documents, and automated content scrapers.
- **Modern Post-LLM Crawls (2023–2025)**: To evaluate modern LLM-generated web text, the same pipeline can be applied to newer Common Crawl WET dumps or modern web snapshots (e.g. FineWeb, Dolma) using `--source noclean` / `--source local`.

## Filter Hierarchy (TFDS v3.0.1 `c4.py`, same order)

1. **WET split + length**: page text $\le 1.9 	imes 10^5$ characters.
2. **URL dedupe**: URLs normalised (`http(s)://`, `www.`, `?utm_/ref/feed…`, trailing `/` removed); one page per URL (max md5 of text).
3. **`clean_page`** (per line, in this order):
   - drop line if any word is > 1000 characters;
   - strip citation markers `[1]`, `[citation needed]`, `[edit]`;
   - drop line unless it ends with `.`, `?`, `!` or `"`; drop if it ends with `...`;
   - drop line if it has < 5 words;
   - drop **page** if `"lorem ipsum"`;
   - drop line if `"javascript"` (checked *before* the `{` rule);
   - drop **page** if `"{"`;
   - drop line if a cookie / policy phrase;
   - drop page if < 3 sentences (NLTK punkt, summed over kept lines).
4. **Line dedupe** (`remove_duplicate_text`): each line (md5 of `strip().lower()`) is kept only in the page with the smallest md5(URL); repeats inside a page are dropped; the page is dropped if it ends up empty or with < 3 sentences.
5. **Language**: `langdetect` top prediction `en` with probability $\ge 0.99$ (seed 0).
6. **Badwords**: the 377 LDNOOBW terms (commit `25e679f`), regex `(?:\W|^)(…)(?:\W|$)` on lower-cased text.
   - Match $\implies$ `C4.BLOCKLIST`
   - Clean $\implies$ `C4.EN`
   - Steps 1–5 alone give `en.noblocklist`.

**Sample-size caveat:** steps 2 and 4 are global in TFDS (the whole crawl). Here they only see the pages in one run, so a one-shard sample keeps many boilerplate lines that real C4 would have removed.

**Verification:** [scripts/test_against_tfds.py](scripts/test_against_tfds.py) runs the TFDS v3.0.1 reference code (`research-empirical-methods/reference-code/tfds_c4/`) and this pipeline on the same WET pages. On 5,000 pages of CC-MAIN-2024-10 shard 0, C4.EN (998 docs) and C4.BLOCKLIST (79 docs) were identical in URL and text.

**Tokenizer note:** TFDS loaded `punkt/english.pickle`. NLTK $\ge$ 3.8.2 can no longer load pickles and ships the same model as `punkt_tab`, which is what this uses.

## Files

| Path | What |
|---|---|
| [scripts/filter_c4_pipeline.py](scripts/filter_c4_pipeline.py) | The C4 filters (TFDS v3.0.1) for one WET shard or a local file |
| [scripts/run_wet_shards.py](scripts/run_wet_shards.py) | Runs the filters on many WET shards, several in parallel and resumably. The shards are seeded random draws over each crawl. Used by [../post_llm_bias](../post_llm_bias) |
| [scripts/test_against_tfds.py](scripts/test_against_tfds.py) | Page-by-page comparison against TFDS v3.0.1 |
| [scripts/machine_text.py](scripts/machine_text.py) | Machine-generated text analysis (`docs` → `detect` → `classify` → `analyze`) |
| [kaggle/c4_filter_pipeline.ipynb](kaggle/c4_filter_pipeline.ipynb) | Filters one shard on Kaggle / Colab |
| [kaggle/machine_text_4shards.ipynb](kaggle/machine_text_4shards.ipynb) | **Machine text in the 4 pilot shards** (GPU T4 ×2) |

Runs are logged in [EXPERIMENTS.md](EXPERIMENTS.md).

## Machine-text analysis (4 pilot shards)

Corpora: CC-MAIN-2024-10 shards 0 and 3, CC-MAIN-2024-51 shards 0 and 3 (C4.EN + C4.BLOCKLIST, in
`../post-llm-crawl/`), plus 2019 C4 shard 00000 as the **negative control**. 2019 predates ChatGPT, so whatever the
detector flags there is a false positive. Only documents with ≥ 100 words are used.

1. **Marker words**: the share of documents whose first 300 words contain a pre-registered LLM style word, compared with control words. This gives a lower bound on the LLM-processed share, f ≥ (p − q)/(1 − q).
2. **Binoculars** (zero-shot, GPU): the thresholds are the 1% and 5% lowest scores on 2019 C4. The flag rate per ethnicity group on 2019 shows whether the detector is biased against some groups' text.
3. **Tone**: lexicon and RoBERTa positivity of flagged vs unflagged documents, and of the group paragraphs inside them.

CIs come from a web-domain bootstrap. With 4 shards the group-level CIs are wide. That is stated, not hidden, and more shards are listed as future work.
