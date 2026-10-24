"""C4 Filtering Pipeline & Blocklist Extraction for Modern Post-LLM Web Crawls (Kaggle / Colab / Local).

Re-implements the C4 pipeline from TFDS v3.0.1 (commit f411fae), in the same order as c4.py:
  1. split_wet_file, then is_valid_length (<= 1.9e5 characters)
  2. normalize_url + dedupe_urls (one page per normalised URL; keeps max md5(text))
  3. clean_page:
     - drop line if a word is > 1000 chars
     - strip citation markers: [1], [citation needed], [edit]
     - drop line unless it ends with ('.', '?', '!', '"'); drop if it ends with '...'
     - drop line if < 5 words
     - drop PAGE if 'lorem ipsum' (case-insensitive)
     - drop line if 'javascript' (checked BEFORE the '{' rule, as in TFDS)
     - drop PAGE if '{'
     - drop line if a cookie / policy substring
     - drop page if < 3 sentences (NLTK punkt, summed over kept lines)
  4. remove_duplicate_text: drop every line (md5 of strip().lower()) that occurs in another
     page, keeping it only in the page with the smallest md5(url); drop repeats within a page;
     drop the page if it becomes empty or has < 3 punkt sentences.
  5. detect_english: langdetect, seed 0, top prediction 'en' with prob >= 0.99
  6. badwords: 377 LDNOOBW terms (commit 25e679f), regex (?:\\W|^)(...)(?:\\W|$) on lower-cased text
     -> match = C4.BLOCKLIST, no match = C4.EN  (steps 1-5 alone = en.noblocklist)

Steps 2 and 4 are global in TFDS (whole crawl). Here they run over the pages read in one run,
so a small sample deduplicates far less than the real C4 did.

Supported Data Sources:
  --source commoncrawl : Downloads a real post-LLM WET shard from Common Crawl
                         (e.g., CC-MAIN-2024-10, CC-MAIN-2024-51, CC-MAIN-2023-50).
  --source noblocklist : allenai/c4 'en.noblocklist' (2019, already steps 1-5), splits by badwords only.
  --source noclean     : allenai/c4 'en.noclean' (2019, langdetect on raw text only), runs steps 2-6.
  --source local       : Reads a local .warc.wet.gz or .json(l)(.gz) file.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Dict, Generator, List, Optional, Tuple

# Default LDNOOBW list URL (commit 25e679f used in Dodge et al. and TFDS v3.0.1)
LDNOOBW_URL = (
    "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/"
    "25e679f03d96baa721cde20db9944649e8d0a844/en"
)

# Common Crawl Base URL
CC_BASE_URL = "https://data.commoncrawl.org/"

# C4 Filter constants (TFDS v3.0.1 c4_utils.py)
_MIN_WORDS_PER_LINE = 5
_MIN_NUM_SENTENCES = 3
_MAX_WORD_LENGTH = 1000
_MAX_PAGE_LENGTH = 1.9e5
_END_MARKS = (".", "?", "!", '"')
_ELLIPSIS = "..."
_POLICY_SUBSTRINGS = (
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "use of cookies",
    "use cookies",
)
_CITATION_REGEX = re.compile(r"\[\d*\]|\[edit\]|\[citation needed\]")


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# 1. WET parsing and download
# -----------------------------------------------------------------------------
def split_wet_stream(stream: io.TextIOBase) -> Generator[Tuple[str, Dict], None, None]:
    """Parse Common Crawl WET format stream into pages (url, features).

    Matches TFDS v3.0.1 c4_utils.py: split_wet_file
    """
    url: Optional[str] = None
    timestamp: Optional[str] = None
    content_type: Optional[str] = None
    content: List[str] = []
    DELIM = "WARC/1.0"

    for line in stream:
        line = line.strip()
        if not line:
            continue

        if line == DELIM:
            if url and content:
                yield url, {
                    "text": "\n".join(content),
                    "timestamp": timestamp or "",
                    "content_type": content_type or "",
                }
            url = ""
            timestamp = ""
            content_type = ""
            content = []
        elif line.startswith("WARC-Target-URI:"):
            url = line[len("WARC-Target-URI:"):].strip()
        elif line.startswith("WARC-Date:"):
            timestamp = line[len("WARC-Date:"):].strip()
        elif line.startswith("Content-Type:"):
            content_type = line[len("Content-Type:"):].strip()
        elif line.startswith(("WARC", "CONTENT-", "Content-")):
            continue
        else:
            content.append(line)

    if url and content:
        yield url, {
            "text": "\n".join(content),
            "timestamp": timestamp or "",
            "content_type": content_type or "",
        }


def _open_url(url: str, timeout: int = 60, retries: int = 6):
    """urlopen with backoff; Common Crawl returns 503 when throttling."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            client_error = isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429
            if client_error or attempt == retries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"  {type(e).__name__}: {e} - retrying in {wait}s")
            time.sleep(wait)


def get_cc_wet_urls(crawl_id: str = "CC-MAIN-2024-10", limit: int = 5) -> List[str]:
    """Fetch real WET shard paths for any Common Crawl dump."""
    paths_url = f"{CC_BASE_URL}crawl-data/{crawl_id}/wet.paths.gz"
    print(f"Fetching WET paths index from: {paths_url}...")
    with _open_url(paths_url, timeout=30) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            text_gz = io.TextIOWrapper(gz, encoding="utf-8")
            urls = []
            for _ in range(limit):
                line = text_gz.readline().strip()
                if line:
                    urls.append(CC_BASE_URL + line)
            return urls


def download_wet(url: str, cache_dir: str) -> str:
    """Download a WET shard to cache_dir (skipped if already there) and return its path."""
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, url.rsplit("/", 1)[-1])
    if os.path.exists(out_path):
        print(f"Using cached WET shard: {out_path}")
        return out_path
    print(f"Downloading WET shard: {url}...")
    tmp_path = out_path + ".part"
    for attempt in range(4):
        try:
            with _open_url(url) as resp, open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 3:
                raise
            print(f"  Download interrupted ({e}); restarting")
            time.sleep(10)
    os.replace(tmp_path, out_path)
    print(f"Saved {os.path.getsize(out_path) / 1e6:.1f} MB to {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# 2. Tokenizers and Badwords Loading
# -----------------------------------------------------------------------------
def get_sentence_tokenizer():
    """NLTK punkt English tokenizer (TFDS loaded punkt/english.pickle; NLTK >= 3.8.2 ships the
    same parameters as punkt_tab). Fails loudly rather than falling back to a different splitter."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize.punkt import PunktTokenizer
    return PunktTokenizer("english")


def load_badwords(badwords_path: Optional[str] = None) -> Tuple[List[str], re.Pattern]:
    """Load the 377 LDNOOBW bad-words list and compile the TFDS regex."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [badwords_path] if badwords_path else []
    candidates += [
        os.path.join(here, "..", "..", "..", "research-empirical-methods", "reference-code", "tfds_c4", "ldnoobw_en_25e679f.txt"),
        "ldnoobw_en_25e679f.txt",
    ]
    words: List[str] = []
    for p in candidates:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                words = [line.strip() for line in f if line.strip()]
            break
    else:
        print("Fetching LDNOOBW 25e679f bad-words list from GitHub...")
        with _open_url(LDNOOBW_URL, timeout=30) as resp:
            content = resp.read().decode("utf-8")
            words = [line.strip() for line in content.splitlines() if line.strip()]

    # TFDS v3.0.1 regex (not re.escape'd; no entry in this list has a regex metacharacter)
    bad_regex = re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))
    return words, bad_regex


# -----------------------------------------------------------------------------
# 3. C4 Filter Hierarchy
# -----------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """TFDS v3.0.1 normalize_url (used as the key for URL and line dedupe)."""
    url = re.sub(r"https?:\/\/(www\.)?", "", url)
    url = re.sub(r"\?(utm_|ref|feed).*", "", url)
    return url.rstrip("/")


def clean_page(
    text: str,
    tokenizer,
    min_words_per_line: int = _MIN_WORDS_PER_LINE,
    min_num_sentences: int = _MIN_NUM_SENTENCES,
    max_word_length: int = _MAX_WORD_LENGTH,
) -> Tuple[Optional[str], str]:
    """Apply C4 line cleaning rules (TFDS v3.0.1 clean_page, same rule order).

    Returns:
        (cleaned_text, status): status is 'passed' or failure reason.
    """
    valid_lines: List[str] = []
    num_sentences = 0

    for line in text.splitlines():
        line = line.strip()

        # Line filter: max word length
        if any(len(w) > max_word_length for w in line.split()):
            continue

        # Citation marker removal
        line = _CITATION_REGEX.sub("", line)

        # Line filter: terminal punctuation and ellipsis check
        if not line.endswith(_END_MARKS) or line.endswith(_ELLIPSIS):
            continue

        # Line filter: minimum words per line
        if len(line.split()) < min_words_per_line:
            continue

        line_lower = line.lower()

        # Doc filter: lorem ipsum
        if "lorem ipsum" in line_lower:
            return None, "loremipsum"

        # Line filter: javascript notice (TFDS checks this before the curly bracket)
        if "javascript" in line_lower:
            continue

        # Doc filter: curly bracket (JavaScript code heuristic)
        if "{" in line:
            return None, "curly_bracket"

        # Line filter: policy / cookie lines
        if any(policy_str in line_lower for policy_str in _POLICY_SUBSTRINGS):
            continue

        # Valid line passed: count sentences
        num_sentences += len(tokenizer.tokenize(line))
        valid_lines.append(line)

    if num_sentences < min_num_sentences:
        return None, "too_few_sentences"

    return "\n".join(valid_lines).strip(), "passed"


def remove_duplicate_lines(
    pages: Dict[str, Dict], tokenizer, counts: Counter, min_num_sentences: int = _MIN_NUM_SENTENCES
) -> Dict[str, Dict]:
    """TFDS v3.0.1 remove_duplicate_text over `pages` ({normalised_url: features})."""
    # Each line hash is kept only in the page whose key has the smallest md5.
    owner: Dict[str, Tuple[str, str]] = {}
    for key, features in pages.items():
        key_hash = _md5(key)
        for line in features["text"].split("\n"):
            h = _md5(line.strip().lower())
            if h not in owner or key_hash < owner[h][0]:
                owner[h] = (key_hash, key)

    kept: Dict[str, Dict] = {}
    for key, features in pages.items():
        new_lines, seen = [], set()
        for line in features["text"].split("\n"):
            h = _md5(line.strip().lower())
            if owner[h][1] != key:
                counts["lines_removed_global_duplicate"] += 1
            elif h in seen:
                counts["lines_removed_local_duplicate"] += 1
            else:
                new_lines.append(line)
                seen.add(h)
        new_text = "\n".join(new_lines)
        if not new_text:
            counts["dropped_dedupe_empty"] += 1
            continue
        if len(tokenizer.tokenize(new_text)) < min_num_sentences:
            counts["dropped_dedupe_too_few_sentences"] += 1
            continue
        kept[key] = dict(features, text=new_text)
    return kept


def check_english(text: str, min_probability: float = 0.99) -> bool:
    """Deterministic C4 langdetect check."""
    import langdetect
    langdetect.DetectorFactory.seed = 0
    try:
        preds = langdetect.detect_langs(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return False
    return bool(preds) and preds[0].prob >= min_probability and preds[0].lang == "en"


# -----------------------------------------------------------------------------
# 4. Main Processing Engine
# -----------------------------------------------------------------------------
def _iter_source(source, crawl_id, wet_url, shard_idx, local_file, cache_dir):
    """Yield (url, features) from the chosen source."""
    if source == "commoncrawl":
        local_file = download_wet(wet_url, cache_dir)

    if source in ("commoncrawl", "local"):
        if not local_file:
            raise ValueError("--local_file must be specified when --source=local")
        opener = gzip.open if local_file.endswith(".gz") else open
        with opener(local_file, "rt", encoding="utf-8", errors="ignore") as f:
            if "warc.wet" in local_file:
                yield from split_wet_stream(f)
            else:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        yield d.get("url", ""), d
        return

    from huggingface_hub import hf_hub_download
    hf_config = "en.noblocklist" if source == "noblocklist" else "en.noclean"
    shard_file = f"{hf_config}/c4-train.{shard_idx:05d}-of-01024.json.gz"
    print(f"Downloading shard {shard_file} from allenai/c4...")
    local_path = hf_hub_download(repo_id="allenai/c4", filename=shard_file, repo_type="dataset")
    with gzip.open(local_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                yield d.get("url", ""), d


def run_pipeline(
    source: str = "commoncrawl",
    crawl_id: str = "CC-MAIN-2024-10",
    wet_url: Optional[str] = None,
    shard_idx: int = 0,
    max_docs: Optional[int] = None,
    out_dir: str = "output",
    badwords_path: Optional[str] = None,
    local_file: Optional[str] = None,
    cache_dir: str = "wet_cache",
) -> Dict:
    """Run filtering pipeline and write c4_en, c4_blocklist, and summary stats."""
    os.makedirs(out_dir, exist_ok=True)
    words, bad_regex = load_badwords(badwords_path)
    tokenizer = get_sentence_tokenizer()

    print(f"Loaded {len(words)} badwords regex terms.")
    print(f"Running mode: {source} (Crawl/Shard: {crawl_id if source=='commoncrawl' else shard_idx}, Max docs: {max_docs})")

    # Output file paths
    prefix = f"{crawl_id}_shard{shard_idx}" if source == "commoncrawl" else f"{source}_shard{shard_idx}"
    en_path = os.path.join(out_dir, f"c4_en_{prefix}.jsonl.gz")
    block_path = os.path.join(out_dir, f"c4_blocklist_{prefix}.jsonl.gz")
    stats_path = os.path.join(out_dir, f"stats_{prefix}.json")

    if source == "commoncrawl" and not wet_url:
        wet_url = get_cc_wet_urls(crawl_id=crawl_id, limit=shard_idx + 1)[shard_idx]

    counts = Counter()
    start_time = time.time()
    full_pipeline = source != "noblocklist"

    # Pass 1: read, length filter, URL dedupe (keep max md5(raw text)), clean.
    # Cleaning is per page, so cleaning before picking the URL winner gives the same result.
    by_url: Dict[str, Tuple[str, str, str, Optional[str], str]] = {}
    for idx, (url, features) in enumerate(_iter_source(source, crawl_id, wet_url, shard_idx, local_file, cache_dir)):
        if max_docs and idx >= max_docs:
            break
        counts["total_input"] += 1
        raw_text = features.get("text", "")

        if source == "noblocklist":
            cleaned_text, status = raw_text, "passed"
        else:
            if len(raw_text) > _MAX_PAGE_LENGTH:
                counts["dropped_too_long"] += 1
                continue
            cleaned_text, status = clean_page(raw_text, tokenizer)

        key = normalize_url(url) if full_pipeline else url
        text_hash = _md5(raw_text)
        if key in by_url:
            counts["dropped_url_duplicate"] += 1
            if text_hash <= by_url[key][0]:
                continue
        by_url[key] = (text_hash, url, features.get("timestamp", ""), cleaned_text, status)

        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"Read {idx + 1} pages ({(idx + 1) / max(0.01, elapsed):.1f} page/s)")

    pages: Dict[str, Dict] = {}
    for key, (_, url, timestamp, cleaned_text, status) in by_url.items():
        if status != "passed":
            counts[f"dropped_{status}"] += 1
            continue
        pages[key] = {"url": url, "timestamp": timestamp, "text": cleaned_text}
    counts["passed_clean"] = len(pages)
    del by_url

    # Pass 2: line dedupe across the pages read in this run.
    if full_pipeline:
        pages = remove_duplicate_lines(pages, tokenizer, counts)
        counts["passed_dedupe"] = len(pages)

    # Pass 3: langdetect, then badwords split.
    with gzip.open(en_path, "wt", encoding="utf-8") as f_en, \
         gzip.open(block_path, "wt", encoding="utf-8") as f_block:

        for n, features in enumerate(pages.values(), 1):
            text = features["text"]
            if source != "noblocklist" and not check_english(text):
                counts["dropped_langdetect"] += 1
                continue
            counts["passed_langdetect_noblocklist"] += 1

            bad_matches = bad_regex.findall(text.lower())
            if bad_matches:
                counts["blocklist_excluded"] += 1
                record = dict(features, matched_badwords=sorted(set(bad_matches)))
                f_block.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                counts["c4_en_kept"] += 1
                f_en.write(json.dumps(features, ensure_ascii=False) + "\n")

            if n % 5000 == 0:
                print(f"Language/badwords: {n}/{len(pages)} | C4.EN: {counts['c4_en_kept']}, Blocklist: {counts['blocklist_excluded']}")

    total = counts["total_input"]
    noblock = counts["passed_langdetect_noblocklist"]
    summary = {
        "source": source,
        "crawl_or_shard": crawl_id if source == "commoncrawl" else shard_idx,
        "wet_url_or_file": wet_url or local_file,
        "total_documents_scanned": total,
        "noblocklist_docs": noblock,
        "c4_en_kept": counts["c4_en_kept"],
        "c4_en_rate": round(counts["c4_en_kept"] / max(1, total), 4),
        "blocklist_excluded": counts["blocklist_excluded"],
        "blocklist_rate_of_all": round(counts["blocklist_excluded"] / max(1, total), 4),
        "blocklist_rate_of_noblocklist": round(counts["blocklist_excluded"] / max(1, noblock), 4),
        "breakdown": dict(counts),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    with open(stats_path, "w", encoding="utf-8") as f_stats:
        json.dump(summary, f_stats, indent=2)

    print("\n--- Pipeline Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"Saved clean C4.EN to: {en_path}")
    print(f"Saved excluded blocklist to: {block_path}")
    print(f"Saved audit statistics to: {stats_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Web / Common Crawl dumps through C4 filtering hierarchy")
    parser.add_argument("--source", choices=["commoncrawl", "noblocklist", "noclean", "local"], default="commoncrawl",
                        help="Data source mode: 'commoncrawl' (modern WET crawl), 'noblocklist', 'noclean', or 'local'")
    parser.add_argument("--crawl", type=str, default="CC-MAIN-2024-10",
                        help="Common Crawl dump ID (e.g. CC-MAIN-2024-10, CC-MAIN-2024-51, CC-MAIN-2023-50)")
    parser.add_argument("--wet_url", type=str, default=None,
                        help="Direct URL to a specific .warc.wet.gz file on data.commoncrawl.org")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--max_docs", type=int, default=None, help="Max documents to scan (e.g. 20000 for dry run)")
    parser.add_argument("--out_dir", type=str, default="c4_filtered_output", help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="wet_cache", help="Where downloaded WET shards are kept")
    parser.add_argument("--badwords_file", type=str, default=None, help="Custom path to badwords txt")
    parser.add_argument("--local_file", type=str, default=None, help="Path to local file if source=local")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        crawl_id=args.crawl,
        wet_url=args.wet_url,
        shard_idx=args.shard,
        max_docs=args.max_docs,
        out_dir=args.out_dir,
        badwords_path=args.badwords_file,
        local_file=args.local_file,
        cache_dir=args.cache_dir,
    )
