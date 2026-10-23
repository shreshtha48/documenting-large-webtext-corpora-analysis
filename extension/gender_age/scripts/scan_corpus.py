"""Extension, corpus: one pass over C4.EN.NOBLOCKLIST shards for gender and age terms.

Per document (fields text / url / timestamp; paragraphs = lines split on "\\n"):
1. Blocklist (Dodge §5): excluded if C4's own regex matches (TFDS v3.0.1, lowercased text, LDNOOBW @ 25e679f, not
   escaped), exactly as reproduction/excluded_data/scripts/scan_noblocklist.py. Kept documents are C4.EN (the
   one-shard pilot matched the released C4.EN for 99.4% of documents).
2. Terms (TERMS below): which gender / age / gender x age / gender x ethnicity terms the document mentions.
3. Mechanical exclusions: LDNOOBW itself contains gender / age / ethnicity words ("girl on", "big black",
   "sultry women", "wrapping men", "milf", ...). A document saying "a girl on the bus" is removed because of the
   phrase that also makes it a "girl" document. For each term we also count exclusions caused by at least one
   blocklist entry that does NOT contain the term's words ("excluded_other"). PMI on that count is the
   non-mechanical part.
4. Dialect (Dodge §5.3, optional): the Blodgett et al. argmax, as in the reproduction (dialect.py), so exclusion can
   be split by dialect x gender ("AAE documents that mention women").
5. URL year: C4's `timestamp` is the crawl date (every document is April 2019; checked on 50,000 documents), so it
   cannot date a page. A year in the URL path (/2014/05/, -2016-) is used as a publication-date proxy, the same idea
   as Dodge §3.2 (which used Internet Archive first-capture dates for a sample). Most URLs carry no year.
6. Sentiment (Dodge A.7) for paragraphs that mention a term, in kept and in excluded documents separately: positive
   and negative token counts under the three reproduction lexicons (SocialSent top 250+250, VADER |score| > 1,
   UnQover). The term words themselves are removed before counting (SocialSent scores "man" as positive: +2.1).

Matching (ours; Dodge's A.7 used case-sensitive fgrep " Jewish "): case-insensitive with word boundaries, because
gender words often start a sentence ("Women ..."). " Jewish " / " Arab " are also counted the paper's way
(case-sensitive, spaces) as a positive control: on these shards their SocialSent gap should match Arm B (7.1).

Counts are summed per (web domain, URL year bucket) so the analysis can bootstrap over domains.

Outputs per shard, in --out_dir:
    doc_counts_<shard>.npz    rows "host<TAB>year" x DOC_COLUMNS
    sent_counts_<shard>.npz   rows "host<TAB>year<TAB>kept" x SENT_COLUMNS
    para_sample_<shard>.jsonl.gz   a random sample of matching paragraphs per term (kept and excluded), for the
                              sentiment classifier (classify_paragraphs.py)
    stats_<shard>.json        written last; marks the shard done

Usage:
    python scripts/scan_corpus.py --n 12 --seed 0 --workers 3 --out_dir results/C1_noblocklist_12_shards
    python scripts/scan_corpus.py --files /path/c4-train.00000-of-01024.json.gz --out_dir ...
Seed 0 with n=12 gives the same shard indices as the reproduction's Arm B (B1) and F5 (S1) runs.
"""
import argparse, gzip, json, os, random, re, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.parse import urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))
DATA = os.path.join(ROOT, "data")
BADWORDS = os.path.join(DATA, "ldnoobw_en_25e679f.txt")
LEXICONS = {"SocialSent": "socialsent_250.tsv", "VADER": "vader.tsv", "UnQover": "unqover.tsv"}
MODEL_DIR = os.path.join(DATA, "twitteraae_model")
N_SHARDS = 1024

FEM = ["woman", "women", "girl", "girls", "female", "females", "lady", "ladies"]
MAS = ["man", "men", "boy", "boys", "male", "males", "guy", "guys"]
ETHNICITIES = {"Jewish": ["jewish"], "Arab": ["arab", "arabs"], "Black": ["black"], "White": ["white"],
               "Asian": ["asian"], "Hispanic": ["hispanic"], "Latina/o": ["latina", "latino", "latinas", "latinos"],
               "African American": ["african-american", "african american"], "Indian": ["indian"],
               "European": ["european"], "Chinese": ["chinese"], "Muslim": ["muslim"]}
# Terms are matched on words: the lowercased paragraph is split into words (letters, hyphen-joined) and punctuation
# marks, and each term is a set of word sequences of length 1-3 ("teenage girl" = ("teenage", "girl")). A phrase
# never spans punctuation or a paragraph break. Matching is case-insensitive (ours; A.7's fgrep was case-sensitive).
# name -> (word sequences, axis, gender, age, ethnicity)
def _seqs(*phrases):
    return [tuple(p.split()) for p in phrases]


def _cross(first, second):
    return [tuple(a.split()) + (b,) for a in first for b in second]


TERMS = {
    "women": (_seqs("woman", "women"), "gender", "female", "", ""),
    "men": (_seqs("man", "men"), "gender", "male", "", ""),
    "female": (_seqs("female", "females"), "gender", "female", "", ""),
    "male": (_seqs("male", "males"), "gender", "male", "", ""),
    "girl": (_seqs("girl", "girls"), "gender", "female", "child/teen", ""),
    "boy": (_seqs("boy", "boys"), "gender", "male", "child/teen", ""),
    "lady": (_seqs("lady", "ladies"), "gender", "female", "", ""),
    "gentleman": (_seqs("gentleman", "gentlemen"), "gender", "male", "", ""),
    "mother": (_seqs("mother", "mothers", "mom", "moms", "mum", "mums"), "gender", "female", "", ""),
    "father": (_seqs("father", "fathers", "dad", "dads"), "gender", "male", "", ""),
    "wife": (_seqs("wife", "wives"), "gender", "female", "", ""),
    "husband": (_seqs("husband", "husbands"), "gender", "male", "", ""),
    "daughter": (_seqs("daughter", "daughters"), "gender", "female", "", ""),
    "son": (_seqs("son", "sons"), "gender", "male", "", ""),
    "she/her": (_seqs("she", "her", "hers", "herself"), "pronoun", "female", "", ""),
    "he/him": (_seqs("he", "him", "his", "himself"), "pronoun", "male", "", ""),
    "child": (_seqs("child", "children", "kid", "kids"), "age", "", "child", ""),
    "teen": (_seqs("teen", "teens", "teenager", "teenagers", "teenage", "teenaged", "adolescent", "adolescents"),
             "age", "", "teen", ""),
    "young adult": (_seqs("young adult", "young adults", "young people"), "age", "", "young adult", ""),
    "adult": (_seqs("adult", "adults"), "age", "", "adult", ""),
    "middle-aged": (_seqs("middle-aged", "middle aged"), "age", "", "middle-aged", ""),
    "elderly": (_seqs("elderly", "old people", "older people", "older adults", "older persons", "senior citizen",
                      "senior citizens", "seniors", "pensioner", "pensioners", "retiree", "retirees"),
                "age", "", "elderly", ""),
    "person": (_seqs("person", "people"), "reference", "", "", ""),
    "teen girl": (_cross(["teen", "teenage", "teenaged"], ["girl", "girls"]), "gender_age", "female", "teen", ""),
    "teen boy": (_cross(["teen", "teenage", "teenaged"], ["boy", "boys"]), "gender_age", "male", "teen", ""),
    "little girl": (_cross(["little"], ["girl", "girls"]), "gender_age", "female", "child", ""),
    "little boy": (_cross(["little"], ["boy", "boys"]), "gender_age", "male", "child", ""),
    "young woman": (_cross(["young"], ["woman", "women"]), "gender_age", "female", "young adult", ""),
    "young man": (_cross(["young"], ["man", "men"]), "gender_age", "male", "young adult", ""),
    "middle-aged woman": (_cross(["middle-aged", "middle aged"], ["woman", "women"]), "gender_age", "female",
                          "middle-aged", ""),
    "middle-aged man": (_cross(["middle-aged", "middle aged"], ["man", "men"]), "gender_age", "male", "middle-aged", ""),
    "old woman": (_cross(["old", "older", "elderly"], ["woman", "women"]), "gender_age", "female", "elderly", ""),
    "old man": (_cross(["old", "older", "elderly"], ["man", "men"]), "gender_age", "male", "elderly", ""),
}
for _e, _forms in ETHNICITIES.items():
    TERMS[f"{_e} women"] = (_cross(_forms, FEM), "gender_eth", "female", "", _e)
    TERMS[f"{_e} men"] = (_cross(_forms, MAS), "gender_eth", "male", "", _e)
CASED = {"Jewish (paper)": " Jewish ", "Arab (paper)": " Arab "}   # A.7's fgrep matching, positive control
WORD_TERMS = list(TERMS)
TERMS.update({k: ([], "control", "", "", "") for k in CASED})
TERM_NAMES = list(TERMS)
LOOKUP = defaultdict(set)          # word sequence -> term names
for _t in WORD_TERMS:
    for _seq in TERMS[_t][0]:
        LOOKUP[_seq].add(_t)
MAXLEN = max(len(k) for k in LOOKUP)
FIRST = {k[0] for k in LOOKUP}     # words that can start a term; all other words are skipped quickly
# every word of every term, removed before sentiment counting (SocialSent scores "man" +2.1)
# (tokens() strips punctuation, so "middle-aged" arrives as "middleaged")
TERM_TOKENS = ({w for seq in LOOKUP for x in seq for w in x.split("-")} | {x.replace("-", "") for seq in LOOKUP for x in seq}
               | {"jew", "jews"})
WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*|[^\sa-z]")
DIALECTS = ["AAE", "Hispanic", "Other", "WAE", "unassigned"]
YEAR_BUCKETS = ["none", "<=2009", "2010-12", "2013-15", "2016-17", "2018-19"]
YEAR_RE = re.compile(r"[/_.\-](199[5-9]|20[01][0-9])(?=[/_.\-]|$)")

DOC_COLUMNS = (["docs", "excluded"]
               + [f"{t}|{k}" for t in TERM_NAMES for k in ("docs", "excluded", "excluded_other")]
               + [f"dialect:{d}|{g}|{k}" for d in DIALECTS for g in ("all", "female", "male")
                  for k in ("docs", "excluded")])
SENT_COLUMNS = [f"{t}|{x}" for t in TERM_NAMES for x in
                ["paragraphs", "words"] + [f"{l}:{p}" for l in LEXICONS for p in ("pos", "neg")]]


def shard_name(i):
    return f"c4-train.{i:05d}-of-{N_SHARDS:05d}"


def year_bucket(url):
    m = YEAR_RE.search(urlparse(url).path)
    if not m:
        return "none"
    y = int(m.group(1))
    return ("<=2009" if y <= 2009 else "2010-12" if y <= 2012 else "2013-15" if y <= 2015
            else "2016-17" if y <= 2017 else "2018-19")


def badwords():
    words = [w.strip() for w in open(BADWORDS, encoding="utf-8") if w.strip()]
    return words, re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))   # TFDS v3.0.1: not escaped


def match_terms(text):
    """Term names mentioned in `text` (word sequences, case-insensitive) plus the paper-style cased matches."""
    w = WORD_RE.findall(text.lower())
    h = set()
    for i, x in enumerate(w):
        if x not in FIRST:
            continue
        for k in range(1, MAXLEN + 1):
            t = LOOKUP.get(tuple(w[i:i + k]))
            if t:
                h |= t
    h.update(t for t, s in CASED.items() if s in text)
    return h


def mechanical_overlap(words):
    """term -> blocklist entries that contain one of the term's words ('girl on' for "girl", 'big black' for
    "Black women"). An exclusion caused only by such entries is partly built into the blocklist."""
    out = {}
    for t, (seqs, *_) in TERMS.items():
        tw = {x for seq in seqs for x in seq} or {CASED[t].strip().lower()}
        out[t] = {e for e in words if tw & set(WORD_RE.findall(e))}
    return out


def load_lexicons():
    lex = {}
    for name, fn in LEXICONS.items():
        rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(DATA, fn), encoding="utf-8")][1:]
        lex[name] = {w: p for w, p, *_ in rows}
    return lex


PUNCT = re.compile(r"[^\w\s]")


def tokens(text):
    return [t for t in (PUNCT.sub("", w).lower() for w in text.split()) if t and t not in TERM_TOKENS]


class Reservoir:
    def __init__(self, cap, rng):
        self.cap, self.rng, self.n, self.items = cap, rng, 0, []

    def add(self, x):
        self.n += 1
        if len(self.items) < self.cap:
            self.items.append(x)
        else:
            j = self.rng.randrange(self.n)
            if j < self.cap:
                self.items[j] = x


def scan_file(path, shard, args):
    words, bad = badwords()
    overlap = mechanical_overlap(words)
    lex = load_lexicons()
    dcol = {c: i for i, c in enumerate(DOC_COLUMNS)}
    scol = {c: i for i, c in enumerate(SENT_COLUMNS)}
    female_terms = {t for t, v in TERMS.items() if v[2] == "female" and v[1] != "pronoun"}
    male_terms = {t for t, v in TERMS.items() if v[2] == "male" and v[1] != "pronoun"}
    model = None
    if args.dialect:
        from dialect import DialectModel
        model = DialectModel(MODEL_DIR)
    rng = random.Random(f"{args.seed}-{shard}")
    samples = {(t, k): Reservoir(args.sample_kept if k else args.sample_excluded, rng)
               for t in TERM_NAMES for k in (1, 0)}
    doc_rows = defaultdict(lambda: np.zeros(len(DOC_COLUMNS), dtype=np.int64))
    sent_rows = defaultdict(lambda: np.zeros(len(SENT_COLUMNS), dtype=np.int64))
    n = n_ex = 0

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if args.limit_docs and n >= args.limit_docs:
                break
            d = json.loads(line)
            text, url = d["text"], d["url"]
            n += 1
            low = text.lower()
            matched = {m.strip() for m in bad.findall(low)}
            ex = bool(matched)
            n_ex += ex
            host, yb = urlparse(url).netloc, year_bucket(url)
            row = doc_rows[f"{host}\t{yb}"]
            row[0] += 1
            row[1] += ex
            par_hits = [(p, match_terms(p)) for p in text.split("\n")]
            doc_terms = set().union(*(h for _, h in par_hits))
            for t in doc_terms:
                row[dcol[f"{t}|docs"]] += 1
                if ex:
                    row[dcol[f"{t}|excluded"]] += 1
                    row[dcol[f"{t}|excluded_other"]] += bool(matched - overlap[t])
            if model is not None:
                post = model.fast(model.tokens(text))
                dl = DIALECTS[int(np.argmax(post))] if post is not None else "unassigned"
                for g, present in (("all", True), ("female", bool(doc_terms & female_terms)),
                                   ("male", bool(doc_terms & male_terms))):
                    if present:
                        row[dcol[f"dialect:{dl}|{g}|docs"]] += 1
                        row[dcol[f"dialect:{dl}|{g}|excluded"]] += ex
            if not doc_terms:
                continue
            kept = 0 if ex else 1
            srow = sent_rows[f"{host}\t{yb}\t{kept}"]
            for p, ph in par_hits:
                if not ph:
                    continue
                toks = tokens(p)
                counts = {l: [0, 0] for l in LEXICONS}
                for tok in toks:
                    for l, lx in lex.items():
                        pol = lx.get(tok)
                        if pol == "pos":
                            counts[l][0] += 1
                        elif pol == "neg":
                            counts[l][1] += 1
                nw = len(p.split())
                for t in ph:
                    srow[scol[f"{t}|paragraphs"]] += 1
                    srow[scol[f"{t}|words"]] += nw
                    for l, (cp, cn) in counts.items():
                        srow[scol[f"{t}|{l}:pos"]] += cp
                        srow[scol[f"{t}|{l}:neg"]] += cn
                    samples[(t, kept)].add({"term": t, "kept": kept, "url": url, "year": yb, "text": p[:1500]})

    def save_counts(rows, columns, name):
        keys = list(rows)
        np.savez_compressed(os.path.join(args.out_dir, f"{name}_{shard}.npz"), keys=np.array(keys, dtype=str),
                            counts=np.stack([rows[k] for k in keys]) if keys else np.zeros((0, len(columns)), np.int64),
                            columns=np.array(columns))

    save_counts(doc_rows, DOC_COLUMNS, "doc_counts")
    save_counts(sent_rows, SENT_COLUMNS, "sent_counts")
    tmp = os.path.join(args.out_dir, f"para_sample_{shard}.jsonl.gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as g:
        for r in samples.values():
            for x in r.items:
                g.write(json.dumps({**x, "shard": shard}, ensure_ascii=False) + "\n")
    os.replace(tmp, tmp[:-4])
    stats = {"shard": shard, "documents": n, "excluded": n_ex, "excluded_rate": n_ex / max(1, n),
             "dialect": bool(model), "terms": TERM_NAMES,
             "mechanical_overlap": {t: sorted(s) for t, s in overlap.items() if s}}
    with open(os.path.join(args.out_dir, f"stats_{shard}.json"), "w") as f:   # written last = shard done
        json.dump(stats, f, indent=1)
    return stats


def process(shard, path, args):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(REPO, "reproduction", "excluded_data", "scripts"))
    t0 = time.time()
    downloaded = False
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("allenai/c4", f"en.noblocklist/{shard}.json.gz", repo_type="dataset",
                               cache_dir=args.cache_dir)
        downloaded = True
    s = scan_file(path, shard, args)
    if downloaded and not args.keep_shards:
        os.remove(os.path.realpath(path))
    return f"{shard}: {s['documents']} docs, {100 * s['excluded_rate']:.2f}% excluded, {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--range", type=int, nargs=2, metavar=("START", "END"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0, help="shard choice (seed 0 = the reproduction's shard indices)")
    ap.add_argument("--files", nargs="*", help="local en.noblocklist shard files instead of downloading")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no_dialect", dest="dialect", action="store_false", help="skip the dialect model (faster)")
    ap.add_argument("--sample_kept", type=int, default=300, help="paragraphs sampled per term from kept docs")
    ap.add_argument("--sample_excluded", type=int, default=100, help="paragraphs sampled per term from excluded docs")
    ap.add_argument("--limit_docs", type=int, default=0, help="only the first N documents per shard (smoke tests)")
    ap.add_argument("--keep_shards", action="store_true")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(REPO, "reproduction", "excluded_data", "scripts"))
    sys.path.insert(0, HERE)
    if args.dialect:
        from dialect import ensure_files
        ensure_files(MODEL_DIR)                      # download once, before workers start

    if args.files:
        jobs = [(re.search(r"c4-train\.\d{5}-of-\d{5}", os.path.basename(p)).group(0), p) for p in args.files]
    else:
        if args.shards:
            idx = args.shards
        elif args.range:
            idx = list(range(max(0, args.range[0]), min(N_SHARDS, args.range[1])))
        else:
            idx = sorted(random.Random(args.seed).sample(range(N_SHARDS), args.n))
        jobs = [(shard_name(i), None) for i in idx]
    meta_path = os.path.join(args.out_dir, "shards.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"shards": []}
    if meta.get("terms", TERM_NAMES) != TERM_NAMES:
        raise SystemExit(f"{args.out_dir} was scanned with another term list; use another --out_dir")
    meta.update(shards=sorted(set(meta["shards"]) | {s for s, _ in jobs}), config="en.noblocklist", terms=TERM_NAMES)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    todo = [(s, p) for s, p in jobs if not os.path.exists(os.path.join(args.out_dir, f"stats_{s}.json"))]
    print(f"{len(jobs)} shards requested, {len(jobs) - len(todo)} done already, {len(todo)} to scan", flush=True)
    t_all = time.time()
    if args.workers <= 1:
        for s, p in todo:
            print(process(s, p, args), flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(process, s, p, args) for s, p in todo]
            for i, fu in enumerate(as_completed(futs), 1):
                print(f"[{i}/{len(todo)}, {(time.time() - t_all) / 60:.0f} min] {fu.result()}", flush=True)


if __name__ == "__main__":
    main()
