"""Arm B of Dodge et al. 2021 §4.3 / A.7: build the three sentiment lexicons.

1. SocialSent-derived (the paper's primary lexicon). Hamilton et al. (2016) induced one lexicon per subreddit
   for 250 subreddits. A.7: "we aggregated these by averaging the sentiment scores for all words that appeared in
   at least 40 subreddit-specific lexicons ... manually filtered to remove any overtly ethnic terms, then took the
   top 250 most polarized words from each side". The Stanford download page is unreachable, so the 250 lexicons
   come from the copy bundled in shifterator (pinned commit below).
2. VADER (Hutto and Gilbert 2014): positive if mean score > 1.0, negative if < -1.0 (A.7).
3. UnQover-derived: the paper's "small manually-created" lexicon from the UnQover question words was not
   released. We rebuild an approximation from the UnQover attribute file: content words of the negative
   stereotypes are negative candidates, content words of their positive counterparts are positive candidates.
   Instead of a hand pass, a candidate is kept only if VADER or the SocialSent aggregate (|score| >= 0.5) gives
   it the same sign; this drops neutral words such as "house" or "because". Ethnic terms are removed.

Writes data/lexicons/{socialsent_250.tsv, vader.tsv, unqover.tsv} (word, polarity in {pos, neg}, score) plus
socialsent_aggregate.tsv (every word kept by the >=40 rule), socialsent_raw.json.gz (the 250 source lexicons)
and MANIFEST.json (sources, commits, counts).

Usage: python scripts/build_lexicons.py   (downloads the sources once, then reuses socialsent_raw.json.gz)
"""
import gzip, json, os, re, urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "lexicons")

SHIFTERATOR_SHA = "8d74d74a94b5a046928f99ba076ff4fac74dd2c4"
VADER_SHA = "44fc044cd877310ee8278a0eadf34bcd50d41d06"
SS_TREE = f"https://api.github.com/repos/ryanjgallagher/shifterator/git/trees/{SHIFTERATOR_SHA}?recursive=1"
SS_RAW = f"https://raw.githubusercontent.com/ryanjgallagher/shifterator/{SHIFTERATOR_SHA}/"
VADER_URL = f"https://raw.githubusercontent.com/cjhutto/vaderSentiment/{VADER_SHA}/vaderSentiment/vader_lexicon.txt"
UNQOVER_ATTRS = os.path.join(ROOT, "third_party", "unqover", "word_lists", "activities", "biased_ethnicity")

MIN_LEXICONS = 40     # A.7
TOP_K = 250           # A.7
VADER_THRESHOLD = 1.0 # A.7

# Our manual filter of "overtly ethnic terms" (A.7 does not list theirs). Group names, demonyms and
# religion/nationality words; applied before taking the top 250 on each side. Reported as an assumption.
ETHNIC_TERMS = {
    "jewish", "jew", "jews", "arab", "arabs", "arabic", "muslim", "muslims", "islam", "islamic", "christian",
    "christians", "black", "blacks", "white", "whites", "african", "africans", "asian", "asians", "european",
    "europeans", "american", "americans", "mexican", "mexicans", "chinese", "japanese", "korean", "indian",
    "indians", "hispanic", "latino", "latina", "native", "natives", "caucasian", "israeli", "israelis",
    "palestinian", "palestinians", "russian", "russians", "german", "germans", "french", "british", "english",
    "irish", "italian", "italians", "canadian", "canadians", "australian", "australians", "swedish", "dutch",
    "polish", "greek", "turkish", "iranian", "iraqi", "syrian", "pakistani", "somali", "gypsy", "gypsies",
    "roma", "nigger", "niggers", "negro", "kike", "spic", "chink", "gook", "wetback", "raghead", "towelhead",
}

PUNCT = re.compile(r"[^\w\s]")   # the same as count_sentiment.py

STOPWORDS = {
    # function words after punctuation removal; "doing" ("doing nails"), "going" ("going to hijack") and "hard"
    # ("hard to understand"; also "hard drive", "hard work") carry no sentiment on their own
    "didnt", "wasnt", "werent", "doing", "going", "hard",
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "at", "for", "with", "by", "from", "as", "their", "them",
    "they", "were", "was", "had", "have", "has", "been", "be", "is", "are", "very", "mostly", "always", "even",
    "didn't", "did", "not", "no", "how", "use", "other", "others", "others'", "about", "like", "looked", "seemed",
    "when", "into", "irrespective", "toward", "towards", "it", "its", "that", "this", "who", "all", "boot", "day",
    "room", "entering", "deal", "citizens", "street", "weekend", "work", "clothes", "wives", "daughters", "women",
    "children", "visitors", "tourists", "skin", "color", "accent", "accents", "teeth", "music", "dishes", "border",
    "computer", "cellphones", "pants", "money", "plane", "wilderness", "huts", "cartels", "alcohol",
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def load_socialsent():
    cache = os.path.join(OUT, "socialsent_raw.json.gz")
    if os.path.exists(cache):
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return json.load(f)
    tree = json.loads(fetch(SS_TREE))["tree"]
    paths = sorted(t["path"] for t in tree if t["path"].startswith("shifterator/lexicons/SocialSent-Reddit/")
                   and t["path"].endswith(".tsv"))
    lexicons = {}
    for i, p in enumerate(paths):
        name = p.rsplit("SocialSent-Reddit_", 1)[1][:-4]
        lex = {}
        for line in fetch(SS_RAW + p).splitlines():
            if line.strip():
                w, s = line.split("\t")
                lex[w] = float(s)
        lexicons[name] = lex
        if (i + 1) % 50 == 0:
            print(f"  downloaded {i + 1}/{len(paths)} subreddit lexicons")
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump(lexicons, f)
    return lexicons


def build_socialsent(lexicons):
    vals = defaultdict(list)
    for lex in lexicons.values():
        for w, s in lex.items():
            vals[w].append(s)
    agg = {w: sum(v) / len(v) for w, v in vals.items() if len(v) >= MIN_LEXICONS}
    counts = {w: len(vals[w]) for w in agg}
    kept = {w: s for w, s in agg.items() if w not in ETHNIC_TERMS}
    removed = sorted(w for w in agg if w in ETHNIC_TERMS)
    pos = sorted((w for w in kept if kept[w] > 0), key=lambda w: -kept[w])[:TOP_K]
    neg = sorted((w for w in kept if kept[w] < 0), key=lambda w: kept[w])[:TOP_K]
    return agg, counts, removed, [(w, "pos", kept[w]) for w in pos] + [(w, "neg", kept[w]) for w in neg]


def build_vader():
    rows, scores = [], {}
    for line in fetch(VADER_URL).splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        w, s = parts[0], float(parts[1])
        scores[w] = s
        if s > VADER_THRESHOLD:
            rows.append((w, "pos", s))
        elif s < -VADER_THRESHOLD:
            rows.append((w, "neg", s))
    return rows, scores


def build_unqover(vader_scores, ss_agg):
    pos, neg = set(), set()
    for line in open(UNQOVER_ATTRS, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("[p_act") or "|" not in line:
            continue
        phrase = line.split("|", 1)[1]
        # tokenise exactly as count_sentiment.py tokenises C4 (whitespace split, punctuation removed, lowercased),
        # so "well-behaved" stays one word ("wellbehaved"). v1 split on hyphens and added "well" and "mean" as
        # sentiment words, which then dominated the counts (kept as unqover_v1_hyphen_split.tsv).
        words = {w for w in (PUNCT.sub("", t).lower() for t in phrase.split()) if w not in STOPWORDS and len(w) > 2}
        # slot tags: [p_act], [p_act:neg], [p_act-single], [p_act-single:neg]; ":neg" marks the positive counterpart
        (pos if line.split("]", 1)[0].endswith(":neg") else neg).update(words)
    both = pos & neg

    def agrees(w, sign):
        refs = [vader_scores.get(w), ss_agg.get(w) if abs(ss_agg.get(w, 0)) >= 0.5 else None]
        return w not in ETHNIC_TERMS and any(r is not None and r * sign > 0 for r in refs)

    return ([(w, "pos", 1.0) for w in sorted(pos - both) if agrees(w, 1)] +
            [(w, "neg", -1.0) for w in sorted(neg - both) if agrees(w, -1)])


def write(name, rows):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write("word\tpolarity\tscore\n")
        for w, p, s in rows:
            f.write(f"{w}\t{p}\t{s:.4f}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    lexicons = load_socialsent()
    agg, counts, removed, ss = build_socialsent(lexicons)
    with open(os.path.join(OUT, "socialsent_aggregate.tsv"), "w", encoding="utf-8") as f:
        f.write("word\tmean_score\tn_lexicons\n")
        for w in sorted(agg, key=lambda w: -agg[w]):
            f.write(f"{w}\t{agg[w]:.4f}\t{counts[w]}\n")
    write("socialsent_250.tsv", ss)
    vader, vader_scores = build_vader()
    write("vader.tsv", vader)
    unq = build_unqover(vader_scores, agg)
    write("unqover.tsv", unq)
    n = lambda rows, p: sum(1 for r in rows if r[1] == p)
    manifest = {
        "socialsent": {"source": f"shifterator@{SHIFTERATOR_SHA} SocialSent-Reddit", "subreddit_lexicons": len(lexicons),
                       "words_in_>=40_lexicons": len(agg), "ethnic_terms_removed": removed,
                       "pos": n(ss, "pos"), "neg": n(ss, "neg")},
        "vader": {"source": f"vaderSentiment@{VADER_SHA}", "threshold": VADER_THRESHOLD,
                  "pos": n(vader, "pos"), "neg": n(vader, "neg")},
        "unqover": {"source": "third_party/unqover word_lists/activities/biased_ethnicity (our reconstruction, "
                              "sign-checked against VADER / SocialSent aggregate)",
                    "pos": n(unq, "pos"), "neg": n(unq, "neg")},
    }
    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
