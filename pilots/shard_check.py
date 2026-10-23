"""One-shard reproduction checks for Dodge et al. (2021) F5, plus sample-size counts.
Usage: python shard_check.py en_shard.json.gz noblocklist_shard.json.gz ldnoobw_en.txt
1) Blocklist correctness: en == noblocklist minus docs matching the TFDS v3.0.1 badwords regex (by URL).
2) Identity-term PMI with the blocklist (Dodge Table 6 regexes), natural log, document level, bootstrap CI.
3) Counts relevant to power: Jewish/Arab paragraphs (Dodge A.7 matching), gender, teen terms, patents.google.com."""
import gzip, json, re, sys, math, random, collections
en_path, nb_path, bw_path = sys.argv[1:4]
words=[w.strip() for w in open(bw_path,encoding="utf-8") if w.strip()]
BAD=re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))
IDS={k:re.compile(r"\b(?:"+v+r")\b",re.I) for k,v in {
 "homosexual":r"homosexuals?","gay":r"gays?","nonbinary":r"non[ -]?binary","trans":r"trans(?:|\+|gender)","lesbian":r"lesbians?",
 "black":r"blacks?","african american":r"african[ -]americans?","latinx":r"latin[oax]s?","asian":r"asian(?:[ -]american)?s?",
 "muslim":r"muslims?","jew":r"jew(?:|s|ish)","woman":r"wom[ae]n","female":r"females?","man":r"m[ae]n","male":r"males?",
 "straight":r"straights?","heterosexual":r"heterosexuals?","bisexual":r"bi-?sexuals?","white":r"whites?","caucasian":r"caucasians?",
 "european":r"european(?:[ -]american)?s?","christian":r"christians?",
 # extension terms (not in Dodge Table 6)
 "teen":r"teens?|teenagers?|adolescents?","girl":r"girls?","boy":r"boys?"}.items()}
def docs(p):
    with gzip.open(p,"rt",encoding="utf-8") as f:
        for line in f: yield json.loads(line)
en_urls=set(d["url"] for d in docs(en_path))
n=0; filt=0; mismatch=collections.Counter(); id_n=collections.Counter(); id_f=collections.Counter()
para=collections.Counter(); patents=0; per_doc=[]
for d in docs(nb_path):
    n+=1; t=d["text"]; is_f = BAD.search(t.lower()) is not None
    filt+=is_f
    in_en = d["url"] in en_urls
    mismatch[("filtered" if is_f else "kept", "in_en" if in_en else "not_in_en")]+=1
    hits=[k for k,r in IDS.items() if r.search(t)]
    for k in hits: id_n[k]+=1; id_f[k]+=is_f
    per_doc.append((is_f,hits))
    if "patents.google.com" in d["url"]: patents+=1
    if not is_f:
        for p in t.split("\n"):
            if " Jewish " in p: para["Jewish (kept docs)"]+=1
            if " Arab " in p: para["Arab (kept docs)"]+=1
print(f"noblocklist docs: {n}; en docs: {len(en_urls)}; regex-filtered: {filt} ({filt/n:.4%})")
print("agreement table (regex decision x presence in en):", dict(mismatch))
agree=(mismatch[("kept","in_en")]+mismatch[("filtered","not_in_en")])/n
print(f"agreement with released en: {agree:.4%}")
P_f=filt/n
def pmi(k):
    a=id_f[k]; b=id_n[k]
    return float("nan") if a==0 else math.log((a/n)/((b/n)*P_f))
random.seed(0); B=200; boots=collections.defaultdict(list)
for _ in range(B):
    s=[per_doc[random.randrange(n)] for _ in range(n)]
    nf=sum(x[0] for x in s); cn=collections.Counter(); cf=collections.Counter()
    for is_f,h in s:
        for k in h: cn[k]+=1; cf[k]+=is_f
    for k in IDS:
        if cf[k]>0: boots[k].append(math.log((cf[k]/n)/((cn[k]/n)*(nf/n))))
print("\nterm, docs_with_term, filtered, filter_rate, PMI(ln) [95% bootstrap CI, doc-level, B=200]")
for k in sorted(IDS,key=lambda k:-(pmi(k) if id_f[k] else -9)):
    bs=sorted(boots[k]); ci=(bs[int(.025*len(bs))],bs[int(.975*len(bs))-1]) if len(bs)>10 else (float('nan'),)*2
    print(f"{k:18s} {id_n[k]:7d} {id_f[k]:6d} {id_f[k]/max(1,id_n[k]):.3f} {pmi(k):6.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
print("\nparagraph counts (Dodge A.7 matching: word surrounded by spaces):",dict(para))
print("patents.google.com docs in noblocklist shard:",patents)
