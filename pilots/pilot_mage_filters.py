"""Pilot: do C4's heuristic filters keep machine-generated text at a different rate than human text?
Data: MAGE test split (yaful/MAGE), balanced human/machine per domain."""
import csv,sys,json,math,collections,random
import nltk; nltk.download("punkt_tab",quiet=True)
sys.path.insert(0,sys.argv[3]); from c4_filters_v301 import clean_page, load_badwords
csv.field_size_limit(10**8)
rows=list(csv.DictReader(open(sys.argv[1],encoding="utf-8-sig")))
bad=load_badwords(sys.argv[2])
def wilson(k,n,z=1.96):
    if n==0: return (float("nan"),)*2
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)
modern=("gpt-3.5-trubo","text-davinci-003")
stats=collections.defaultdict(lambda:collections.Counter())
for r in rows:
    dom=r["src"].split("_")[0]; human=r["label"]=="1"
    groups=[("human" if human else "machine_all")]
    if not human and r["src"].endswith(modern): groups.append("machine_modern")
    cleaned,why=clean_page(r["text"])
    kept = cleaned is not None and bad.search(cleaned.lower()) is None
    nw=len(r["text"].split()); lb="len<100" if nw<100 else ("len100-250" if nw<250 else "len>=250")
    for g in groups:
        for key in [(dom,g),("ALL",g),(lb,g)]:
            stats[key]["n"]+=1; stats[key]["kept"]+=kept; stats[key][why]+=1
            if cleaned is not None and not kept: stats[key]["blocklist"]+=1
out={}
for (k,g),c in sorted(stats.items()):
    lo,hi=wilson(c["kept"],c["n"]); out[f"{k}|{g}"]={"n":c["n"],"kept":c["kept"],"rate":round(c["kept"]/c["n"],4),"ci95":[round(lo,4),round(hi,4)],
      "too_few_sentences":c["too_few_sentences"],"curly":c["curly"],"blocklist":c["blocklist"]}
json.dump(out,open("pilot_mage_results.json","w"),indent=1)
for k,v in out.items():
    if k.split("|")[0] in ("ALL","len<100","len100-250","len>=250") or True:
        print(f"{k:40s} n={v['n']:5d} kept={v['rate']:.3f} CI{v['ci95']} fewsent={v['too_few_sentences']} curly={v['curly']} block={v['blocklist']}")
