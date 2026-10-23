"""Pilot: exact-match presence of safety/bias benchmark items in C4.EN (train) via the Infini-gram API.
Counts are case- and punctuation-sensitive (Llama-2 tokens), i.e. an un-normalised lower bound."""
import json, random, time, urllib.request, urllib.parse, sys
random.seed(0)
API="https://api.infini-gram.io/"
def http(url, data=None):
    for a in range(6):
        try:
            req=urllib.request.Request(url, data=json.dumps(data).encode() if data else None, headers={"Content-Type":"application/json"})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            time.sleep(3*(a+1))
    return None
def count(q):
    r=http(API, {"index":"v4_c4train_llama","query_type":"count","query":q}); time.sleep(0.15)
    return None if r is None or "count" not in r else r["count"]
def nrows(ds,cfg,split):
    r=http(f"https://datasets-server.huggingface.co/size?dataset={urllib.parse.quote(ds)}")
    for s in r["size"]["splits"]:
        if s["config"]==cfg and s["split"]==split: return s["num_rows"]
def sample_rows(ds,cfg,split,n,N):
    """Union of up to 5 random 100-row blocks, then a uniform sample of N rows from the union."""
    pool=[]; starts=sorted(set(random.randrange(0,max(1,n-100)) for _ in range(5))) if n>500 else list(range(0,n,100))
    for s in starts:
        r=http(f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds)}&config={cfg}&split={split}&offset={s}&length=100")
        if r and "rows" in r: pool+= [x["row"] for x in r["rows"]]
        time.sleep(1.0)
    return random.sample(pool,min(N,len(pool)))
def fix(s):
    try: return s.encode("cp1252").decode("utf-8")
    except Exception: return s
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
specs=[
 ("TruthfulQA question","truthfulqa/truthful_qa","generation","validation",lambda r:r["question"]),
 ("TruthfulQA best_answer","truthfulqa/truthful_qa","generation","validation",lambda r:r["best_answer"]),
 ("BOLD wikipedia sentence","AmazonScience/bold","default","train",lambda r:r["wikipedia"][0].strip()),
 ("BOLD prompt","AmazonScience/bold","default","train",lambda r:r["prompts"][0].strip()),
 ("RTP prompt+continuation","allenai/real-toxicity-prompts","default","train",lambda r:fix(r["prompt"]["text"]+r["continuation"]["text"]).strip()),
 ("RTP prompt","allenai/real-toxicity-prompts","default","train",lambda r:fix(r["prompt"]["text"]).strip()),
 ("BBQ context (Race_ethnicity)","heegyu/bbq","Race_ethnicity","test",lambda r:r["context"]),
 ("ToxiGen text (annotated test)","toxigen/toxigen-data","annotated","test",lambda r:r["text"]),
 ("StereoSet intersentence context","McGill-NLP/stereoset","intersentence","validation",lambda r:r["context"]),
]
xsum=["The takeover of Bradford Bulls by Omar Khan's consortium has been ratified by the Rugby Football League.",
"Two men who were sued over the Omagh bomb have been found liable for the 1998 atrocity at their civil retrial.",
"Grimsby fought back from two goals down to beat Aldershot and boost their National League play-off hopes.",
"Doctors say a potential treatment for peanut allergy has transformed the lives of children taking part in a large clinical trial.",
"A breast surgeon who intentionally wounded his patients has had his 15-year jail term increased to 20 years.",
"Peterborough United have signed forward Danny Lloyd on a free transfer from National League North side Stockport.",
"The first major trial to see if losing weight reduces the risk of cancers coming back is about to start in the US and Canada.",
"Villarreal central defender Eric Bailly is set to be Jose Mourinho's first signing as Manchester United manager."]
out={"meta":{"index":"v4_c4train_llama","N":N,"seed":0},"results":{}}
res=[{"text":t,"count":count(t)} for t in xsum]
out["results"]["POSITIVE CONTROL: Dodge Table 5 XSum"]=res
print("XSum control", sum(1 for r in res if r["count"]), "/", len(res), flush=True)
cache={}
for name,ds,cfg,split,get in specs:
    key=(ds,cfg,split)
    if key not in cache:
        n=nrows(ds,cfg,split); cache[key]=(n,sample_rows(ds,cfg,split,n,N))
    n,rows=cache[key]
    res=[]
    for r in rows:
        t=get(r); res.append({"text":t,"count":count(t)})
    hit=sum(1 for r in res if r["count"]); ok=sum(1 for r in res if r["count"] is not None)
    out["results"][name]={"dataset_rows":n,"items":res}
    print(f"{name}: {hit}/{ok} found (dataset size {n})", flush=True)
json.dump(out,open(sys.argv[2] if len(sys.argv)>2 else "pilot_results.json","w"),indent=1,ensure_ascii=False)
