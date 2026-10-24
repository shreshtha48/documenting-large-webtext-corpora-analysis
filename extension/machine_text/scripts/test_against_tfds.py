"""Differential test: filter_c4_pipeline.py vs TFDS v3.0.1 c4_utils.py on real WET pages.

Usage: python test_against_tfds.py <file.warc.wet.gz> <num_pages>
Runs the reference code (TF/Beam stubbed out) and this pipeline on the same pages and checks
that C4.EN and C4.BLOCKLIST come out identical (URL and text).
"""
import gzip, io, os, sys, types, importlib.util

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WET = sys.argv[1]
N = int(sys.argv[2])

# --- stub tensorflow / tfds so the reference module imports ---
import nltk, langdetect
from nltk.tokenize.punkt import PunktTokenizer
tf = types.ModuleType("tensorflow"); compat = types.ModuleType("compat"); v2 = types.ModuleType("v2")
v2.compat = types.SimpleNamespace(as_text=lambda x: x.decode() if isinstance(x, bytes) else x)
tf.compat = compat; compat.v2 = v2
sys.modules.update({"tensorflow": tf, "tensorflow.compat": compat, "tensorflow.compat.v2": v2})
class _C:
    def inc(self, *a): pass
beam = types.SimpleNamespace(metrics=types.SimpleNamespace(Metrics=types.SimpleNamespace(counter=lambda *a: _C())))
nltk_stub = types.SimpleNamespace(download=lambda *a, **k: None,
                                  data=types.SimpleNamespace(load=lambda *a: PunktTokenizer("english")))
tfds = types.ModuleType("tensorflow_datasets"); pub = types.ModuleType("public_api")
pub.core = types.SimpleNamespace(lazy_imports=types.SimpleNamespace(apache_beam=beam, nltk=nltk_stub, langdetect=langdetect))
tfds.public_api = pub
sys.modules.update({"tensorflow_datasets": tfds, "tensorflow_datasets.public_api": pub})
absl = types.ModuleType("absl"); absl.logging = types.SimpleNamespace(info=lambda *a: None)
sys.modules.update({"absl": absl, "absl.logging": absl.logging})

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
ref = load("ref", REPO + "/research-empirical-methods/reference-code/tfds_c4/c4_utils_v3.0.1_f411fae.py")
mine = load("mine", REPO + "/extension/machine_text/scripts/filter_c4_pipeline.py")
tok = mine.get_sentence_tokenizer()

# --- parse: compare WET splitters ---
with gzip.open(WET, "rt", encoding="utf-8") as f:
    my_pages = []
    for i, p in enumerate(mine.split_wet_stream(f)):
        if i >= N: break
        my_pages.append(p)
ref_pages = []
# reference split_wet_file uses tf.io.gfile; emulate by patching GFile
v2.io = types.SimpleNamespace(gfile=types.SimpleNamespace(GFile=lambda p, m: open(p, m)))
for i, p in enumerate(ref.split_wet_file(WET, counter_inc_fn=lambda *a: None)):
    if i >= N: break
    ref_pages.append(p)
assert len(my_pages) == len(ref_pages), (len(my_pages), len(ref_pages))
assert all(a[0] == b[0] and a[1]["text"] == b[1]["text"] and a[1]["timestamp"] == b[1]["timestamp"]
           for a, b in zip(my_pages, ref_pages)), "WET split mismatch"
print(f"split_wet: {len(my_pages)} pages identical")

# --- reference pipeline: length -> url dedupe -> clean -> line dedupe -> langdetect -> badwords ---
pages = [p for p in ref_pages if ref.is_valid_length(p)]
groups = {}
for url, feats in pages:
    groups.setdefault(ref.normalize_url((url, feats))[0], []).append(feats)
pages = [ref.dedupe_urls((u, ps)) for u, ps in groups.items()]
cre = __import__("re").compile(r"\[\d*\]|\[edit\]|\[citation needed\]")
cleaned = []
for p in pages:
    cleaned += list(ref.clean_page((p[0], dict(p[1])), citation_regex=cre, counter_inc_fn=lambda *a: None))
owner = {}
for p in cleaned:
    for h, u in ref._emit_url_to_lines(p):
        if h not in owner or ref._hash_text(u) < ref._hash_text(owner[h]): owner[h] = u
keep = {}
for h, u in owner.items(): keep.setdefault(u, []).append(h)
deduped = []
for u, feats in cleaned:
    deduped += list(ref._remove_lines_from_text((u, {"features": [feats], "lines": keep.get(u, [])}),
                                                counter_inc_fn=lambda *a: None, min_num_sentences=3))
english = []
for p in deduped: english += list(ref.detect_english(p))
words = [l.strip() for l in open(REPO + "/research-empirical-methods/reference-code/tfds_c4/ldnoobw_en_25e679f.txt", encoding="utf-8")]
bf = ref.get_badwords_filter_fn({"en": set(words)})
ref_en = {f["url"]: f["text"] for _, f in english if bf((None, f))}
ref_block = {f["url"]: f["text"] for _, f in english if not bf((None, f))}
print(f"reference: after clean={len(cleaned)} dedupe={len(deduped)} english={len(english)} en={len(ref_en)} blocklist={len(ref_block)}")

# --- mine ---
import tempfile, json
out = tempfile.mkdtemp()
mine.run_pipeline(source="local", local_file=WET, max_docs=N, out_dir=out, shard_idx=0)
def read(path): return {json.loads(l)["url"]: json.loads(l)["text"] for l in gzip.open(path, "rt", encoding="utf-8")}
my_en = read(os.path.join(out, "c4_en_local_shard0.jsonl.gz"))
my_block = read(os.path.join(out, "c4_blocklist_local_shard0.jsonl.gz"))
print(f"mine: en={len(my_en)} blocklist={len(my_block)}")
print("C4.EN identical:", my_en == ref_en)
print("BLOCKLIST identical:", my_block == ref_block)
for name, a, b in (("en", my_en, ref_en), ("block", my_block, ref_block)):
    if a != b:
        print(name, "only mine:", list(set(a) - set(b))[:5], "only ref:", list(set(b) - set(a))[:5],
              "text diffs:", [u for u in set(a) & set(b) if a[u] != b[u]][:5])
