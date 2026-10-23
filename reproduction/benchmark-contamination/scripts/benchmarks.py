"""Dodge et al. 2021 §4.2 / Table 4: the benchmark texts whose presence in C4.EN is measured.

Each row of Table 4 becomes one "field": a list of texts. This script downloads the sources and writes
data/benchmarks/<field>.jsonl.gz with one {"id", "text"} per line.

Sources and choices (the paper names the datasets and splits only):
- LAMA T-REx (34,039 facts; paper 34,014) and Google-RE (5,528; paper 5,528): the paper's "template-generated
  sentences with a masked token that we fill in". Templates from LAMA's relations.jsonl (T-REx) and LAMA's
  run_experiments.py (Google-RE), filled with the subject and object labels. Source: dl.fbaipublicfiles.com/LAMA.
- XSum test summaries (11,334; paper 11,334): EdinburghNLP/xsum.
- TIFU-short (79,740) and TIFU-long (42,139): the Reddit TIFU post titles and TL;DRs, as the Hugging Face
  ctr4si/reddit_tifu "short" / "long" configs define them (original casing).
- WikiBio test targets (72,831; paper 72,831): michaelauli/wiki_bio, the full biography text. ONLY lowercased,
  tokenized text exists ("-lrb-" for "("), so an exact, case-sensitive index cannot test it fairly. Built only with
  --wikibio and reported as not comparable.
- GLUE test inputs: nyu-mll/glue (MNLI: the mismatched test set, 9,847 rows, which matches the paper's count).
- BoolQ test (3,245; paper 3,245): aps/super_glue "boolq". The paper does not say which field; both the question
  and the passage are built.
- AMR-to-text (LDC2017T10) is licensed and not freely available: not built.

Usage: python scripts/benchmarks.py [--wikibio]
"""
import argparse, gzip, json, os, re, urllib.request, zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "benchmarks")
CACHE = os.path.join(ROOT, "data", "downloads")

LAMA_ZIP = "https://dl.fbaipublicfiles.com/LAMA/data.zip"
TIFU = "https://huggingface.co/datasets/ctr4si/reddit_tifu/resolve/main/data/tifu_all_tokenized_and_filtered.json.gz"
WIKIBIO = "https://huggingface.co/datasets/michaelauli/wiki_bio/resolve/main/data/wikipedia-biography-dataset.zip"
GOOGLE_RE_TEMPLATES = {"place_of_birth": "[X] was born in [Y] .", "date_of_birth": "[X] (born [Y]).",
                       "place_of_death": "[X] died in [Y] ."}   # LAMA scripts/run_experiments.py

# Table 4 of the paper: field -> (dataset, part, matched, total)
PAPER = {
    "lama_trex": ("LAMA T-REx", "label", 1585, 34014), "lama_google_re": ("LAMA Google-RE", "label", 314, 5528),
    "xsum": ("XSum", "label", 1756, 11334), "tifu_short": ("TIFU-short", "label", 19843, 79740),
    "tifu_long": ("TIFU-long", "label", 790, 42139), "wikibio": ("WikiBio", "label", 2712, 72831),
    "boolq": ("BoolQ", "input", 79, 3245), "cola": ("CoLA", "input", 153, 1063),
    "mnli_hypothesis": ("MNLI hypothesis", "input", 1402, 9847), "mnli_premise": ("MNLI premise", "input", 1494, 9847),
    "mrpc_sentence1": ("MRPC sentence 1", "input", 46, 1725), "mrpc_sentence2": ("MRPC sentence 2", "input", 46, 1725),
    "qnli_sentence": ("QNLI sentence", "input", 2931, 5463), "qnli_question": ("QNLI question", "input", 97, 5463),
    "rte_sentence1": ("RTE sentence 1", "input", 179, 3000), "rte_sentence2": ("RTE sentence 2", "input", 325, 3000),
    "sst2": ("SST-2", "input", 200, 1821), "stsb_sentence1": ("STS-B sentence 1", "input", 253, 1379),
    "stsb_sentence2": ("STS-B sentence 2", "input", 256, 1379), "wnli_sentence1": ("WNLI sentence 1", "input", 7, 146),
    "wnli_sentence2": ("WNLI sentence 2", "input", 3, 146),
}
# our extra field (the paper does not say which BoolQ field it used)
EXTRA = {"boolq_passage": "boolq"}
GLUE = {"cola": ("cola", "test", ["sentence"]), "mnli": ("mnli", "test_mismatched", ["hypothesis", "premise"]),
        "mrpc": ("mrpc", "test", ["sentence1", "sentence2"]), "qnli": ("qnli", "test", ["sentence", "question"]),
        "rte": ("rte", "test", ["sentence1", "sentence2"]), "sst2": ("sst2", "test", ["sentence"]),
        "stsb": ("stsb", "test", ["sentence1", "sentence2"]), "wnli": ("wnli", "test", ["sentence1", "sentence2"])}


def fetch(url, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print(f"  downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def hf_parquet(dataset, config, split):
    """A split from the Hugging Face dataset viewer's parquet export."""
    with urllib.request.urlopen(f"https://datasets-server.huggingface.co/parquet?dataset={dataset}", timeout=60) as r:
        files = [f for f in json.load(r)["parquet_files"] if f["config"] == config and f["split"] == split]
    return pd.concat([pd.read_parquet(fetch(f["url"], f"{dataset.replace('/', '_')}_{config}_{split}_{i}.parquet"))
                      for i, f in enumerate(files)], ignore_index=True)


def write(field, texts):
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, f"{field}.jsonl.gz"), "wt", encoding="utf-8") as g:
        for i, t in enumerate(texts):
            g.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")
    print(f"{field}: {len(texts)} texts", flush=True)


def fill(template, x, y):
    s = template.replace("[X]", x).replace("[Y]", y)
    return re.sub(r"\s+([.,;:!?)])", r"\1", s)        # LAMA templates put a space before the final period


def lama():
    z = zipfile.ZipFile(fetch(LAMA_ZIP, "lama_data.zip"))
    rel = {r["relation"]: r["template"] for r in map(json.loads, z.read("data/relations.jsonl").decode().splitlines())}
    trex = []
    for n in sorted(x for x in z.namelist() if "/TREx/" in x and x.endswith(".jsonl")):
        p = os.path.basename(n)[:-6]
        trex += [fill(rel[p], d["sub_label"], d["obj_label"]) for d in map(json.loads, z.read(n).decode().splitlines())]
    write("lama_trex", trex)
    gre = []
    for r, t in GOOGLE_RE_TEMPLATES.items():
        gre += [fill(t, d["sub_label"], d["obj_label"])
                for d in map(json.loads, z.read(f"data/Google_RE/{r}_test.jsonl").decode().splitlines())]
    write("lama_google_re", gre)


def recase(low, original):
    """The span of `original` that equals the lowercased `low`, in its original case (None if not found)."""
    low, o = low.strip(), original or ""
    if not low or len(o.lower()) != len(o):
        return None
    i = o.lower().find(low)
    return o[i:i + len(low)] if i >= 0 else None


def tifu():
    """As the Hugging Face loader (v1.1.0): summary = trimmed_title (short) or tldr (long); posts with an empty body or
    summary are skipped, which gives the paper's 79,740 / 42,139. Both summaries are stored lowercased, so each is
    located in the original title / post body and taken back in its original case (found for 100% of rows)."""
    short, long_ = [], []
    with gzip.open(fetch(TIFU, "tifu_all_tokenized_and_filtered.json.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if not (d.get("selftext_without_tldr") or "").strip():
                continue
            if d["trimmed_title"].strip():
                short.append(recase(d["trimmed_title"], d["title"]) or d["trimmed_title"].strip())
            if (d.get("tldr") or "").strip():
                long_.append(recase(d["tldr"], d.get("selftext")) or d["tldr"].strip())
    write("tifu_short", short)
    write("tifu_long", long_)


def wikibio():
    z = zipfile.ZipFile(fetch(WIKIBIO, "wikipedia-biography-dataset.zip"))
    base = next(n for n in z.namelist() if n.endswith("test/test.nb")).rsplit("/", 1)[0]
    nb = [int(x) for x in z.read(f"{base}/test.nb").decode().split()]
    sents = z.read(f"{base}/test.sent").decode("utf-8").splitlines()
    out, k = [], 0
    for n in nb:
        out.append(" ".join(sents[k:k + n]).strip())
        k += n
    write("wikibio", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikibio", action="store_true", help="also build WikiBio (334 MB download; lowercased text)")
    args = ap.parse_args()
    lama()
    write("xsum", hf_parquet("EdinburghNLP/xsum", "default", "test")["summary"].str.strip().tolist())
    tifu()
    bq = hf_parquet("aps/super_glue", "boolq", "test")
    write("boolq", bq["question"].str.strip().tolist())
    write("boolq_passage", bq["passage"].str.strip().tolist())
    for name, (cfg, split, cols) in GLUE.items():
        df = hf_parquet("nyu-mll/glue", cfg, split)
        for c in cols:
            write(name if len(cols) == 1 else f"{name}_{c}", df[c].str.strip().tolist())
    if args.wikibio:
        wikibio()


if __name__ == "__main__":
    main()
