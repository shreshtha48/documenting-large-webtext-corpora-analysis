"""Extension, probe: download BBQ (Parrish et al., Findings of ACL 2022) for the gender / age / race x gender arm.

BBQ is the published QA bias benchmark that covers what UnQover does not: gender identity, age, and intersectional
race x gender, with author-written, literature-sourced stereotypes. Each example has an ambiguous or disambiguated
context, a negative or non-negative question, and three answers (two people and an UNKNOWN option).

Pinned to nyu-mll/BBQ @ bea11bd (CC-BY 4.0). Downloads, for --categories (default Age, Gender_identity,
Race_x_gender):
    data/<category>.jsonl                       the examples
    analysis_scripts/additional_metadata.csv    target_loc (index of the bias-target answer), label_type
                                                (identity label vs proper name), full_cond for intersectional items
    results/UnifiedQA/preds_<category>.jsonl    the authors' UnifiedQA-11B predictions (RACE / ARC / question-only
                                                formats), used to check that analyze_bbq.py reproduces the paper

Writes data/bbq/<category>.jsonl.gz (examples merged with the metadata; the 16 examples without target_loc are
dropped, as in the official R script) and data/bbq/preds11b_<category>.jsonl.gz.

Usage: python scripts/prepare_bbq.py
"""
import argparse, csv, gzip, io, json, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "bbq")
SHA = "bea11bd97d79217245b5871acd247b9d6eb24598"
BASE = f"https://raw.githubusercontent.com/nyu-mll/BBQ/{SHA}/"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return r.read().decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["Age", "Gender_identity", "Race_x_gender"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for r in csv.DictReader(io.StringIO(get("analysis_scripts/additional_metadata.csv"))):
        meta[(r["category"], int(r["example_id"]))] = r
    for cat in args.categories:
        out = os.path.join(OUT, f"{cat}.jsonl.gz")
        n = dropped = 0
        with gzip.open(out + ".tmp", "wt", encoding="utf-8") as g:
            for line in get(f"data/{cat}.jsonl").splitlines():
                r = json.loads(line)
                m = meta.get((cat, r["example_id"]))
                if m is None or m["target_loc"] in ("", "NA"):
                    dropped += 1
                    continue
                r.update(id=f"{cat}:{r['example_id']}", target_loc=int(m["target_loc"]), label_type=m["label_type"],
                         full_cond=m["full_cond"].replace("\n", "").replace("  ", " ") if m["full_cond"] != "NA" else "")
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        os.replace(out + ".tmp", out)
        with gzip.open(os.path.join(OUT, f"preds11b_{cat}.jsonl.gz"), "wt", encoding="utf-8") as g:
            for line in get(f"results/UnifiedQA/preds_{cat}.jsonl").splitlines():
                r = json.loads(line)
                g.write(json.dumps({"id": f"{cat}:{r['example_id']}", **{k: v for k, v in r.items() if k.startswith("unifiedqa")}}) + "\n")
        print(f"{cat}: {n} examples ({dropped} without target_loc dropped) -> {out}")


if __name__ == "__main__":
    main()
