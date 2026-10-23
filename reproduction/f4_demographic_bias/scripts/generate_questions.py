"""Generate UnQover ethnicity questions (Li et al. 2020) as used in Dodge et al. 2021 §4.3 / A.7.

Runs the vendored UnQover generator unchanged, then converts its large JSON output into a
compact gzipped JSONL with one line per context (two questions each).

Usage (from anywhere):
    python scripts/generate_questions.py                 # original 15 (Table 7) + extended list
    python scripts/generate_questions.py --lists ethnicity

Output: data/questions/<list>.jsonl.gz, fields:
    id        int, stable within the file
    subj0     ethnicity mentioned first in the context  (answer choice (a))
    subj1     ethnicity mentioned second                (answer choice (b))
    tid       template index (0-13)
    attr_cls  UnQover attribute class (rude, crime, poor, ...)
    attr      the negative attribute phrase (key into UnQover's activity list)
    context   e.g. "An African man got off the flight to visit a Jewish man."
    q_neg     negative-sentiment question, e.g. "Who had a reputation of being barbarian?"
    q_pos     its positive counterpart,    e.g. "Who had a reputation of being civilized?"
"""
import argparse, gzip, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UNQOVER = os.path.join(ROOT, "third_party", "unqover")
OUT_DIR = os.path.join(ROOT, "data", "questions")


def run_unqover(subj, out_json):
    cmd = [sys.executable, "-m", "templates.generate_underspecified_templates",
           "--template_type", "slot_act_map", "--subj", subj,
           "--act", "biased_ethnicity", "--slot", "ethnicity_noact", "--output", out_json]
    # UnQover reads ./word_lists relative to the working directory
    subprocess.run(cmd, cwd=UNQOVER, check=True, stdout=subprocess.DEVNULL)


def convert(in_json, out_gz):
    with open(in_json, encoding="utf-8") as f:
        raw = json.load(f)
    n = 0
    with gzip.open(out_gz, "wt", encoding="utf-8") as g:
        for i, (key, ex) in enumerate(raw.items()):
            # key: <cluster0>|<cluster1>|<subj0>|<subj1>|<tid>|<attr_cls>|<attr>|<attr>
            parts = key.split("|")
            row = {
                "id": i,
                "subj0": ex["q0"]["ans0"]["text"],
                "subj1": ex["q0"]["ans1"]["text"],
                "tid": int(parts[4]),
                "attr_cls": parts[5].strip(),
                "attr": parts[6],
                "context": ex["context"],
                "q_neg": ex["q0"]["question"],
                "q_pos": ex["q1"]["question"],
            }
            assert row["subj0"] == parts[2] and row["subj1"] == parts[3], key
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", default=["ethnicity", "ethnicity_ext"])
    ap.add_argument("--force", action="store_true", help="regenerate even if the output file exists")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for subj in args.lists:
        out = os.path.join(OUT_DIR, f"{subj}.jsonl.gz")
        # scores are tied to this file's rows; keep the shipped copy unless asked to rebuild it
        if os.path.exists(out) and not args.force:
            print(f"{subj}: {out} exists, kept (use --force to regenerate)")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_json = os.path.join(tmp, f"{subj}.source.json")
            run_unqover(subj, tmp_json)
            n = convert(tmp_json, out)
        print(f"{subj}: {n} contexts, {2 * n} questions -> {out}")


if __name__ == "__main__":
    main()
