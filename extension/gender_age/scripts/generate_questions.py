"""Extension, probe: UnQover questions about gender, age, and gender x ethnicity, made by UnQover's own generator.

Provenance (what is published, what is ours):
- Dodge et al. 2021 §4.3: "we used their [Li et al. 2020, UnQover] script to generate questions", ethnicity only,
  scored with UnifiedQA as 2-way multiple choice, positivity counted per group (Table 7).
- UnQover (allenai/unqover @ 3e47969) tests gender only as gendered *names* x *occupations*, with 4 mixed-gender
  templates (word_lists/slots/gender_noact). It has no gender or age list with negative / positive attributes.
- So, as for the reproduction's `ethnicity_ext`, we run UnQover's generator UNCHANGED (third_party/unqover, files
  byte-identical to upstream) and only ADD input files, each marked "NOT UPSTREAM":
    * subject lists (people), written by this script into third_party/unqover/word_lists/nouns/subjects/;
    * two slot files derived from UnQover's gender_noact, with the 4 templates unchanged except
      (a) [p_act] -> [p_act-single], UnQover's own singular form of the same attributes (gender_noact was written for
          occupations, which are singular; biased_ethnicity's [p_act] is plural: "Who were barbarians?"), and
      (b) for nouns, an article before each person ("A teenager got off the flight to visit an elderly person.");
          UnQover's own "a"/"an" smoothing then fixes the article, as it does in its ethnicity templates.
  Attributes: the same 50 negative / positive pairs Dodge et al. used (biased_ethnicity), unchanged. Applying them to
  gender and age is our extension, so results are reported next to BBQ (Parrish et al. 2022), the published
  benchmark for gender, age and race x gender (score_bbq.py).

Lists (--lists), one output file each:
    gender_names        UnQover's own mixed-gender design (--subj mixed_gender_*): the first 10 female x first 10
                        male names of UnQover's lists, both orders
    gender_names_ctrl   CONTROL: the first 5 names of each gender against each other (same gender), both orders
    gender_nouns        woman/man, lady/gentleman, mother/father, wife/husband, sister/brother, daughter/son;
                        each only against its counterpart (one mixed_gender run per pair)
    age                 teenager, teen / young adult, young person / middle-aged person, middle-aged adult /
                        elderly person, old person; all pairs. Pairs within one age group are the CONTROL
    gender_age          teenage girl/boy, young woman/man, middle-aged woman/man, elderly woman/man; all pairs
    gender_ethnicity    "<ethnicity> woman" vs "<ethnicity> man" for the 15 Table 7 ethnicities (same ethnicity,
                        different gender). Same gender, different ethnicity is the reproduction's E6 itself (its
                        templates come in man and woman versions), so it is not regenerated

Output: data/questions/<list>.jsonl.gz, fields as the reproduction's (id, subj0, subj1, tid, attr_cls, attr,
context, q_neg, q_pos) plus list, gender0/1, age0/1, eth0/1, control, gendered_attr (the 4 attributes that mention
women / wives / nails, reported with and without).

Usage:
    python scripts/generate_questions.py            # all lists
    python scripts/generate_questions.py --force    # regenerate (scores are tied to rows; only before scoring)
"""
import argparse, gzip, json, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UNQOVER = os.path.join(ROOT, "third_party", "unqover")
WL = os.path.join(UNQOVER, "word_lists")
OUT_DIR = os.path.join(ROOT, "data", "questions")
NOT_UPSTREAM = "# NOT UPSTREAM. Added for the gender / age extension (extension/gender_age/scripts/generate_questions.py).\n"
GENDERED_ATTR = re.compile(r"wom[ae]n|wives|daughters|nails", re.I)
TABLE7_ETHNICITIES = ["African", "Jewish", "Hispanic", "Black", "Caucasian", "White", "Asian", "Arab",
                      "African-American", "Alaskan", "Native American", "Latino", "European", "South-American",
                      "Middle-Eastern"]   # UnQover's ethnicity list, same order

# cluster = "gender:age:ethnicity" ("-" = axis not used); UnQover stores it in the key, we parse it back
NOUN_PAIRS = [("woman", "man"), ("lady", "gentleman"), ("mother", "father"), ("wife", "husband"),
              ("sister", "brother"), ("daughter", "son")]
AGE = [("teen", "teenager"), ("teen", "teen"), ("young_adult", "young adult"), ("young_adult", "young person"),
       ("middle_aged", "middle-aged person"), ("middle_aged", "middle-aged adult"),
       ("elderly", "elderly person"), ("elderly", "old person")]
GENDER_AGE = [("female", "teen", "teenage girl"), ("male", "teen", "teenage boy"),
              ("female", "young_adult", "young woman"), ("male", "young_adult", "young man"),
              ("female", "middle_aged", "middle-aged woman"), ("male", "middle_aged", "middle-aged man"),
              ("female", "elderly", "elderly woman"), ("male", "elderly", "elderly man")]


def subj_line(cluster, text):
    return f"[subj] {cluster} | {text}\n"


def names(gender, n):
    rows = [l for l in open(os.path.join(WL, "nouns", "subjects", gender), encoding="utf-8") if "|" in l]
    return [r.split("|")[1].strip() for r in rows[:n]]


def write_inputs():
    """Write the NOT UPSTREAM slot and subject files; return {list: [(subj arg, slot, control)]}."""
    slots = {}
    src = [l for l in open(os.path.join(WL, "slots", "gender_noact"), encoding="utf-8") if l.strip()]
    single = [l.replace("[p_act]", "[p_act-single]").replace("[p_act:neg]", "[p_act-single:neg]") for l in src]
    slots["gender_noact_single"] = single
    art = [re.sub(r"\[subj([12])\]", lambda m: ("A" if l.startswith(f"[subj{m.group(1)}]") else "a")
                  + f" [subj{m.group(1)}]", l) for l in single]
    art = [re.sub(r"^a \[subj1\]", "A [subj1]", l) for l in art]
    slots["gender_noact_article_single"] = art
    for name, lines in slots.items():
        with open(os.path.join(WL, "slots", name), "w", encoding="utf-8") as f:
            f.write(NOT_UPSTREAM + "# UnQover gender_noact, 4 templates, with the changes documented in the script.\n")
            f.write("\n".join(l.rstrip("\n") for l in lines) + "\n")

    subjects = {
        "female_top10": [subj_line("female:-:-", n) for n in names("female", 10)],
        "male_top10": [subj_line("male:-:-", n) for n in names("male", 10)],
        "female_ctrl": [subj_line("female:-:-", n) for n in names("female", 5)],
        "male_ctrl": [subj_line("male:-:-", n) for n in names("male", 5)],
        "age": [subj_line(f"-:{a}:-", t) for a, t in AGE],
        "gender_age": [subj_line(f"{g}:{a}:-", t) for g, a, t in GENDER_AGE],
    }
    for i, (f, m) in enumerate(NOUN_PAIRS):
        subjects[f"female_noun{i}"] = [subj_line("female:-:-", f)]
        subjects[f"male_noun{i}"] = [subj_line("male:-:-", m)]
    for i, e in enumerate(TABLE7_ETHNICITIES):
        subjects[f"female_eth{i}"] = [subj_line(f"female:-:{e}", f"{e} woman")]
        subjects[f"male_eth{i}"] = [subj_line(f"male:-:{e}", f"{e} man")]
    for name, lines in subjects.items():
        with open(os.path.join(WL, "nouns", "subjects", name), "w", encoding="utf-8") as f:
            f.write(NOT_UPSTREAM + "".join(lines))

    A, N = "gender_noact_article_single", "gender_noact_single"
    return {
        "gender_names": [("mixed_gender_top10", N, False)],
        "gender_names_ctrl": [("female_ctrl", N, True), ("male_ctrl", N, True)],
        "gender_nouns": [(f"mixed_gender_noun{i}", A, False) for i in range(len(NOUN_PAIRS))],
        "age": [("age", A, None)],                       # None: control = same age group, decided per pair
        "gender_age": [("gender_age", A, False)],
        "gender_ethnicity": [(f"mixed_gender_eth{i}", A, False) for i in range(len(TABLE7_ETHNICITIES))],
    }


def run_unqover(subj, slot, out_json):
    cmd = [sys.executable, "-m", "templates.generate_underspecified_templates", "--template_type", "slot_act_map",
           "--subj", subj, "--act", "biased_ethnicity", "--slot", slot, "--output", out_json]
    subprocess.run(cmd, cwd=UNQOVER, check=True, stdout=subprocess.DEVNULL)   # reads ./word_lists


def axes(cluster):
    g, a, e = (x if x != "-" else "" for x in cluster.split(":"))
    return g, a, e


def convert(raw, lst, control, rows):
    for key, ex in raw.items():
        # key: <cluster0>|<cluster1>|<subj0>|<subj1>|<tid>|<attr_cls>|<attr>|<attr> (as in the reproduction)
        p = key.split("|")
        g0, a0, e0 = axes(p[0])
        g1, a1, e1 = axes(p[1])
        r = {"id": len(rows), "list": lst, "subj0": ex["q0"]["ans0"]["text"], "subj1": ex["q0"]["ans1"]["text"],
             "tid": int(p[4]), "attr_cls": p[5].strip(), "attr": p[6], "context": ex["context"],
             "q_neg": ex["q0"]["question"], "q_pos": ex["q1"]["question"],
             "gender0": g0, "age0": a0, "eth0": e0, "gender1": g1, "age1": a1, "eth1": e1,
             "control": (a0 == a1) if control is None else control,
             "gendered_attr": bool(GENDERED_ATTR.search(p[6]))}
        assert r["subj0"] == p[2] and r["subj1"] == p[3], key
        rows.append(r)


LISTS = ["gender_names", "gender_names_ctrl", "gender_nouns", "age", "gender_age", "gender_ethnicity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", default=LISTS, choices=LISTS)
    ap.add_argument("--force", action="store_true", help="regenerate even if the output file exists")
    args = ap.parse_args()
    runs = write_inputs()
    os.makedirs(OUT_DIR, exist_ok=True)
    for lst in args.lists:
        out = os.path.join(OUT_DIR, f"{lst}.jsonl.gz")
        if os.path.exists(out) and not args.force:
            print(f"{lst}: {out} exists, kept (use --force to regenerate)")
            continue
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, (subj, slot, control) in enumerate(runs[lst]):
                js = os.path.join(tmp, f"{i}.json")
                run_unqover(subj, slot, js)
                convert(json.load(open(js, encoding="utf-8")), lst, control, rows)
        assert len({(r["subj0"], r["subj1"], r["tid"], r["attr"]) for r in rows}) == len(rows), f"{lst}: duplicate keys"
        with gzip.open(out + ".tmp", "wt", encoding="utf-8") as g:
            for r in rows:
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(out + ".tmp", out)
        print(f"{lst}: {len(rows)} contexts, {2 * len(rows)} questions, {sum(r['control'] for r in rows)} control -> {out}")


if __name__ == "__main__":
    main()
