"""Extension, probe analysis: positivity by gender, age and their intersections, from UnifiedQA option scores.

The metric is the reproduction's (Dodge et al. A.7, analyze_arm_a.py): a subject is "associated with positive
sentiment" when the model picks it for the positive question, or picks the other subject for the negative question.
Each question gives both of its subjects one trial. Scoring, the answer rule (summed log-prob), the join check and the
bootstrap over the 50 attributes are imported from the reproduction unchanged.

Clean contrasts. In the intersectional lists a "female" subject also meets female opponents, which would pull every
gender value toward 50%. So each axis is compared only on pairs that differ on that axis alone: gender on
"elderly woman" vs "elderly man", age on "elderly woman" vs "teenage girl", ethnicity on "Arab woman" vs
"Jewish woman". On those pairs the two values' positivities sum to 100, and the female - male gap is
2 x (female positivity - 50).

Controls and checks:
- gender_names' same-gender pairs (control=True): their per-name positivities show how much two names of one
  gender differ by chance; the female-vs-male result is compared with that spread.
- every gender contrast is repeated without the 4 attributes that mention women / wives / nails.
- diagnostics: share of picks that are option (a) (position bias), and how often the model gives the same answer
  to the negative and the positive question (then it is not reading the question).

Also, with --e6: the reproduction's own E6 run (UnifiedQA-3B, the paper's 147,000 ethnicity contexts) split by the
gender of the template (even tid = "man" templates, odd = "woman"). This needs no GPU: is Table 7's ethnicity
ranking, and the Jewish - Arab gap, the same when the people are women?

Usage:
    python scripts/analyze_probe.py --scores results/P1_unifiedqa3b/scores --out_dir results/P1_unifiedqa3b/analysis
    python scripts/analyze_probe.py --e6 --out_dir results/E6_gender_split      # reproduction scores, CPU only
"""
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))
REPRO = os.path.join(REPO, "reproduction", "demographic_bias")
sys.path.insert(0, HERE)                                   # on Kaggle analyze_arm_a.py sits next to this file
sys.path.insert(0, os.path.join(REPRO, "scripts"))
from analyze_arm_a import (TABLE7, add_picks, boot_ci, check_join, diagnostics, load, long_credits,  # noqa: E402
                           devbars, save, style, INK2, SURFACE, POS, NEG)

AXES = ["gender", "age", "eth"]
LISTS = ["gender_nouns", "gender_names", "age", "gender_age", "gender_ethnicity"]


def subject_axes(q):
    """subject string -> {gender, age, eth}."""
    m = {}
    for i in (0, 1):
        for s, g, a, e in q[[f"subj{i}", f"gender{i}", f"age{i}", f"eth{i}"]].itertuples(index=False):
            m[s] = {"gender": g, "age": a, "eth": e}
    return m


def with_axes(long, axes_of):
    for side, col in (("self", "eth"), ("other", "other")):
        for ax in AXES:
            long[f"{side}_{ax}"] = long[col].map(lambda s: axes_of[s][ax])
    return long


def boot_diff(long, key, a, b, B, seed):
    """Positivity(a) - positivity(b) on rows grouped by `key`, bootstrapping attributes jointly. Returns point, lo, hi."""
    g = long[long[key].isin([a, b])].groupby(["attr", key])["credit"].agg(["sum", "count"]).unstack(key).fillna(0)
    S, N = g["sum"][[a, b]].to_numpy(), g["count"][[a, b]].to_numpy()
    rng = np.random.default_rng(seed)
    w = rng.multinomial(S.shape[0], [1 / S.shape[0]] * S.shape[0], size=B)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = S.sum(0) / N.sum(0)
        boot = (w @ S) / (w @ N)
    d = 100 * (boot[:, 0] - boot[:, 1])
    return 100 * (point[0] - point[1]), *np.nanpercentile(d, [2.5, 97.5])


def contrasts(long, B, seed):
    """Per axis: positivity of each value on pairs that differ on that axis only; overall and per fixed context."""
    rows = []
    for ax in AXES:
        others = [o for o in AXES if o != ax]
        clean = long[(long[f"self_{ax}"] != long[f"other_{ax}"]) & (long[f"self_{ax}"] != "")]
        for o in others:
            clean = clean[clean[f"self_{o}"] == clean[f"other_{o}"]]
        if clean.empty:
            continue
        for subset, d in (("all attributes", clean), ("without gendered attributes", clean[~clean["gendered_attr"]])):
            t = boot_ci(d, [f"self_{ax}"], B, seed).rename(columns={f"self_{ax}": "value"})
            t.insert(0, "within", "all"), t.insert(0, "subset", subset), t.insert(0, "axis", ax)
            rows.append(t)
            for o in others:                            # e.g. the gender contrast within each age group
                if (d[f"self_{o}"] != "").any():
                    t = boot_ci(d, [f"self_{o}", f"self_{ax}"], B, seed).rename(columns={f"self_{ax}": "value"})
                    t["within"] = o + "=" + t.pop(f"self_{o}")
                    t.insert(0, "subset", subset), t.insert(0, "axis", ax)
                    rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def name_control(long, B, seed):
    """Per name: positivity in same-gender control pairs and in cross-gender pairs; plus the spread of each."""
    out = []
    for kind, d in (("control (same gender)", long[long["control"]]), ("cross gender", long[~long["control"]])):
        t = boot_ci(d, ["eth", "self_gender"], B, seed).rename(columns={"eth": "name", "self_gender": "gender"})
        t.insert(0, "pairs", kind)
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    spread = t.groupby(["pairs", "gender"])["positivity"].agg(["mean", "std", "min", "max"]).round(2).reset_index()
    return t, spread


def fig_contrasts(c, title, path, dpi, plt):
    c = c[c["subset"] == "all attributes"].copy()
    c["label"] = c["axis"] + ": " + c["value"] + np.where(c["within"] == "all", "", "  (" + c["within"] + ")")
    c = c.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(c) + 1.4))
    style(ax)
    y = np.arange(len(c))
    devbars(ax, y, c["positivity"].to_numpy(), c["ci_lo"].to_numpy(), c["ci_hi"].to_numpy())
    ax.set_yticks(y, c["label"], fontsize=8.5, color=INK2)
    ax.set_xlabel("positivity on pairs that differ on this axis only (%; 50 = no preference)", color=INK2)
    ax.set_title(title, loc="left", fontsize=11)
    save(fig, path, dpi)
    return fig


def analyze_list(lst, scores_dir, qpath, args, plt):
    q = pd.read_json(qpath, lines=True, compression="gzip", dtype={"eth0": str, "eth1": str, "age0": str, "age1": str,
                                                                    "gender0": str, "gender1": str})
    df = load(scores_dir, q)
    rep = check_join(df)
    df = add_picks(df, args.rule)
    out = os.path.join(args.out_dir, lst)
    os.makedirs(out, exist_ok=True)
    diag = {"list": lst, "contexts in question file": int(len(q)), **diagnostics(df), **rep}
    if len(df) < len(q):
        diag["WARNING"] = f"only {len(df)} of {len(q)} contexts scored; results are partial"
    long = long_credits(df)
    # long_credits keeps attr, attr_cls, tid; it stacks 4 copies of df in row order, so the flags tile
    assert len(long) == 4 * len(df)
    long["control"] = np.tile(df["control"].astype(bool).to_numpy(), 4)
    long["gendered_attr"] = np.tile(df["gendered_attr"].astype(bool).to_numpy(), 4)
    long = with_axes(long, subject_axes(q))

    main_trials = long[~long["control"]]
    subj = boot_ci(main_trials, ["eth"], args.B, args.seed).rename(columns={"eth": "subject"})
    subj.sort_values("positivity", ascending=False).round(2).to_csv(os.path.join(out, "subjects.csv"), index=False)
    c = contrasts(main_trials, args.B, args.seed)
    c.round(2).to_csv(os.path.join(out, "contrasts.csv"), index=False)
    summary = {}
    for ax, a, b in (("gender", "female", "male"),):
        clean = main_trials[(main_trials["self_gender"] != main_trials["other_gender"])
                            & (main_trials["self_age"] == main_trials["other_age"])
                            & (main_trials["self_eth"] == main_trials["other_eth"])]
        if clean["self_gender"].isin([a, b]).any():
            for subset, d in (("all", clean), ("no gendered attrs", clean[~clean["gendered_attr"]])):
                p, lo, hi = boot_diff(d, "self_gender", a, b, args.B, args.seed)
                summary[f"female - male gap, {subset}"] = [round(p, 2), round(lo, 2), round(hi, 2)]
    if lst == "gender_names":
        t, spread = name_control(long, args.B, args.seed)
        t.round(2).to_csv(os.path.join(out, "names.csv"), index=False)
        spread.to_csv(os.path.join(out, "names_spread.csv"), index=False)
        summary["name positivity spread"] = spread.to_dict(orient="records")
    diag["summary"] = summary
    json.dump(diag, open(os.path.join(out, "diagnostics.json"), "w"), indent=1)
    if plt is not None and not c.empty:
        plt.close(fig_contrasts(c, f"{lst}: UnifiedQA positivity, clean contrasts (95% CI over attributes)",
                                os.path.join(out, "fig_contrasts.png"), args.dpi, plt))
    print(f"{lst}: {len(df)} contexts; {json.dumps(summary)}")
    return c.assign(list=lst)


def analyze_e6(args, plt):
    """The reproduction's E6 scores split by template gender (even tid = men, odd tid = women)."""
    q = pd.read_json(os.path.join(REPRO, "data", "questions", "ethnicity.jsonl.gz"), lines=True, compression="gzip")
    df = add_picks(load(os.path.join(REPRO, "results", "E6_full_3b", "scores", "unifiedqa-t5-3b"), q), args.rule)
    check_join(df)
    long = long_credits(df)
    long["template"] = np.where(long["tid"] % 2 == 0, "men", "women")
    t = boot_ci(long, ["template", "eth"], args.B, args.seed).rename(columns={"eth": "ethnicity"})
    wide = t.pivot(index="ethnicity", columns="template", values="positivity")
    diffs = []
    for e in wide.index:
        d = long[long["eth"] == e]
        p, lo, hi = boot_diff(d, "template", "women", "men", args.B, args.seed)
        diffs.append({"ethnicity": e, "women - men": p, "ci_lo": lo, "ci_hi": hi})
    res = wide.reset_index().merge(pd.DataFrame(diffs), on="ethnicity")
    res["dodge_table7"] = res["ethnicity"].map(TABLE7)
    res = res.sort_values("dodge_table7", ascending=False)
    os.makedirs(args.out_dir, exist_ok=True)
    res.round(2).to_csv(os.path.join(args.out_dir, "e6_by_template_gender.csv"), index=False)
    t.round(2).to_csv(os.path.join(args.out_dir, "e6_by_template_gender_ci.csv"), index=False)
    h2h = {}
    ja = df[((df["subj0"] == "Jewish") & (df["subj1"] == "Arab")) | ((df["subj0"] == "Arab") & (df["subj1"] == "Jewish"))]
    lj = long_credits(ja)
    lj["template"] = np.where(lj["tid"] % 2 == 0, "men", "women")
    for tmpl in ("men", "women"):
        p, lo, hi = boot_diff(lj[lj["template"] == tmpl], "eth", "Jewish", "Arab", args.B, args.seed)
        h2h[f"Jewish - Arab head-to-head gap, {tmpl} templates"] = [round(p, 2), round(lo, 2), round(hi, 2)]
    rho = {k: round(float(res[k].rank().corr(res["dodge_table7"].rank())), 3) for k in ("men", "women")}
    summary = {"spearman_vs_table7": rho, "rank correlation men vs women templates":
               round(float(res["men"].rank().corr(res["women"].rank())), 3), **h2h}
    json.dump(summary, open(os.path.join(args.out_dir, "e6_summary.json"), "w"), indent=1)
    print(res.round(1).to_string(index=False))
    print(json.dumps(summary, indent=1))
    if plt is not None:
        r = res.sort_values("women - men").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        style(ax)
        y = np.arange(len(r))
        v = r["women - men"].to_numpy()
        ax.barh(y, v, color=[POS if x >= 0 else NEG for x in v], height=0.62, edgecolor=SURFACE)
        ax.errorbar(v, y, xerr=[v - r["ci_lo"], r["ci_hi"] - v], fmt="none", ecolor=INK2, capsize=3)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(y, r["ethnicity"], color=INK2)
        ax.set_xlabel("positivity in 'woman' templates minus 'man' templates (points, 95% CI)", color=INK2)
        ax.set_title("E6 (UnifiedQA-3B): does an ethnicity's positivity change when the people are women?",
                     loc="left", fontsize=10)
        save(fig, os.path.join(args.out_dir, "fig_e6_women_minus_men.png"), args.dpi)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", help="folder holding one sub-folder per list (<list>/chunk_*.csv.gz)")
    ap.add_argument("--questions", default=os.path.join(ROOT, "data", "questions"))
    ap.add_argument("--lists", nargs="+", default=LISTS)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--e6", action="store_true", help="split the reproduction's E6 run by template gender")
    ap.add_argument("--rule", default="sum", choices=["sum", "mean"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()
    plt = None
    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    if args.e6:
        analyze_e6(args, plt)
        return
    allc = []
    for lst in args.lists:
        d = os.path.join(args.scores, lst)
        if not glob.glob(os.path.join(d, "chunk_*.csv.gz")):
            print(f"{lst}: no scores in {d}, skipped")
            continue
        allc.append(analyze_list(lst, d, os.path.join(args.questions, f"{lst}.jsonl.gz"), args, plt))
    if allc:
        pd.concat(allc).round(2).to_csv(os.path.join(args.out_dir, "contrasts_all_lists.csv"), index=False)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
