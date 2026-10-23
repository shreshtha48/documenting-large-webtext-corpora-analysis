"""Collect every Arm A experiment under results/ into one table (results/experiments_summary.csv / .md).

Reads results/<experiment>/analysis_<rule>/<model>/{diagnostics.json, table7.csv}, as written by
analyze_arm_a.py. Experiments that did not reproduce the paper are kept on purpose.

Usage: python scripts/summarize_experiments.py [--results results]
"""
import argparse, glob, json, os

import pandas as pd

SIZE_ORDER = ["small", "base", "large", "3b", "11b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rows = []
    for diag_path in sorted(glob.glob(os.path.join(args.results, "*", "analysis_*", "*", "diagnostics.json"))):
        model_dir = os.path.dirname(diag_path)
        exp = os.path.basename(os.path.dirname(os.path.dirname(model_dir)))
        rule = os.path.basename(os.path.dirname(model_dir)).replace("analysis_", "")
        model = os.path.basename(model_dir)
        d = json.load(open(diag_path))
        t = pd.read_csv(os.path.join(model_dir, "table7.csv")).set_index("ethnicity")
        rows.append({
            "experiment": exp, "model": model, "rule": rule,
            "contexts_scored": d["contexts scored"],
            "spearman_vs_table7": round(d["spearman_vs_table7"], 3),
            "answer_flips_with_sentiment_%": round(100 * (1 - d["same pick for neg and pos question"]), 1),
            "share_option_a": d["share of picks = option (a)"],
            "Jewish_%": round(t.loc["Jewish", "positivity"], 1),
            "Jewish_CI": f"{t.loc['Jewish', 'ci_lo']:.1f}-{t.loc['Jewish', 'ci_hi']:.1f}",
            "Arab_%": round(t.loc["Arab", "positivity"], 1),
            "Arab_CI": f"{t.loc['Arab', 'ci_lo']:.1f}-{t.loc['Arab', 'ci_hi']:.1f}",
            "spread_max_minus_min": round(t["positivity"].max() - t["positivity"].min(), 1),
            # dispersion vs the paper (sd 9.4): rank agreement alone hides values that are too extreme
            "sd_noise_corrected": d.get("sd_ours_noise_corrected"),
            "slope_vs_paper": d.get("slope_ours_on_paper"),
            "spearman_no_negation_attrs": d.get("spearman_vs_table7_without_negation_attributes"),
            "coverage_ok": d.get("coverage ok"),
        })
    if not rows:
        raise SystemExit(f"no analyses found under {args.results}")
    df = pd.DataFrame(rows)
    df["_size"] = df["model"].map(lambda m: next((i for i, k in enumerate(SIZE_ORDER) if k in m.split("-")), 99))
    df = df.sort_values(["experiment", "rule", "_size"]).drop(columns="_size")
    paper = "Paper (Dodge et al. 2021, Table 7): Jewish 67.1%, Arab 37.0%, spread 30.5.\n"
    df.to_csv(os.path.join(args.results, "experiments_summary.csv"), index=False)
    with open(os.path.join(args.results, "experiments_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Arm A experiments (all runs, reproduced or not)\n\n" + paper + "\n")
        f.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for r in df.itertuples(index=False):
            f.write("| " + " | ".join(str(v) for v in r) + " |\n")
    print(df.to_string(index=False))
    print("\n" + paper + f"wrote {args.results}/experiments_summary.csv and .md")


if __name__ == "__main__":
    main()
