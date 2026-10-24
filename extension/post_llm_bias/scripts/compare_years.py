"""Compare Arm B's group sentiment between 2019 C4 and post-LLM C4 (the same filters run on 2024 crawls).

Measures per (corpus, group): positivity under the three Arm B lexicons (count_sentiment.py's exact counting) and,
if classify_sentiment.py has run, under the RoBERTa classifier (sum of p_pos / sum of p_pos + p_neg).

Numbers reported, each with a 95% percentile CI from a web-domain bootstrap (B=1000, corpora resampled
independently, as in Arm B):
  positivity           per corpus, group, measure
  relative             group minus Baseline (random paragraphs of the same corpus): the group's tone against the
                       corpus's general tone
  change               corpus minus the reference corpus (2019), for positivity and for relative
                       (the second is a difference-in-differences: it removes any general shift in web tone)
  gap                  Jewish minus Arab (the paper's comparison), and with geopolitical "Arab" paragraphs removed
  dispersion           SD of `relative` across groups with enough paragraphs everywhere: how unequal the groups
                       are, raw and noise-corrected (minus the mean bootstrap variance); plus the Spearman
                       correlation of the group ranking with the reference corpus

Usage:
    python compare_years.py --corpus 2019=paragraphs/2019 --corpus CC-MAIN-2024-10=paragraphs/CC-MAIN-2024-10 \
        --corpus CC-MAIN-2024-51=paragraphs/CC-MAIN-2024-51 \
        --corpus "2024 pooled=paragraphs/CC-MAIN-2024-10,paragraphs/CC-MAIN-2024-51" --out_dir analysis
The first --corpus is the reference.
"""
import argparse, glob, gzip, json, os, sys
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "..", "..", "reproduction", "demographic_bias", "scripts")):
    sys.path.insert(0, os.path.abspath(p))
import count_sentiment as cs
from scan_c4_paragraphs import TABLE7_GROUPS
from scan_groups import BASELINE

ARAB_NG = "Arab (non-geopolitical)"
NG_VARIANT = "Arab: no geopolitical 'Arab' phrases"
GROUPS = list(TABLE7_GROUPS)
TARGETS = GROUPS + [BASELINE, ARAB_NG]
CORPUS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # reference palette slots 1-4 (validated)


def load_corpus(dirs, lex):
    """Per-domain (targets x measures x [pos, neg]) sums for one corpus, plus paragraph counts."""
    files = [f for d in dirs for f in sorted(glob.glob(os.path.join(d, "paragraphs_*.jsonl.gz")))]
    if not files:
        raise SystemExit(f"no paragraphs in {dirs}")
    targets = GROUPS + [BASELINE]
    per_domain, robust, kept, _, _, scale, shards = cs.count(files, lex, targets)
    L = len(lex)
    has_cls = all(os.path.exists(f.replace("paragraphs_", "scores_").replace(".jsonl.gz", ".npy")) for f in files)
    M = L + has_cls
    X = defaultdict(lambda: np.zeros((len(TARGETS), M, 2)))
    a = targets.index("Arab")
    for host, arr in per_domain.items():
        X[host][:len(targets), :L] = arr
    for host, arr in robust[NG_VARIANT].items():
        X[host][TARGETS.index(ARAB_NG), :L] = arr[a]
    n_par = {t: scale[f"{t} paragraphs"] for t in targets}
    n_par[ARAB_NG] = kept[(NG_VARIANT, "Arab")]
    if has_cls:
        for f in files:
            probs = np.load(f.replace("paragraphs_", "scores_").replace(".jsonl.gz", ".npy")).astype(np.float64)
            with gzip.open(f, "rt", encoding="utf-8") as g:
                for line, p in zip(g, probs):
                    r = json.loads(line)
                    host = urlparse(r["url"]).netloc
                    for t in r["targets"]:
                        X[host][TARGETS.index(t), L] += (p[2], p[0])
                        if t == "Arab" and not any(x in r["text"] for x in cs.ARAB_GEOPOLITICAL):
                            X[host][TARGETS.index(ARAB_NG), L] += (p[2], p[0])
    measures = list(lex) + (["RoBERTa"] if has_cls else [])
    return dict(X), measures, n_par, len(shards)


def ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5], axis=0)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True, help="LABEL=DIR[,DIR...]; first = reference")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_paragraphs", type=int, default=100, help="groups below this in any corpus are flagged")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lex = cs.load_lexicons()

    corpora = {}
    for spec in args.corpus:
        label, dirs = spec.split("=", 1)
        X, measures, n_par, n_shards = load_corpus(dirs.split(","), lex)
        tot, point, pb, n_hosts = cs.boot(X, args.B, args.seed)
        corpora[label] = dict(point=point, pb=pb, n_par=n_par, n_shards=n_shards, n_hosts=n_hosts, measures=measures)
        print(f"{label}: {n_shards} shards, {n_hosts} domains, measures {measures}, paragraphs "
              f"{ {t: n_par[t] for t in ('Jewish', 'Arab', BASELINE)} }", flush=True)
    measures = min((c["measures"] for c in corpora.values()), key=len)   # RoBERTa only if every corpus has it
    labels = list(corpora)
    ref = labels[0]
    ti = {t: i for i, t in enumerate(TARGETS)}
    b = ti[BASELINE]

    # positivity and relative (group - Baseline), per corpus
    rows = []
    for c, d in corpora.items():
        for t in TARGETS:
            for mi, m in enumerate(measures):
                rel_s = d["pb"][:, ti[t], mi] - d["pb"][:, b, mi]
                lo, hi = ci(d["pb"][:, ti[t], mi])
                rlo, rhi = ci(rel_s)
                rows.append({"corpus": c, "target": t, "measure": m, "paragraphs": d["n_par"][t],
                             "positivity": d["point"][ti[t], mi], "ci_lo": lo, "ci_hi": hi,
                             "relative": d["point"][ti[t], mi] - d["point"][b, mi], "rel_lo": rlo, "rel_hi": rhi})
    pos = pd.DataFrame(rows)
    enough = {t: all(corpora[c]["n_par"][t] >= args.min_paragraphs for c in labels) for t in TARGETS}
    pos["enough_paragraphs_everywhere"] = pos["target"].map(enough)

    # change vs the reference corpus (independent bootstraps, replicate b paired with b)
    rows = []
    R = corpora[ref]
    for c in labels[1:]:
        d = corpora[c]
        for t in TARGETS:
            for mi, m in enumerate(measures):
                dp = d["pb"][:, ti[t], mi] - R["pb"][:, ti[t], mi]
                dd = (d["pb"][:, ti[t], mi] - d["pb"][:, b, mi]) - (R["pb"][:, ti[t], mi] - R["pb"][:, b, mi])
                (lo, hi), (dlo, dhi) = ci(dp), ci(dd)
                rows.append({"corpus": c, "vs": ref, "target": t, "measure": m,
                             "change": d["point"][ti[t], mi] - R["point"][ti[t], mi], "ci_lo": lo, "ci_hi": hi,
                             "change_vs_baseline": (d["point"][ti[t], mi] - d["point"][b, mi])
                                                   - (R["point"][ti[t], mi] - R["point"][b, mi]),
                             "dd_lo": dlo, "dd_hi": dhi, "enough_paragraphs_everywhere": enough[t]})
    chg = pd.DataFrame(rows)

    # Jewish - Arab gap per corpus, and its change vs the reference
    rows = []
    for arab in ("Arab", ARAB_NG):
        for c, d in corpora.items():
            for mi, m in enumerate(measures):
                g = d["pb"][:, ti["Jewish"], mi] - d["pb"][:, ti[arab], mi]
                g0 = R["pb"][:, ti["Jewish"], mi] - R["pb"][:, ti[arab], mi]
                lo, hi = ci(g)
                clo, chi = ci(g - g0) if c != ref else (np.nan, np.nan)
                pt = d["point"][ti["Jewish"], mi] - d["point"][ti[arab], mi]
                pt0 = R["point"][ti["Jewish"], mi] - R["point"][ti[arab], mi]
                rows.append({"arab_paragraphs": arab, "corpus": c, "measure": m, "gap": pt, "ci_lo": lo, "ci_hi": hi,
                             "change_vs_ref": pt - pt0 if c != ref else np.nan, "change_lo": clo, "change_hi": chi,
                             "Jewish_paragraphs": d["n_par"]["Jewish"], "Arab_paragraphs": d["n_par"][arab]})
    gap = pd.DataFrame(rows)

    # dispersion of relative positivity across groups, and rank agreement with the reference
    keep = [t for t in GROUPS if enough[t]]
    rows = []
    for c, d in corpora.items():
        for mi, m in enumerate(measures):
            rel = np.array([d["point"][ti[t], mi] - d["point"][b, mi] for t in keep])
            rel_s = np.stack([d["pb"][:, ti[t], mi] - d["pb"][:, b, mi] for t in keep], 1)
            sd_s = np.nanstd(rel_s, axis=1, ddof=1)
            noise = np.nanmean(np.nanvar(rel_s, axis=0, ddof=1))
            raw = np.std(rel, ddof=1) if len(keep) > 1 else np.nan
            lo, hi = ci(sd_s)
            ref_rel = pd.Series([R["point"][ti[t], mi] for t in keep])
            rho = pd.Series(rel).rank().corr(ref_rel.rank()) if len(keep) > 2 else np.nan
            rows.append({"corpus": c, "measure": m, "groups": len(keep), "sd_relative": raw, "ci_lo": lo, "ci_hi": hi,
                         "sd_noise_corrected": float(np.sqrt(max(0.0, raw ** 2 - noise))) if len(keep) > 1 else np.nan,
                         "spearman_vs_ref": rho})
    disp = pd.DataFrame(rows)

    for name, df in (("positivity", pos), ("change", chg), ("gap", gap), ("dispersion", disp)):
        df.round(3).to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures(pos, chg, gap, labels, measures, keep, args, plt)

    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Group sentiment: 2019 C4 vs post-LLM C4\n\n")
        for c, d in corpora.items():
            f.write(f"- **{c}**: {d['n_shards']} shards, {d['n_hosts']} domains, "
                    f"{d['n_par']['Jewish']} Jewish / {d['n_par']['Arab']} Arab / {d['n_par'][BASELINE]} baseline paragraphs\n")
        f.write(f"\nReference: {ref}. Groups with < {args.min_paragraphs} paragraphs in any corpus are left out of "
                f"the dispersion and the figures: {[t for t in GROUPS if not enough[t]]}\n\n")
        f.write("## Jewish - Arab gap\n\n" + cs.md_table(gap.round(2)) + "\n")
        f.write("## Dispersion of group positivity (relative to baseline)\n\n" + cs.md_table(disp.round(2)) + "\n")
        sel = chg[chg["enough_paragraphs_everywhere"]].drop(columns="enough_paragraphs_everywhere")
        f.write("## Change vs reference\n\n" + cs.md_table(sel.round(2)) + "\n")
        f.write("## Positivity\n\n" + cs.md_table(pos.round(2)) + "\n")
    print(gap.round(2).to_string(index=False))
    print(disp.round(2).to_string(index=False))
    print(f"wrote {args.out_dir}")


def figures(pos, chg, gap, labels, measures, keep, args, plt):
    from matplotlib.lines import Line2D
    colors = dict(zip(labels, CORPUS_COLORS))
    ref = labels[0]
    order = (pos[(pos.corpus == ref) & (pos.measure == measures[0]) & pos.target.isin(keep)]
             .sort_values("positivity")["target"].tolist()) + [BASELINE]
    y = np.arange(len(order))
    off = np.linspace(0.25, -0.25, len(labels))
    legend = [Line2D([], [], color=colors[c], marker="o", linestyle="none", label=c) for c in labels]

    # 1. positivity per group, every corpus
    fig, axes = plt.subplots(1, len(measures), figsize=(4.2 * len(measures), 0.5 * len(order) + 2.2),
                             sharey=True, facecolor=cs.SURFACE)
    for ax, m in zip(np.atleast_1d(axes), measures):
        cs.style(ax)
        for c, dy in zip(labels, off):
            d = pos[(pos.corpus == c) & (pos.measure == m)].set_index("target").reindex(order)
            v = d["positivity"].to_numpy()
            ax.errorbar(v, y + dy, xerr=[v - d["ci_lo"], d["ci_hi"] - v], fmt="o", color=colors[c], ms=6,
                        mec=cs.SURFACE, mew=1.2, ecolor=colors[c], elinewidth=1.2, capsize=2, zorder=3)
        ax.set_title(m, color=cs.INK, loc="left", fontsize=10)
        ax.set_xlabel("Positivity (%)", color=cs.INK2)
    np.atleast_1d(axes)[0].set_yticks(y, order, color=cs.INK)
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.01, 0), ncol=len(labels), frameon=False, labelcolor=cs.INK2, fontsize=9)
    fig.suptitle("Positivity of paragraphs mentioning each group, by corpus (95% CIs, domain bootstrap)\n"
                 "Baseline = random paragraphs of the same corpus", color=cs.INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(os.path.join(args.out_dir, "fig_positivity_by_corpus.png"), dpi=args.dpi, facecolor=cs.SURFACE)
    plt.close(fig)

    # 2. difference-in-differences vs the reference
    others = labels[1:]
    order2 = order[:-1]
    y2 = np.arange(len(order2))
    off2 = np.linspace(0.2, -0.2, len(others)) if len(others) > 1 else [0.0]
    fig, axes = plt.subplots(1, len(measures), figsize=(4.2 * len(measures), 0.5 * len(order2) + 2.2),
                             sharey=True, facecolor=cs.SURFACE)
    for ax, m in zip(np.atleast_1d(axes), measures):
        cs.style(ax)
        for c, dy in zip(others, off2):
            d = chg[(chg.corpus == c) & (chg.measure == m)].set_index("target").reindex(order2)
            v = d["change_vs_baseline"].to_numpy()
            ax.errorbar(v, y2 + dy, xerr=[v - d["dd_lo"], d["dd_hi"] - v], fmt="o", color=colors[c], ms=6,
                        mec=cs.SURFACE, mew=1.2, ecolor=colors[c], elinewidth=1.2, capsize=2, zorder=3)
        ax.axvline(0, color=cs.INK2, linewidth=1)
        ax.set_title(m, color=cs.INK, loc="left", fontsize=10)
        ax.set_xlabel(f"Change in (group - baseline) vs {ref}, points", color=cs.INK2)
    np.atleast_1d(axes)[0].set_yticks(y2, order2, color=cs.INK)
    fig.legend(handles=legend[1:], loc="lower left", bbox_to_anchor=(0.01, 0), ncol=len(others), frameon=False, labelcolor=cs.INK2, fontsize=9)
    fig.suptitle(f"How each group's tone changed since {ref}, net of the change in the web's general tone\n"
                 "> 0: the group is talked about more positively than before, relative to everything else",
                 color=cs.INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(os.path.join(args.out_dir, "fig_change_vs_baseline.png"), dpi=args.dpi, facecolor=cs.SURFACE)
    plt.close(fig)

    # 3. the paper's Jewish - Arab gap per corpus
    variants = ["Arab", ARAB_NG]
    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 0.45 * len(measures) * len(labels) + 2),
                             sharey=True, facecolor=cs.SURFACE)
    yy = np.arange(len(measures))[::-1]
    off3 = np.linspace(0.27, -0.27, len(labels))
    for ax, a in zip(axes, variants):
        cs.style(ax)
        for c, dy in zip(labels, off3):
            d = gap[(gap.arab_paragraphs == a) & (gap.corpus == c)].set_index("measure").reindex(measures)
            v = d["gap"].to_numpy()
            ax.errorbar(v, yy + dy, xerr=[v - d["ci_lo"], d["ci_hi"] - v], fmt="o", color=colors[c], ms=6,
                        mec=cs.SURFACE, mew=1.2, ecolor=colors[c], elinewidth=1.2, capsize=2, zorder=3)
        ax.axvline(0, color=cs.INK2, linewidth=1)
        ax.set_title(f"Arab paragraphs: {'all' if a == 'Arab' else 'without geopolitical phrases'}",
                     color=cs.INK, loc="left", fontsize=10)
        ax.set_xlabel("Jewish minus Arab positivity (points)", color=cs.INK2)
    axes[0].set_yticks(yy, measures, color=cs.INK)
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.01, 0), ncol=len(labels), frameon=False, labelcolor=cs.INK2, fontsize=9)
    fig.suptitle("The paper's Jewish - Arab gap in each corpus (95% CIs, domain bootstrap)", color=cs.INK,
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(os.path.join(args.out_dir, "fig_gap_by_corpus.png"), dpi=args.dpi, facecolor=cs.SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
