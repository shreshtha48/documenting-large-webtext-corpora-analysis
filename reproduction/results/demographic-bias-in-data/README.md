# Arm A figures (UnifiedQA-3B, all 147,000 contexts = 294,000 questions; rule = summed log-likelihood)

**A1. Table 7, paper vs ours.** Rank agreement with the paper: Spearman 0.84 (Pearson 0.87).
The paper's two focus groups reproduce: Jewish 65.7 [62.7, 68.8] vs paper 67.1, Arab 34.6 [31.8, 37.5] vs paper 37.0.
Groups where the paper lies outside our 95% CI: African, Black, Latino, South-American, Hispanic, European, Asian. Our values are somewhat more spread out than the
paper's (noise-corrected SD 12.09 vs 9.76; slope 1.086).
CIs: bootstrap over the 50 stereotype attributes.

**A2. Where a group's score comes from.** Positivity is the average of "picked for the positive question" and "not
picked for the negative one". Groups above the diagonal are favoured. For example, Black is picked for
69% of negative questions but 19% of positive ones; European 38% and
75%. Distance *along* the diagonal is a different thing: how often the model picks that name at all,
whatever the question. Native American sits on the diagonal at 69% / 68%, so its near-50% Table 7
value hides a strong name preference that Table 7 cannot show; Middle-Eastern is the opposite (rarely picked
either way).

**A3. Model size.** On the 1,516-context sample, rank agreement rises from -0.00 (small) to
0.79 (3B), and the share of contexts where the answer changes between the negative and
positive question rises from 17% to 50%. A model that gives the same answer to both questions
scores every group near 50%, so only the 3B model can show the paper's bias. Diamond: the full 3B run
(Spearman 0.84).

**A4. Jewish vs Arab head to head** (1,400 contexts naming both). For negative questions the model picks the
Arab person 84% of the time; for positive ones it picks the Jewish person 81% of the time. The paper
reports only the pooled Table 7 values; this is the direct comparison behind them.

**A5. Head to head, every pair.** Each cell is the row group's positivity in the contexts that name the row and column
group. The near-monotone pattern (blue upper-left, red lower-right) shows that Table 7 reflects a consistent ordering,
not a few extreme pairings. Per-cell CIs: `pairwise_ci.csv`.
