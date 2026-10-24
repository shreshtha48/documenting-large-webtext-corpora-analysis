# Experiment log: 2019 vs post-LLM group sentiment

Every run is kept, including the ones that fail or don't reproduce. Raw outputs go in `results/<id>/`.

| ID | Date | What | Where | Outcome |
|---|---|---|---|---|
| P0 | 2026-10-24 | Pipeline test: 2019 shard 00000 vs the 4 pilot 2024 shards (lexicons only; RoBERTa on a 3,000-paragraph 2019 subset) | local CPU | Pipeline works. 2019 gap 7.32 matches B0. 2024 CIs are about ±30 points, too wide to read. Not a result |
| P1 | — | Main run: 12 shards of 2019 C4 vs 24 + 24 WET shards of 2024 C4 (lexicons + RoBERTa) | Kaggle T4 ×2 | Done; outputs in `post_llm_bias_results.zip`, figures in `../results/post_llm_bias/`. The Jewish–Arab gap shrinks: SocialSent 7.1 [5.6, 8.7] in 2019 vs 0.3 [−5.8, 6.5] in 2024 (change −6.8 [−13.3, −0.3]); RoBERTa 8.0 vs 2.9 (n.s.). Arab paragraphs gain +13.9 [7.3, 20.0] points (RoBERTa) beyond the whole web's shift in Feb 2024. Group ranking mostly kept (Spearman 0.62–0.97). Caucasian, Alaskan and South-American dropped (< 100 paragraphs in a 2024 crawl) |
