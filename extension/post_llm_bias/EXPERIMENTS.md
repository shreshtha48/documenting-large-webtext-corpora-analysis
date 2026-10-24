# Experiment log: 2019 vs post-LLM group sentiment

Every run is kept, including the ones that fail or don't reproduce. Raw outputs go in `results/<id>/`.

| ID | Date | What | Where | Outcome |
|---|---|---|---|---|
| P0 | 2026-10-24 | Pipeline test: 2019 shard 00000 vs the 4 pilot 2024 shards (lexicons only; RoBERTa on a 3,000-paragraph 2019 subset) | local CPU | Pipeline works. 2019 gap 7.32 matches B0. 2024 CIs are about ±30 points, too wide to read. Not a result |
| P1 | — | Main run: 12 shards of 2019 C4 vs 24 + 24 WET shards of 2024 C4 (lexicons + RoBERTa) | Kaggle T4 ×2 | not run yet |
