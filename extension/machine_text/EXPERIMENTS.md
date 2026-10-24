# Experiment log: post-LLM C4 and machine-generated text

Every run is kept, including the ones that fail or don't reproduce. Raw outputs go in `results/<id>/`.

| ID | Date | What | Where | Outcome |
|---|---|---|---|---|
| F0 | 2026-10-24 | Filter check: `filter_c4_pipeline.py` vs the TFDS v3.0.1 reference code on 5,000 pages of CC-MAIN-2024-10 shard 0 | local | Identical C4.EN (998) and C4.BLOCKLIST (79). Found and fixed: wrong LDNOOBW commit URL (404), `{`/javascript rule order, missing URL and line dedupe |
| F1 | 2026-10-24 | Pilot shards: CC-MAIN-2024-10 shards 0 and 3, CC-MAIN-2024-51 shards 0 and 3 | Kaggle CPU | 9,496 / 9,510 / 7,769 / 7,813 C4.EN docs. Blocklist removes about 5% of `en.noblocklist` (2019: about 7.6%). Files in `../post-llm-crawl/` |
| M0 | 2026-10-24 | Pipeline test of `machine_text.py`: all 4 shards for markers; Binoculars with the small Qwen2.5-0.5B pair on 150 docs per corpus | local CPU | Pipeline test only, not a result |
| M1 | — | Main run: 4 shards + 2019 control, Binoculars Qwen2.5-1.5B pair, RoBERTa | Kaggle T4 ×2 | not run yet |
