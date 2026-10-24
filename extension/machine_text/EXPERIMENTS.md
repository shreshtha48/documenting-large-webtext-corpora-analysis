# Experiment log: post-LLM C4 and machine-generated text

Every run is kept, including the ones that fail or don't reproduce. Raw outputs go in `results/<id>/`.

| ID | Date | What | Where | Outcome |
|---|---|---|---|---|
| F0 | 2026-10-24 | Filter check: `filter_c4_pipeline.py` vs the TFDS v3.0.1 reference code on 5,000 pages of CC-MAIN-2024-10 shard 0 | local | Identical C4.EN (998) and C4.BLOCKLIST (79). Found and fixed: wrong LDNOOBW commit URL (404), `{`/javascript rule order, missing URL and line dedupe |
| F1 | 2026-10-24 | Pilot shards: CC-MAIN-2024-10 shards 0 and 3, CC-MAIN-2024-51 shards 0 and 3 | Kaggle CPU | 9,496 / 9,510 / 7,769 / 7,813 C4.EN docs. Blocklist removes about 5% of `en.noblocklist` (2019: about 7.6%). Files in `../post-llm-crawl/` |
| R0 | 2026-10-24 | `run_wet_shards.py` test: CC-MAIN-2024-10 shard 0 via the parallel runner, then re-run to check resume | local | Output identical to the F1 Kaggle run (9,496 / 514). Re-run skipped the finished shard |
| M0 | 2026-10-24 | Pipeline test of `machine_text.py`: all 4 shards for markers; Binoculars with the small Qwen2.5-0.5B pair on 150 docs per corpus | local CPU | Pipeline works; Binoculars stopped at 108 docs (time cap) and resumes. Marker share 2.0% → 6.7% → 15.3% (2019 → Feb → Dec) on 150 docs per corpus. Pipeline test only, not a result |
| M1 | — | Main run: 4 shards + 2019 control, Binoculars Qwen2.5-1.5B pair, RoBERTa | Kaggle T4 ×2 | Done; outputs in `machine_text_results.zip`, figures in `../results/machine_text/`. Docs with any marker word: 4.6% → 7.6% → 9.7% (2019 → Feb → Dec 2024); control words flat (69.2 / 70.8 / 69.6). Lower bound on LLM-processed share: 3.2% [2.4, 3.9] (Feb), 5.4% [4.5, 6.2] (Dec). Binoculars at a 1% false-alarm threshold: 1.02% → 1.23% → 1.32% flagged (Dec CI 1.05–1.60, above 1%). The two signals barely overlap on single documents. Marker-word documents read more positively (RoBERTa, 2019: 91.4 vs 81.9) |
