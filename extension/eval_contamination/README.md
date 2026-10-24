# Extension: are popular safety, bias and capability benchmarks in C4.EN? (building on §4.2)

The reproduction confirmed Dodge et al.'s Table 2: test texts of 2019-era NLP benchmarks appear verbatim in C4.EN.
This asks the same question for the benchmarks people use **now** for safety, bias and capabilities. As far as our
literature search found, only CrowS-Pairs has been measured in C4 before (WIMBD: 0%).

## Design: provenance decides what to expect

C4 is the April 2019 crawl, so benchmarks fall into three groups:

| Group | Fields | What a match means |
|---|---|---|
| **Web text, before Apr 2019** | BOLD (Wikipedia sentences), RealToxicityPrompts (OpenWebText), ETHICS commonsense (Reddit AITA + MTurk), HellaSwag (WikiHow / ActivityNet), MMLU (exam questions collected online) | The benchmark and C4 share a web source, not a leak |
| **Written before Apr 2019** | WinoBias (2018 templates), ARC-Challenge (2018, science exams) | Could be online before the crawl |
| **Written after Apr 2019: negative control** | TruthfulQA, BBQ, CrowS-Pairs, StereoSet, ToxiGen, XSTest, AdvBench, HarmBench, Do-Not-Answer, GSM8K, WinoGrande | Cannot be in C4: any match is chance wording, i.e. the method's false-positive rate |

- **Positive control:** the reproduction's XSum Table 5 summaries (8 of 10 found).
- **Method:** the reproduction's infini-gram client, unchanged. It runs exact matching over all 364.9M C4.EN
  documents, a **lower bound** because case and punctuation must agree.
- **Sampling:** a random sample per field (seed 0), with Wilson 95% CIs.
- **Length:** results are also reported by text length, because short generic items match by chance.

## Data notes

- **HellaSwag:** uses `ctx_a`, the source sentences. `ctx` ends in a lowercased fragment of the answer, so it would
  never match.
- **RealToxicityPrompts:** 29 mis-encoded texts are repaired.
- **Gated sets:** AdvBench, HarmBench and XSTest are gated on Hugging Face, so they come from the authors' GitHub
  CSVs. CrowS-Pairs also comes from GitHub, because its HF loader is a script.
- **Long texts:** ETHICS posts run up to 1,400 words. Exact matches of long posts are rare even when the post is in
  C4, because C4 dropped lines.

## Run

```bash
python scripts/build_benchmarks.py
python ../../reproduction/benchmark-contamination/scripts/query_infinigram.py --bench_dir data/benchmarks \
    --out_dir results/E1_infinigram_200 --sample 200 --group_size 8 --fields <negative-control fields>
python ../../reproduction/benchmark-contamination/scripts/query_infinigram.py --bench_dir data/benchmarks \
    --out_dir results/E1_infinigram_200 --sample 200 --group_size 3 --fields <web / pre-2019 fields>
python scripts/analyze.py --counts results/E1_infinigram_200 --out_dir results/E1_infinigram_200/analysis
```

Outputs: `by_field.csv`, `by_group.csv`, `by_field_and_length.csv`, `negative_control_matches.csv`, `summary.md`,
`fig_by_field.png`.
