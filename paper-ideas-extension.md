# Extension directions: what papers citing Dodge et al. (2021) have done

**Purpose:** give you an overview of work since 2021 that builds on, extends or tests the C4 audit. For each of your ideas in [papers/implementation-ideas/ideas.txt](papers/implementation-ideas/ideas.txt), it says what already exists, so the extension adds something new.

## How this list was built

1. **Fetching citing papers.** I pulled every citing paper from **Semantic Scholar**, which counts **785** citations, and from **OpenAlex**, which counts **299**. After merging and removing duplicates by title, that gave **829 unique papers**.
   - The "~1,050" figure you saw is most likely Google Scholar, which also counts theses, preprint duplicates and so on.
   - Semantic Scholar also returned the sentence where each paper cites Dodge et al. for 636 of them.
   - The full list is in [papers/citing-papers/citing_papers.csv](papers/citing-papers/citing_papers.csv).
2. **Triage.** I scored all 829 on keyword themes, read every title below the cutoff by hand (and added back about 45 that the keywords missed), then read the abstract and citing sentence for the **476 candidates**.
3. **Full-text check.** For **35 key papers** I downloaded the arXiv PDF, confirmed it actually cites Dodge et al. 2021, and read the relevant sections.
   - **One correction from this step:** FineWeb (arXiv:2406.17557) does **not** cite Dodge et al. Its "Dodge" mentions are in the Dolma author list. I list it under related work only.

**Tags on each paper:**
- **[PDF✓]**: I read the PDF and confirmed the citation.
- **[CTX]**: Semantic Scholar returned the citing sentence, but I did not read the PDF.
- **[ABS]**: I have the abstract only.

Numbers are quoted from the paper or its abstract. Where I computed a number myself, I say so.

---

## 1. Your ideas, mapped to existing work

### Idea 1: does langdetect itself introduce bias?

**Already done, in part.**
- **AboutMe** (Lucy, Gururangan, Soldaini, Strubell, Bamman, Klein, **Dodge**; ACL 2024; arXiv:2401.06408) **[PDF✓]**.
  - They built 10.3M self-descriptions of website creators and ran ten "quality" and English langID filters over them. **C4's own `langdetect` with a 0.99 cutoff is one of the filters.**
  - Finding: "common langID classifiers can overlook English content in non-anglophone regions of the world, especially Asia, even when controlling for … webpage length and topic."
  - They also note that CLD2 and langdetect "score the vast majority (>90%) of pages similarly".
  - Code: `github.com/lucy3/whos_filtered` (verified to exist).
- **Blevins & Zettlemoyer 2022** (EMNLP; arXiv:2204.08110) **[PDF✓]**. The inverse problem: non-English text that *survives* in "English" corpora, C4.EN included. They estimate it with fastText langID on the first 50M C4 examples.

**What's left:** AboutMe measures by the *website's* region. I did not find a paper that runs C4's exact langdetect 0.99 rule over **labelled varieties of English** (Indian, Nigerian, Singaporean, Philippine English and so on), holding the text type fixed. That would test Dodge §3.3 (low representation of India, Nigeria and others) as a *filter* effect rather than a crawl effect. You would need to find a labelled-varieties dataset; I have not checked which are freely licensed.

### Idea 2: harmful text with no explicit "bad words"

- **WIMBD** (Elazar … **Dodge**; ICLR 2024; arXiv:2310.20707) **[PDF✓]**. Measured with a lexicon of "toxic triggers", "C4 is the least toxic … only 0.01%". With an SVM toxicity classifier, "5.75%", "which may indicate subtleties that the lexicon used for filtering documents from C4 did not catch."
- **Data Caricatures** (Deas et al.; arXiv:2503.10789) **[PDF✓]**. Of AAL texts that remain *in* C4.EN, annotators judged **28.9%** inappropriate and **24.5%** stereotype-reinforcing. Many contain variants of blocklist terms or asterisk self-censorship that the exact-word list misses.
- **Epistemic Injustice in LMs: an audit of pretraining filters and guardrails** (Stranisci et al.; arXiv:2606.05936) **[PDF✓]**. Four pretraining filters, including the C4 lexicon, and three guardrails, run on Common Crawl sentences. Decisions are "strongly associated with blocklist-based lexical cues, while frequently failing to flag … explicit hate speech." Transgender people, women and Central Americans are "significantly over-flagged".
- **What Are They Filtering Out?** (Stranisci & Hardmeier; arXiv:2503.05721) **[PDF✓]**. Benchmarks filtering strategies drawn from 55 LLM technical reports. Harm reduction "has the side effect of increasing the underrepresentation of vulnerable groups".
- **Towards Safer Pretraining** (Mendu et al.; arXiv:2505.02009) **[PDF✓]**. Annotated 3M pages, 1M each from Common Crawl, C4 and FineWeb, with a harm taxonomy.

### Idea 3: is machine-generated text an echo chamber for bias?

- **Data Caricatures [PDF✓]**. Of the C4.EN texts annotators judged to contain AAL features, **51%** were judged unlikely to have been written by a native AAL speaker, and "most of these texts (**70%**) were perceived as not human-written." This is the closest existing evidence for your idea, and it is on C4 itself.
- **A Shocking Amount of the Web is Machine Translated** (Thompson et al.; arXiv:2401.05749) **[PDF✓]**. Multi-way parallel, machine-translated content dominates lower-resource languages. There is "a selection bias in the type of content which is translated", with low-quality English translated en masse.
- **You Can Tell Who's Asking** (arXiv:2609.24106) **[PDF✓]**. Across 110 FineWeb snapshots (2013–2025), the genuine human share of question occurrences "fell by 79%" (42–56% after controlling for crawl composition).
- **Automatic MT detection with a surrogate model** (García-Romero et al.; arXiv:2511.02958) **[PDF✓]**. A detector you could reuse on C4's patent text.

**What's left:** nobody I found measures a bias statistic (for example Dodge's Jewish/Arab sentiment gap, or identity-term exclusion) **separately on machine-generated and human-written portions** of C4.

### Idea 4: fine-tune models with and without C4

- **A Pretrainer's Guide to Training Data** (Longpre et al.; arXiv:2305.13169) **[PDF✓]**. Pretrains **28 models of 1.5B parameters**, varying data age and toxicity/quality filters. Their C4 version keeps the bad-words documents. They found "a trade-off between performance on standard benchmarks and risk of toxic generations."
- **From Pretraining Data to LMs to Downstream Tasks** (Feng et al.; arXiv:2305.08283) **[PDF✓]**. Traces political bias from pretraining data to downstream fairness.
- **Feasibility:** this needs pretraining or large fine-tuning runs, which is beyond a laptop or Colab. The brief says the problems were chosen so that free compute suffices. **I recommend against this direction.**

### Idea 5: patents and copyright, and has the source distribution changed?

- **Consent in Crisis** (Longpre et al.; arXiv:2407.14933) **[PDF✓]**. Longitudinal audit of robots.txt and Terms of Service for 14,000 domains, using the Wayback Machine. Between 2023 and 2024, "~5%+ of all tokens in C4, or 28%+ of the most actively maintained, critical sources in C4" became fully restricted; "a full 45% of C4" is restricted by Terms of Service. Code: `github.com/Data-Provenance-Initiative/Data-Provenance-Collection` (verified).
- **Practical Datasets for Analyzing LLM Corpora Derived from Common Crawl** (ICWSM 2025; DOI 10.1609/icwsm.v19i1.35948) **[ABS]**. Releases domain-level statistics **across 96 Common Crawl snapshots**, plus standardised URL lists for C4, RefinedWeb and CulturaX. **This is the ready-made data for asking whether the domain distribution has changed over time.**
- **Dated Data** (Cheng et al.; arXiv:2403.12958) **[PDF✓]**. New CC dumps contain "non-trivial amounts of old data", which confirms Dodge §3.2.
- **Web Crawler Restrictions & Political Biases** (Bouchaud & Ramaciotti; arXiv:2510.09031) **[PDF✓]**. Blocking is uneven: 34.2% of news outlets block GPTBot, but only 4.1% of right-leaning outlets block OpenAI.
- On **copyright**, Dodge et al. (§6) explicitly "leave quantifying or removing such text to future work".

### Idea 6: has the dialect or language mix changed as corpora evolved?

- **Data Caricatures [PDF✓]**. Measures AAL across 12 corpora, from OpenWebText and C4 to Dolma, RefinedWeb, FineWeb-Edu and RedPajama-v2. AAL is 0.007%–0.18% of documents. **C4.EN: 280,604 of 365M (0.07%). C4.EN.NOBLOCKLIST: 447,032 of 395M (0.11%).** It uses Dodge's argmax approach with the same Blodgett classifier. Code: `github.com/NickDeas/DataCaricatures` (verified).
  - *My arithmetic from their table:* the blocklist removes 1 − 280,604/447,032 = **37.2%** of AAL-argmax documents. Dodge §5.3 reports **42%**. This gap has not been explained, and is a good reproduction target.
- **How Does "English (US)" Become the Default?** (Nayeem & Rafiei; arXiv:2604.04204) **[PDF✓]**. American English is favoured over British English "across all six audited pretraining corpora". Its link list includes `allenai/c4`.

### Idea 7: what percentage of C4 is machine-generated?

I found **no paper among the 829 that estimates this for C4 as a whole.** Dodge et al. say it "motivate[s] further work". The tools exist (MT detection, Infini-gram's machine-text analysis). The hard part is the construct: OCR, machine translation and template text are different things.

### Idea 8: bias in machine-generated vs human text

See Idea 3. Data Caricatures is the only C4-specific evidence I found.

### Ideas 9, 10 and 15: contamination, evals not in C4, and safety evals in C4

- **Data Contamination: From Memorization to Exploitation** (Magar & Schwartz; ACL 2022; arXiv:2203.08242) **[PDF✓]**. Pretrains BERT with seeded contamination. Exploitation "exists in some cases, but in others the models memorize the contaminated data, but do not exploit it." **This is directly relevant to whether contamination actually changes eval scores.**
- **Rethinking Benchmark and Contamination with Rephrased Samples** (Yang et al.; arXiv:2311.04850) **[PDF✓]**. String matching is insufficient, because paraphrase and translation evade it. Code: `github.com/lm-sys/llm-decontaminator` (verified).
- **An Open Source Data Contamination Report for LLMs** (Li et al.; arXiv:2310.17589) **[PDF✓]**. Contamination of 1% to 45% across six QA benchmarks. It "does not necessarily lead to increased model metrics."
- **WIMBD [PDF✓]**. Contamination of 82 PromptSource test sets in C4, OpenWebText, the Pile and RedPajama (appendix Table 16). Method: "all input fields are present in a single document"; they describe it as an "upper bound of exact-match" contamination.
- **Data Portraits** (Marone & Van Durme; arXiv:2303.03919) **[PDF✓]**. Explicitly "do not seek to exactly replicate" Dodge's Table 2; they use 50-gram overlap on the Pile.
- **Infini-gram** (Liu et al.; arXiv:2401.17377) **[PDF✓]**. Suffix-array engine. **The public API indexes all of C4.EN train (`v4_c4train_llama`). I ran a query against it** (see [research-empirical-methods/README.md](research-empirical-methods/README.md) §3).
- **SoftMatcha 2** (arXiv:2602.10908) **[PDF✓]**. Fuzzy matching at trillion-token scale. It finds contamination that exact match misses. Code: `github.com/softmatcha/softmatcha2` (verified).
- **Evals not in C4:** LatestEval (arXiv:2312.12343) **[PDF✓]**, Automating Dataset Updates (arXiv:2402.11894) **[PDF✓]**, and "Stop Uploading Test Data in Plain Text" (arXiv:2305.10160) **[PDF✓]**.
- **Safety evals in C4:** **I found no paper among the 829 that measures this.** This is open.

### Ideas 11 and 12: ethnic, racial and gender bias, including against teens and women

- **Who's in and who's out?** (Hong, **Agnew** et al.; arXiv:2405.08209) **[PDF✓]**. Reuses **Dodge's exact identity regexes** on image-text CLIP filtering. LGBTQ+ terms are again excluded at higher rates. Code: `github.com/hongrachel/clip-filtering-bias` (verified).
- **From Data to Model in Bias: political bias in C4** (WSDM 2026; DOI 10.1145/3773966.3777990) **[ABS]**. 15 topics, LLM persona annotation, equivalence testing. Reports left-leaning bias in social and cultural topics.
- **Protected-attribute detection and regard classification** (Udagawa et al.; arXiv:2504.14212) **[PDF✓]**. A corpus-level pipeline for measuring bias, positioned as filling a gap left by Dodge et al.
- **Model checkpoints for "then vs now":** Pythia (arXiv:2304.01373) **[PDF✓]** releases "154 checkpoints for each one of the 16 models". It is trained on the Pile, not C4.

### Idea 13: PII and langID

- **A Common Pool of Privacy Problems** (arXiv:2506.17185) **[PDF✓]**: an audit of **DataComp CommonPool**, finding a "significant presence of personally identifiable information despite sanitization efforts".
- **Detecting Personal Information in Training Corpora** (2023) **[ABS]**.
- **WIMBD** has a PII module, saved in `research-empirical-methods/reference-code/wimbd/wimbd/pii/`.
- **What's left:** I found nothing on *whose* PII survives filtering, as your idea asks.

### Idea 14: misinformation and toxicity proportions

See Idea 2 (WIMBD toxicity, Towards Safer Pretraining).

---

## 2. Two results I found that are new, small and checkable

1. **The filter thresholds swapped by Dodge et al. have propagated into tooling.**
   - Dodge §2 says C4 drops lines with fewer than **3 words** and documents with fewer than **5 sentences**.
   - The T5 paper and the TFDS code say the reverse: at least **5 words** per line and at least **3 sentences** per page. See [findings.md](findings.md) §0.
   - Hugging Face **datatrove**'s `C4QualityFilter` has defaults `min_num_sentences=5, min_words_per_line=3`. Its docstring says "fewer than 5 sentences … at least 3 words" while citing the T5 paper and linking the TFDS code.
   - datatrove's `examples/fineweb.py` uses `C4QualityFilter` with those defaults, changing only `filter_no_terminal_punct=False`.
   - The FineWeb paper (§3.5) does not state thresholds, so **I can't say what FineWeb's production run used, only what the published example does.**
   - Quantifying how much the swap changes what is kept, on a WET sample, is a cheap, concrete side result.
2. **AAL blocklist removal: 37.2% (computed from Data Caricatures' counts) vs 42% (Dodge).** Same classifier, same corpora, different number. Reproducing it yourself and explaining the gap would be a strong reported discrepancy, which the brief explicitly rewards.

---

## 3. Recommended extensions, checked against the brief's four criteria

The brief asks for: a hypothesis stated in advance, a control or baseline, quantified uncertainty, and an alternative explanation.

### A. Is the blocklist's dialect gap explained by document length? *(a construct test)*

- **Claim tested:** F5. AAE documents are removed at 42% and WAE at 6.2%.
- **Alternative explanation, and the baseline:** a document gets removed if *any* of 377 words appears anywhere in it. So the chance of removal rises with document length. The dialect classifier also behaves differently with length: it returns `None` when fewer than 20% of tokens are in its vocabulary, and it was trained on tweets. If AAE-classified documents differ in length or source, part of the gap could be length or source rather than dialect.
- **Design:**
  1. Reproduce removal rate by dialect on N `noblocklist` shards.
  2. Re-estimate within **length deciles**, and also with a logistic regression of `removed ~ dialect + log(length) + domain`.
- **Control:** a fake blocklist of 377 frequency-matched neutral words, which should show no dialect gap once length is controlled.
- **Uncertainty:** shard-level bootstrap confidence intervals.
- **Threshold sensitivity:** compare argmax (Dodge), posterior ≥ 0.8 (common in prior work, according to Deas et al.) and posterior ≥ 0.3 (Deas et al.).
- **Cost:** a laptop, streaming from Hugging Face.

### B. Contamination with a built-in negative control *(tests generalisation)*

- **Claim tested:** F3, extended to post-2019 and safety benchmarks.
- **Key control:** C4 is the **April 2019** crawl. A benchmark whose *text* was first written after April 2019 cannot be in C4. Any hits on such a benchmark measure the method's false-positive rate. Benchmarks built from pre-2019 web text, such as Wikipedia-derived ones, *can* match even if the benchmark itself never went online. This separates a leaked test set from a test set that was built out of web text.
- **Tool:** the Infini-gram API over the full C4 train split, querying several casings and punctuation variants to approximate Dodge's normalisation. Validate by reproducing a few Table 4 rows first.
- **Uncertainty:** Wilson confidence intervals per benchmark.
- **Alternative explanations:** generic short items such as "Yes." match by chance, so report results by item-length bucket. Normalisation choice also matters, so report a sensitivity table.

### C. Apply the same filters to a 2025/26 crawl *(tests robustness over time)*

- **Design:** run the exact TFDS v3.0.1 `clean_page` and blocklist steps on a small sample of WET files from `CC-MAIN-2019-18` and from a recent dump.
- **Control:** the 2019 sample should reproduce C4's survival rate (about 92.4% of `noblocklist` survives the blocklist) and C4's identity-term PMI pattern.
- **Measure:** whether exclusion rates for identity terms and the top-domain mix (patents, .mil) have shifted.
- **Cost:** moderate. Each WET file must be downloaded. I have not checked current file sizes.
- **Existing data:** "Practical Datasets" (ICWSM 2025) already has per-snapshot domain statistics, so cite and reuse it rather than recompute domains.

---

## 4. Related, but not a citing paper (verified)

- **FineWeb** (Penedo et al.; arXiv:2406.17557). No citation of Dodge et al. in the PDF. Relevant because it re-applies C4's heuristic filters per Common Crawl dump (§3.5): "the Terminal punctuation filter … removes around 30% of all tokens."
