# F5 reproduction: what the blocklist excluded (Dodge et al. 2021, §5 and A.6)

C4.EN drops every document containing a word from LDNOOBW, the "List of Dirty, Naughty, Obscene, and Otherwise Bad
Words". §5 characterises what that removed. This folder reproduces all of §5 on free-tier CPU:
- the clusters of excluded documents (§5.1, A.6, Fig. 6);
- which identity mentions are excluded (§5.2, Fig. 5, Table 6, and the manual check);
- whose English is excluded (§5.3, dialect removal rates).

**Status:** see [RESULTS.md](RESULTS.md). One shard is analysed locally (S0). The 12-shard run (S1) is the Kaggle
notebook's main run and has not been run yet.

## Layout

| Path | What |
|---|---|
| `kaggle/f5_excluded_kaggle.ipynb` | **The notebook to run** (Kaggle, CPU, Internet on; Save & Run All runs in the background, about 1-1.5 h). Rebuilt by `kaggle/build_kaggle_notebook.py` |
| `colab/f5_excluded_colab.ipynb` | Colab alternative (output on Google Drive, resumable; keep the tab open). Upload the .ipynb via File -> Upload notebook; the zip goes into its first cell |
| `colab/build_colab_notebook.py` | Rebuilds the notebook from `scripts/`; run it after editing a script |
| `scripts/scan_noblocklist.py` | One pass per `en.noblocklist` shard: blocklist decision, 22 identity regexes, dialect, samples |
| `scripts/dialect.py` | Blodgett et al. (2016) dialect model ported to Python 3; exact port plus a fast version that is validated against it on every shard |
| `scripts/analyze_excluded.py` | Fig. 5, Table 6, dialect tables and figure, Fig. 6 clustering, the manual-check sheet |
| `data/ldnoobw_en_25e679f.txt` | The blocklist C4 used (TFDS v3.0.1) |
| `data/twitteraae_model/` | Model files from `slanglab/twitteraae` @ `db4d6d1` (MIT); downloaded automatically if missing |
| `results/S0_noblocklist_shard00000/` | One-shard local run |

## Method (with the paper's gaps filled in, and marked as ours)

- **Excluded set.** `en.noblocklist` documents that match C4's own regex (TFDS v3.0.1, lowercased text, list not
  escaped). The paper's definition, "in C4.EN.NOBLOCKLIST but not in C4.EN", gives the same set, because the
  blocklist is the last filter in the pipeline. The pilot matched the released C4.EN for 99.4% of documents.
- **§5.2.** The 22 Table 6 regexes, case-insensitive, with word boundaries (ours). Document-level
  PMI = ln[P(term, excluded) / (P(term) P(excluded))]. The natural log is ours; it fits the paper's 0–2.5 axis. The
  paper prints no numbers, so its Fig. 5 bars were read off the figure (±0.05) and are compared by rank.
- **§5.3.** The Blodgett et al. model (CVB0, alpha=1, 5 passes, scoring gates as in the repo), run on the whole
  document. The argmax gives the dialect, and documents the model cannot score are reported as "unassigned". The
  tokenizer is ours: lowercased runs of letters and digits.
  - The fast inference updates all tokens at once instead of one at a time. Each shard measures its argmax
    agreement with the exact port on 1,000 documents.
  - Following the model's authors, only the AAE and WAE categories are validated.
- **§5.1.** About 100,000 excluded documents (sampled evenly across shards); all settings below are ours.
  - TF-IDF with 50k features, English stop words, min_df=5 and sublinear tf.
  - k-means with k=50, repeated over 3 seeds.
  - t-SNE of a 50-d truncated SVD projection, 5,000 documents shown.
  - A cluster counts as "sexual" when at least 2 of its top 10 terms are blocklist words. The paper labelled by
    hand, so `clusters.csv` has a column for hand labels.
- **CIs:** 95% intervals from a bootstrap over web domains (Poisson weights, B=200). The paper reports none.
