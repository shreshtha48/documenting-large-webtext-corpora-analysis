# §5 reproduction: 10 shards of en.noblocklist, 3,841,713 documents, 1,741,415 domains

- Excluded by the blocklist: 7.24% (expected about 7.6% from 395M vs 365M docs)
- Dialect inference: fast vs exact CVB0 argmax agreement 99.72% on 10,000 documents
- Figure 5: Spearman with the paper's bars (read off the figure) = 0.968
- Manual-check sheet: 100 documents in inspect_lesbian_gay.csv (label them; paper: 22% / 36% non-offensive or non-sexual)

## §5.2 identity PMI (Fig. 5)

| term | docs | excluded | excluded_% | pmi | ci_lo | ci_hi | paper_pmi_read_off_fig5 | category |
|---|---|---|---|---|---|---|---|---|
| lesbian | 16407 | 13602 | 82.904 | 2.438 | 2.427 | 2.448 | 2.44 | sexual orientation |
| heterosexual | 2441 | 1826 | 74.805 | 2.335 | 2.276 | 2.372 | 2.34 | sexual orientation |
| homosexual | 4502 | 3165 | 70.302 | 2.273 | 2.239 | 2.299 | 2.3 | sexual orientation |
| bisexual | 4491 | 3084 | 68.671 | 2.249 | 2.208 | 2.28 | 2.24 | sexual orientation |
| gay | 28521 | 18651 | 65.394 | 2.2 | 2.187 | 2.216 | 2.34 | sexual orientation |
| latina / latino | 10800 | 5825 | 53.935 | 2.008 | 1.985 | 2.032 | 1.92 | race / ethnicity |
| non-binary | 623 | 272 | 43.66 | 1.796 | 1.701 | 1.881 | 1.84 | gender identity |
| male | 64165 | 27041 | 42.143 | 1.761 | 1.749 | 1.774 | 1.9 | gender identity |
| caucasian | 2396 | 980 | 40.902 | 1.731 | 1.678 | 1.782 | 1.68 | race / ethnicity |
| female | 77419 | 30815 | 39.803 | 1.704 | 1.692 | 1.716 | 1.82 | gender identity |
| asian | 40377 | 14363 | 35.572 | 1.591 | 1.568 | 1.612 | 1.72 | race / ethnicity |
| trans | 14830 | 5241 | 35.341 | 1.585 | 1.56 | 1.611 | 1.75 | gender identity |
| women | 272681 | 82527 | 30.265 | 1.43 | 1.422 | 1.438 | 1.73 | gender identity |
| african american | 13043 | 3798 | 29.119 | 1.391 | 1.362 | 1.418 | 1.44 | race / ethnicity |
| man | 345082 | 97259 | 28.184 | 1.359 | 1.351 | 1.366 | 1.65 | gender identity |
| jewish | 27553 | 7669 | 27.834 | 1.346 | 1.322 | 1.372 | 1.47 | religion |
| muslim | 20694 | 5682 | 27.457 | 1.333 | 1.307 | 1.364 | 1.5 | religion |
| straight | 102455 | 25047 | 24.447 | 1.216 | 1.202 | 1.229 | 1.27 | sexual orientation |
| black | 223588 | 53402 | 23.884 | 1.193 | 1.183 | 1.203 | 1.43 | race / ethnicity |
| christian | 61677 | 14148 | 22.939 | 1.153 | 1.133 | 1.17 | 1.37 | religion |
| white | 239069 | 45494 | 19.03 | 0.966 | 0.956 | 0.977 | 1.12 | race / ethnicity |
| european | 87774 | 13889 | 15.824 | 0.781 | 0.761 | 0.8 | 0.84 | race / ethnicity |

## §5.3 dialects

| dialect | docs_noblocklist | excluded | removal_% | removal_ci_lo | removal_ci_hi | paper_removal_% | share_of_kept_% | share_ci_lo | share_ci_hi | paper_share_of_C4.EN_% |
|---|---|---|---|---|---|---|---|---|---|---|
| AAE | 4129 | 1477 | 35.771 | 33.664 | 38.136 | 42.0 | 0.074 | 0.071 | 0.078 | 0.07 |
| Hispanic | 3502 | 1192 | 34.038 | 32.259 | 36.136 | 32.0 | 0.065 | 0.061 | 0.068 | 0.09 |
| Other | 67501 | 4427 | 6.558 | 6.26 | 6.844 | 7.2 | 1.77 | 1.728 | 1.817 | nan |
| WAE | 3763813 | 271169 | 7.205 | 7.147 | 7.275 | 6.2 | 98.013 | 97.962 | 98.063 | 97.8 |
| unassigned | 2768 | 0 | 0.0 | 0.0 | 0.0 | nan | 0.078 | 0.06 | 0.094 | nan |

## §5.1 clusters (100,000 docs, k=50; paper: 16 sexual clusters, 31%)

| seed | sexual_clusters | sexual_share_% |
|---|---|---|
| 0 | 15 | 30.8 |
| 1 | 14 | 28.9 |
| 2 | 15 | 28.5 |

