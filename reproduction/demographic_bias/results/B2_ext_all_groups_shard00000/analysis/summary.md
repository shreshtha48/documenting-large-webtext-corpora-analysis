# Arm B: C4 paragraph sentiment (1 shards = 1/1024 of C4.EN, 19581 domains)

Extension: Spearman between SocialSent corpus positivity and Table 7 positivity = -0.048

## Positivity

| target | lexicon | pos | neg | sentiment_tokens | positivity | ci_lo | ci_hi | paper |
|---|---|---|---|---|---|---|---|---|
| Jewish | SocialSent | 2693 | 1058 | 3751 | 71.79 | 69.17 | 74.44 | 73.2 |
| Jewish | VADER | 8734 | 4503 | 13237 | 65.98 | 64.15 | 67.75 | nan |
| Jewish | UnQover | 594 | 222 | 816 | 72.79 | 68.65 | 76.93 | nan |
| Arab | SocialSent | 657 | 362 | 1019 | 64.47 | 60.08 | 68.64 | 65.7 |
| Arab | VADER | 3413 | 1906 | 5319 | 64.17 | 61.62 | 66.7 | nan |
| Arab | UnQover | 178 | 101 | 279 | 63.8 | 57.25 | 70.45 | nan |
| Asian | SocialSent | 2142 | 737 | 2879 | 74.4 | 71.85 | 76.99 | nan |
| Asian | VADER | 8093 | 2922 | 11015 | 73.47 | 71.31 | 75.61 | nan |
| Asian | UnQover | 511 | 109 | 620 | 82.42 | 78.81 | 86.37 | nan |
| Caucasian | SocialSent | 64 | 52 | 116 | 55.17 | 36.63 | 69.09 | nan |
| Caucasian | VADER | 270 | 157 | 427 | 63.23 | 53.88 | 70.26 | nan |
| Caucasian | UnQover | 27 | 7 | 34 | 79.41 | 66.67 | 90.92 | nan |
| European | SocialSent | 5571 | 2166 | 7737 | 72.0 | 70.64 | 73.39 | nan |
| European | VADER | 26093 | 10129 | 36222 | 72.04 | 71.19 | 72.89 | nan |
| European | UnQover | 1645 | 430 | 2075 | 79.28 | 76.98 | 81.35 | nan |
| White | SocialSent | 6416 | 1442 | 7858 | 81.65 | 80.12 | 83.17 | nan |
| White | VADER | 19585 | 6817 | 26402 | 74.18 | 72.82 | 75.55 | nan |
| White | UnQover | 1632 | 324 | 1956 | 83.44 | 81.18 | 85.75 | nan |
| Alaskan | SocialSent | 177 | 31 | 208 | 85.1 | 79.23 | 90.45 | nan |
| Alaskan | VADER | 553 | 135 | 688 | 80.38 | 75.65 | 84.77 | nan |
| Alaskan | UnQover | 58 | 4 | 62 | 93.55 | 83.72 | 100.0 | nan |
| Hispanic | SocialSent | 247 | 138 | 385 | 64.16 | 56.74 | 71.78 | nan |
| Hispanic | VADER | 1073 | 470 | 1543 | 69.54 | 64.41 | 74.13 | nan |
| Hispanic | UnQover | 49 | 23 | 72 | 68.06 | 55.88 | 80.68 | nan |
| Native American | SocialSent | 525 | 120 | 645 | 81.4 | 76.48 | 85.77 | nan |
| Native American | VADER | 1578 | 499 | 2077 | 75.97 | 72.04 | 80.03 | nan |
| Native American | UnQover | 107 | 12 | 119 | 89.92 | 82.46 | 96.88 | nan |
| South-American | SocialSent | 167 | 66 | 233 | 71.67 | 65.48 | 77.91 | nan |
| South-American | VADER | 609 | 214 | 823 | 74.0 | 68.67 | 79.34 | nan |
| South-American | UnQover | 59 | 17 | 76 | 77.63 | 66.32 | 89.23 | nan |
| African-American | SocialSent | 804 | 354 | 1158 | 69.43 | 64.38 | 74.3 | nan |
| African-American | VADER | 2516 | 1306 | 3822 | 65.83 | 63.17 | 68.71 | nan |
| African-American | UnQover | 149 | 68 | 217 | 68.66 | 61.26 | 75.92 | nan |
| Latino | SocialSent | 299 | 119 | 418 | 71.53 | 64.15 | 78.98 | nan |
| Latino | VADER | 1163 | 432 | 1595 | 72.92 | 67.57 | 77.92 | nan |
| Latino | UnQover | 52 | 37 | 89 | 58.43 | 44.27 | 73.85 | nan |
| Middle-Eastern | SocialSent | 291 | 58 | 349 | 83.38 | 78.06 | 88.12 | nan |
| Middle-Eastern | VADER | 807 | 271 | 1078 | 74.86 | 71.15 | 78.57 | nan |
| Middle-Eastern | UnQover | 60 | 16 | 76 | 78.95 | 66.25 | 88.9 | nan |
| Black | SocialSent | 7257 | 1480 | 8737 | 83.06 | 81.8 | 84.33 | nan |
| Black | VADER | 20898 | 7530 | 28428 | 73.51 | 72.33 | 74.71 | nan |
| Black | UnQover | 1505 | 341 | 1846 | 81.53 | 79.22 | 83.82 | nan |
| African | SocialSent | 3269 | 1175 | 4444 | 73.56 | 70.63 | 76.09 | nan |
| African | VADER | 12370 | 4683 | 17053 | 72.54 | 70.96 | 74.21 | nan |
| African | UnQover | 763 | 255 | 1018 | 74.95 | 71.58 | 77.89 | nan |

## Gap (Jewish - Arab)

| lexicon | gap | ci_lo | ci_hi | share_of_bootstrap_gap_>0 | paper_gap |
|---|---|---|---|---|---|
| SocialSent | 7.32 | 2.26 | 12.65 | 1.0 | 7.5 |
| VADER | 1.82 | -1.14 | 4.77 | 0.89 | 2.0 |
| UnQover | 8.99 | 1.72 | 16.65 | 0.99 | 1.4 |

## Robustness of the gap (each row changes one thing)

| variant | lexicon | Jewish | Arab | gap | ci_lo | ci_hi | Jewish_paragraphs | Arab_paragraphs | paper_gap |
|---|---|---|---|---|---|---|---|---|---|
| as in paper | SocialSent | 71.79 | 64.47 | 7.32 | 2.26 | 12.65 | 2939 | 1384 | 7.5 |
| as in paper | VADER | 65.98 | 64.17 | 1.82 | -1.14 | 4.77 | 2939 | 1384 | 2.0 |
| as in paper | UnQover | 72.79 | 63.8 | 8.99 | 1.72 | 16.65 | 2939 | 1384 | 1.4 |
| Arab: no 'United Arab Emirates' paragraphs | SocialSent | 71.79 | 61.47 | 10.32 | 4.54 | 15.6 | 2939 | 1064 | 7.5 |
| Arab: no 'United Arab Emirates' paragraphs | VADER | 65.98 | 58.36 | 7.62 | 4.71 | 10.61 | 2939 | 1064 | 2.0 |
| Arab: no 'United Arab Emirates' paragraphs | UnQover | 72.79 | 62.4 | 10.39 | 2.57 | 17.35 | 2939 | 1064 | 1.4 |
| Arab: no geopolitical 'Arab' phrases | SocialSent | 71.79 | 65.47 | 6.33 | 0.66 | 12.97 | 2939 | 753 | 7.5 |
| Arab: no geopolitical 'Arab' phrases | VADER | 65.98 | 59.56 | 6.42 | 2.83 | 10.08 | 2939 | 753 | 2.0 |
| Arab: no geopolitical 'Arab' phrases | UnQover | 72.79 | 61.14 | 11.65 | 1.88 | 20.93 | 2939 | 753 | 1.4 |
| no 1-letter tokens or SocialSent noise words | SocialSent | 71.33 | 64.43 | 6.89 | 1.63 | 12.46 | 2939 | 1384 | 7.5 |
| no 1-letter tokens or SocialSent noise words | VADER | 65.87 | 64.13 | 1.74 | -1.2 | 4.7 | 2939 | 1384 | 2.0 |
| no 1-letter tokens or SocialSent noise words | UnQover | 72.79 | 63.8 | 8.99 | 1.72 | 16.65 | 2939 | 1384 | 1.4 |
| no religious words | SocialSent | 66.76 | 63.21 | 3.55 | -1.82 | 9.28 | 2939 | 1384 | 7.5 |
| no religious words | VADER | 64.13 | 64.01 | 0.12 | -2.9 | 3.15 | 2939 | 1384 | 2.0 |
| no religious words | UnQover | 72.79 | 63.8 | 8.99 | 1.72 | 16.65 | 2939 | 1384 | 1.4 |

## Domains (SocialSent)

| domain | target | lexicon | sentiment_tokens | positivity | paper |
|---|---|---|---|---|---|
| New York Times | Jewish | SocialSent | 17 | 76.47 | 74.0 |
| New York Times | Arab | SocialSent | 1 | 100.0 | 69.5 |
| New York Times | Asian | SocialSent | 8 | 100.0 | nan |
| New York Times | Caucasian | SocialSent | 0 | nan | nan |
| New York Times | European | SocialSent | 3 | 33.33 | nan |
| New York Times | White | SocialSent | 5 | 80.0 | nan |
| New York Times | Alaskan | SocialSent | 0 | nan | nan |
| New York Times | Hispanic | SocialSent | 0 | nan | nan |
| New York Times | Native American | SocialSent | 0 | nan | nan |
| New York Times | South-American | SocialSent | 0 | nan | nan |
| New York Times | African-American | SocialSent | 1 | 100.0 | nan |
| New York Times | Latino | SocialSent | 0 | nan | nan |
| New York Times | Middle-Eastern | SocialSent | 0 | nan | nan |
| New York Times | Black | SocialSent | 0 | nan | nan |
| New York Times | African | SocialSent | 5 | 80.0 | nan |
| Al Jazeera | Jewish | SocialSent | 7 | 42.86 | 42.5 |
| Al Jazeera | Arab | SocialSent | 9 | 66.67 | 42.8 |
| Al Jazeera | Asian | SocialSent | 0 | nan | nan |
| Al Jazeera | Caucasian | SocialSent | 0 | nan | nan |
| Al Jazeera | European | SocialSent | 9 | 11.11 | nan |
| Al Jazeera | White | SocialSent | 0 | nan | nan |
| Al Jazeera | Alaskan | SocialSent | 0 | nan | nan |
| Al Jazeera | Hispanic | SocialSent | 0 | nan | nan |
| Al Jazeera | Native American | SocialSent | 0 | nan | nan |
| Al Jazeera | South-American | SocialSent | 0 | nan | nan |
| Al Jazeera | African-American | SocialSent | 0 | nan | nan |
| Al Jazeera | Latino | SocialSent | 0 | nan | nan |
| Al Jazeera | Middle-Eastern | SocialSent | 3 | 33.33 | nan |
| Al Jazeera | Black | SocialSent | 4 | 25.0 | nan |
| Al Jazeera | African | SocialSent | 9 | 11.11 | nan |

## Scale vs the paper (paper numbers scaled by shards/1024)

| target | paragraphs | words_in_paragraphs | paper_words_scaled_to_sample | socialsent_tokens | paper_socialsent_tokens_scaled_to_sample |
|---|---|---|---|---|---|
| Jewish | 2939 | 262298 | 243945.31 | 3751 | 3320.31 |
| Arab | 1384 | 106309 | 131640.62 | 1019 | 1171.88 |
| Asian | 2901 | 229851 | nan | 2879 | nan |
| Caucasian | 92 | 8700 | nan | 116 | nan |
| European | 10059 | 780515 | nan | 7737 | nan |
| White | 7172 | 519252 | nan | 7858 | nan |
| Alaskan | 185 | 14124 | nan | 208 | nan |
| Hispanic | 482 | 34982 | nan | 385 | nan |
| Native American | 478 | 41929 | nan | 645 | nan |
| South-American | 226 | 17848 | nan | 233 | nan |
| African-American | 932 | 77777 | nan | 1158 | nan |
| Latino | 433 | 32304 | nan | 418 | nan |
| Middle-Eastern | 286 | 22649 | nan | 349 | nan |
| Black | 7363 | 557801 | nan | 8737 | nan |
| African | 4508 | 353162 | nan | 4444 | nan |

