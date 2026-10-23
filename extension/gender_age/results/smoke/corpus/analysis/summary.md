# Gender and age in C4: exclusion and sentiment

1 en.noblocklist shards, 20,000 documents, 7.32% excluded by the blocklist.

## Exclusion by term (PMI with being excluded; 95% CI over domains)

| term | docs | excluded_% | pmi | pmi_lo | pmi_hi | pmi_nonmechanical | paper_fig5 |
|---|---|---|---|---|---|---|---|
| middle-aged man | 4 | 100.00 | 2.61 | 2.57 | 2.65 | 2.61 |  |
| Asian women | 14 | 92.86 | 2.54 | 2.33 | 2.65 | 2.54 |  |
| Latina/o women | 4 | 75.00 | 2.33 | 1.50 | 2.65 | 2.33 |  |
| Asian men | 4 | 75.00 | 2.33 | 1.25 | 2.65 | 2.33 |  |
| teen boy | 18 | 61.11 | 2.12 | 1.59 | 2.43 | 2.12 |  |
| middle-aged | 23 | 60.87 | 2.12 | 1.76 | 2.29 | 2.12 |  |
| teen girl | 25 | 56.00 | 2.03 | 1.47 | 2.31 | 2.03 |  |
| Black men | 29 | 55.17 | 2.02 | 1.60 | 2.29 | 2.02 |  |
| Black women | 33 | 54.55 | 2.01 | 1.63 | 2.23 | 2.01 |  |
| gentleman | 60 | 53.33 | 1.99 | 1.75 | 2.18 | 1.99 |  |
| old woman | 33 | 51.52 | 1.95 | 1.60 | 2.19 | 1.95 |  |
| White men | 22 | 50.00 | 1.92 | 1.15 | 2.30 | 1.92 |  |
| African American men | 6 | 50.00 | 1.92 |  | 2.62 | 1.92 |  |
| Latina/o men | 4 | 50.00 | 1.92 |  | 2.62 | 1.92 |  |
| Chinese men | 4 | 50.00 | 1.92 |  | 2.64 | 1.92 |  |
| Indian women | 4 | 50.00 | 1.92 |  | 2.63 | 1.92 |  |
| middle-aged woman | 4 | 50.00 | 1.92 |  | 2.61 | 1.92 |  |
| Arab men | 2 | 50.00 | 1.92 |  | 2.65 | 1.92 |  |
| White women | 20 | 50.00 | 1.92 | 1.51 | 2.22 | 1.92 |  |
| female | 407 | 44.23 | 1.80 | 1.66 | 1.91 | 1.80 | 1.82 |
| young woman | 59 | 44.07 | 1.80 | 1.42 | 2.03 | 1.80 |  |
| Jewish women | 7 | 42.86 | 1.77 |  | 2.41 | 1.77 |  |
| old man | 51 | 41.18 | 1.73 | 1.38 | 1.93 | 1.73 |  |
| male | 304 | 40.79 | 1.72 | 1.62 | 1.84 | 1.72 | 1.90 |
| girl | 917 | 40.24 | 1.70 | 1.65 | 1.79 | 1.70 |  |
| Jewish men | 5 | 40.00 | 1.70 |  | 2.64 | 1.70 |  |
| Arab women | 5 | 40.00 | 1.70 |  | 2.35 | 1.70 |  |
| Chinese women | 5 | 40.00 | 1.70 |  | 2.23 | 1.70 |  |
| teen | 498 | 39.96 | 1.70 | 1.60 | 1.80 | 1.70 |  |
| lady | 423 | 37.35 | 1.63 | 1.47 | 1.72 | 1.63 |  |
| little boy | 29 | 34.48 | 1.55 | 1.02 | 1.97 | 1.55 |  |
| Hispanic men | 3 | 33.33 | 1.52 |  | 2.63 | 1.52 |  |
| Indian men | 3 | 33.33 | 1.52 |  | 2.62 | 1.52 |  |
| boy | 556 | 33.27 | 1.51 | 1.40 | 1.64 | 1.51 |  |
| wife | 553 | 30.56 | 1.43 | 1.31 | 1.52 | 1.43 |  |
| young man | 89 | 30.34 | 1.42 | 1.04 | 1.66 | 1.42 |  |
| women | 1415 | 29.68 | 1.40 | 1.34 | 1.46 | 1.40 | 1.73 |
| Muslim women | 7 | 28.57 | 1.36 |  | 2.05 | 1.36 |  |
| adult | 655 | 27.79 | 1.33 | 1.24 | 1.46 | 1.33 |  |
| Jewish (paper) | 98 | 27.55 | 1.33 | 1.02 | 1.51 | 1.33 |  |
| men | 1811 | 27.17 | 1.31 | 1.25 | 1.36 | 1.31 | 1.65 |
| little girl | 59 | 27.12 | 1.31 | 0.88 | 1.64 | 1.31 |  |
| mother | 950 | 26.74 | 1.30 | 1.19 | 1.39 | 1.30 |  |
| husband | 464 | 26.08 | 1.27 | 1.09 | 1.42 | 1.27 |  |
| father | 689 | 25.69 | 1.26 | 1.13 | 1.34 | 1.26 |  |
| Arab (paper) | 60 | 23.33 | 1.16 | 0.79 | 1.48 | 1.16 |  |
| son | 604 | 22.35 | 1.12 | 0.98 | 1.24 | 1.12 |  |
| she/her | 3335 | 21.83 | 1.09 | 1.05 | 1.15 | 1.09 |  |
| daughter | 421 | 20.19 | 1.01 | 0.79 | 1.17 | 1.01 |  |
| young adult | 176 | 17.05 | 0.85 | 0.54 | 1.10 | 0.85 |  |
| child | 2258 | 16.25 | 0.80 | 0.72 | 0.89 | 0.80 |  |
| elderly | 235 | 16.17 | 0.79 | 0.41 | 1.05 | 0.79 |  |
| he/him | 5153 | 16.03 | 0.78 | 0.73 | 0.83 | 0.78 |  |
| person | 4830 | 14.51 | 0.68 | 0.62 | 0.73 | 0.68 |  |
| African American women | 6 | 0.00 | -inf |  |  | -inf |  |
| Muslim men | 2 | 0.00 | -inf |  |  | -inf |  |
| Hispanic women | 0 |  |  |  |  |  |  |
| European women | 0 |  |  |  |  |  |  |
| European men | 0 |  |  |  |  |  |  |

## Exclusion: female minus male PMI for matched pairs

| female | male | gap | ci_lo | ci_hi | gap_nonmechanical | nm_lo | nm_hi |
|---|---|---|---|---|---|---|---|
| women | men | 0.09 | -0.01 | 0.16 | 0.09 | -0.01 | 0.16 |
| female | male | 0.08 | -0.05 | 0.20 | 0.08 | -0.05 | 0.20 |
| girl | boy | 0.19 | 0.07 | 0.31 | 0.18 | 0.06 | 0.30 |
| lady | gentleman | -0.36 | -0.56 | -0.09 | -0.36 | -0.56 | -0.09 |
| mother | father | 0.04 | -0.07 | 0.19 | 0.04 | -0.07 | 0.19 |
| wife | husband | 0.16 | -0.03 | 0.35 | 0.16 | -0.03 | 0.35 |
| daughter | son | -0.10 | -0.37 | 0.10 | -0.10 | -0.37 | 0.10 |
| she/her | he/him | 0.31 | 0.25 | 0.36 | 0.31 | 0.25 | 0.36 |
| teen girl | teen boy | -0.09 | -0.69 | 0.49 | -0.09 | -0.69 | 0.49 |
| little girl | little boy | -0.24 | -0.87 | 0.33 | -0.24 | -0.87 | 0.33 |
| young woman | young man | 0.37 | -0.05 | 0.67 | 0.37 | -0.05 | 0.67 |
| middle-aged woman | middle-aged man | -0.69 |  | 0.00 | -0.69 |  | 0.00 |
| old woman | old man | 0.22 | -0.13 | 0.59 | 0.22 | -0.13 | 0.59 |
| Jewish women | Jewish men | 0.07 |  |  | 0.07 |  |  |
| Arab women | Arab men | -0.22 | -1.95 |  | -0.22 | -1.95 |  |
| Black women | Black men | -0.01 | -0.45 | 0.40 | -0.01 | -0.45 | 0.40 |
| White women | White men | 0.00 | -0.47 | 0.78 | 0.00 | -0.47 | 0.78 |
| Asian women | Asian men | 0.21 | -0.29 | 1.24 | 0.21 | -0.29 | 1.24 |
| Hispanic women | Hispanic men |  |  |  |  |  |  |
| Latina/o women | Latina/o men | 0.41 | -0.67 |  | 0.41 | -0.67 |  |
| African American women | African American men | -inf |  |  | -inf |  |  |
| Indian women | Indian men | 0.41 |  |  | 0.41 |  |  |
| European women | European men |  |  |  |  |  |  |
| Chinese women | Chinese men | -0.22 |  |  | -0.22 |  |  |
| Muslim women | Muslim men | inf |  |  | inf |  |  |

URL-year coverage (documents): {'<=2009': 443, '2010-12': 663, '2013-15': 927, '2016-17': 809, '2018-19': 994, 'none': 16164}

## Blocklist removal by dialect and gender mention (paper overall: AAE 42%, Hispanic 32%, WAE 6.2%)

| dialect | documents mentioning | docs | removed_% | ci_lo | ci_hi |
|---|---|---|---|---|---|
| AAE | all | 31 | 45.16 | 31.46 | 60.41 |
| AAE | female | 9 | 66.67 | 25.80 | 91.54 |
| AAE | male | 6 | 66.67 | 25.00 | 100.00 |
| Hispanic | all | 25 | 48.00 | 28.73 | 63.88 |
| Hispanic | female | 9 | 77.78 | 42.25 | 100.00 |
| Hispanic | male | 4 | 75.00 | 21.12 | 100.00 |
| Other | all | 329 | 6.38 | 4.67 | 9.45 |
| Other | female | 24 | 29.17 | 16.90 | 50.00 |
| Other | male | 30 | 26.67 | 18.34 | 43.16 |
| WAE | all | 19604 | 7.23 | 6.96 | 7.55 |
| WAE | female | 3165 | 24.33 | 22.76 | 25.37 |
| WAE | male | 3107 | 22.53 | 21.43 | 23.96 |
| unassigned | all | 11 | 0.00 | 0.00 | 0.00 |
| unassigned | female | 0 |  |  |  |
| unassigned | male | 0 |  |  |  |

## Positive control: Jewish - Arab positivity gap (paper-style matching)

{"SocialSent": {"ours": -13.94, "ci": [-30.8, 1.38], "arm_b_c4en_12_shards": 7.1}, "VADER": {"ours": 3.69, "ci": [-10.43, 14.28], "arm_b_c4en_12_shards": 2.0}}

## Sentiment: female minus male positivity (points), kept documents

| documents | lexicon | female | male | gap | ci_lo | ci_hi |
|---|---|---|---|---|---|---|
| kept (C4.EN) | SocialSent | women | men | -3.73 | -7.77 | -0.21 |
| kept (C4.EN) | SocialSent | female | male | -0.96 | -7.24 | 8.99 |
| kept (C4.EN) | SocialSent | girl | boy | -1.35 | -5.53 | 3.54 |
| kept (C4.EN) | SocialSent | lady | gentleman | 10.86 | -13.10 | 27.01 |
| kept (C4.EN) | SocialSent | mother | father | -2.45 | -7.35 | 1.09 |
| kept (C4.EN) | SocialSent | wife | husband | -2.32 | -7.41 | 2.42 |
| kept (C4.EN) | SocialSent | daughter | son | 5.22 | 0.90 | 10.78 |
| kept (C4.EN) | SocialSent | she/her | he/him | 1.04 | -0.90 | 2.91 |
| kept (C4.EN) | SocialSent | teen girl | teen boy | -5.15 | -35.18 | 19.10 |
| kept (C4.EN) | SocialSent | little girl | little boy | 5.26 | -22.01 | 24.36 |
| kept (C4.EN) | SocialSent | young woman | young man | 1.80 | -18.46 | 21.91 |
| kept (C4.EN) | SocialSent | middle-aged woman | middle-aged man |  |  |  |
| kept (C4.EN) | SocialSent | old woman | old man | -1.92 | -58.32 | 23.67 |
| kept (C4.EN) | SocialSent | Jewish women | Jewish men | -30.00 | -48.07 | 50.00 |
| kept (C4.EN) | SocialSent | Arab women | Arab men |  |  |  |
| kept (C4.EN) | SocialSent | Black women | Black men | -21.15 | -46.93 | 10.12 |
| kept (C4.EN) | SocialSent | White women | White men | 16.91 | -18.93 | 33.33 |
| kept (C4.EN) | SocialSent | Asian women | Asian men |  |  |  |
| kept (C4.EN) | SocialSent | Hispanic women | Hispanic men |  |  |  |
| kept (C4.EN) | SocialSent | Latina/o women | Latina/o men | -50.00 | -66.67 | 0.00 |
| kept (C4.EN) | SocialSent | African American women | African American men | -20.00 | -33.21 | 0.00 |
| kept (C4.EN) | SocialSent | Indian women | Indian men |  |  |  |
| kept (C4.EN) | SocialSent | European women | European men |  |  |  |
| kept (C4.EN) | SocialSent | Chinese women | Chinese men | 44.44 | 40.00 | 50.00 |
| kept (C4.EN) | SocialSent | Muslim women | Muslim men | -13.33 | -24.24 | 0.00 |
| kept (C4.EN) | VADER | women | men | 4.39 | 1.88 | 7.26 |
| kept (C4.EN) | VADER | female | male | 6.96 | 2.55 | 11.01 |
| kept (C4.EN) | VADER | girl | boy | 1.57 | -2.30 | 3.82 |
| kept (C4.EN) | VADER | lady | gentleman | 11.56 | -0.17 | 21.99 |
| kept (C4.EN) | VADER | mother | father | 1.60 | -0.87 | 3.79 |
| kept (C4.EN) | VADER | wife | husband | 0.30 | -2.56 | 3.50 |
| kept (C4.EN) | VADER | daughter | son | 5.51 | 2.30 | 9.93 |
| kept (C4.EN) | VADER | she/her | he/him | 3.85 | 2.52 | 4.90 |
| kept (C4.EN) | VADER | teen girl | teen boy | -15.43 | -51.63 | 21.38 |
| kept (C4.EN) | VADER | little girl | little boy | 4.50 | -13.79 | 21.12 |
| kept (C4.EN) | VADER | young woman | young man | 6.65 | -11.33 | 21.09 |
| kept (C4.EN) | VADER | middle-aged woman | middle-aged man |  |  |  |
| kept (C4.EN) | VADER | old woman | old man | -2.57 | -23.62 | 14.39 |
| kept (C4.EN) | VADER | Jewish women | Jewish men | 1.12 | -9.33 | 35.71 |
| kept (C4.EN) | VADER | Arab women | Arab men | 56.52 | 50.00 | 56.84 |
| kept (C4.EN) | VADER | Black women | Black men | 29.21 | -4.68 | 48.16 |
| kept (C4.EN) | VADER | White women | White men | 2.49 | -22.01 | 25.50 |
| kept (C4.EN) | VADER | Asian women | Asian men | -62.50 | -62.50 | -62.50 |
| kept (C4.EN) | VADER | Hispanic women | Hispanic men |  |  |  |
| kept (C4.EN) | VADER | Latina/o women | Latina/o men | 30.00 | 12.50 | 100.00 |
| kept (C4.EN) | VADER | African American women | African American men | 15.00 | -26.79 | 46.08 |
| kept (C4.EN) | VADER | Indian women | Indian men | -5.00 | -5.00 | -5.00 |
| kept (C4.EN) | VADER | European women | European men |  |  |  |
| kept (C4.EN) | VADER | Chinese women | Chinese men | 26.67 | 15.38 | 35.29 |
| kept (C4.EN) | VADER | Muslim women | Muslim men | 47.37 | 0.00 | 100.00 |
| kept (C4.EN) | UnQover | women | men | 3.82 | -0.62 | 9.96 |
| kept (C4.EN) | UnQover | female | male | 2.81 | -9.61 | 16.80 |
| kept (C4.EN) | UnQover | girl | boy | -5.69 | -14.74 | 2.50 |
| kept (C4.EN) | UnQover | lady | gentleman | 29.13 | -12.89 | 71.55 |
| kept (C4.EN) | UnQover | mother | father | 5.61 | 0.71 | 10.88 |
| kept (C4.EN) | UnQover | wife | husband | 1.10 | -4.63 | 8.56 |
| kept (C4.EN) | UnQover | daughter | son | 7.50 | -1.16 | 14.06 |
| kept (C4.EN) | UnQover | she/her | he/him | 5.36 | 2.41 | 7.12 |
| kept (C4.EN) | UnQover | teen girl | teen boy | 50.00 | 0.00 | 100.00 |
| kept (C4.EN) | UnQover | little girl | little boy | 3.95 | -39.50 | 80.63 |
| kept (C4.EN) | UnQover | young woman | young man | 7.41 | -26.56 | 40.55 |
| kept (C4.EN) | UnQover | middle-aged woman | middle-aged man |  |  |  |
| kept (C4.EN) | UnQover | old woman | old man | 2.27 | -51.11 | 42.88 |
| kept (C4.EN) | UnQover | Jewish women | Jewish men | 0.00 | 0.00 | 0.00 |
| kept (C4.EN) | UnQover | Arab women | Arab men |  |  |  |
| kept (C4.EN) | UnQover | Black women | Black men | -16.67 | -50.00 | 48.25 |
| kept (C4.EN) | UnQover | White women | White men |  |  |  |
| kept (C4.EN) | UnQover | Asian women | Asian men |  |  |  |
| kept (C4.EN) | UnQover | Hispanic women | Hispanic men |  |  |  |
| kept (C4.EN) | UnQover | Latina/o women | Latina/o men |  |  |  |
| kept (C4.EN) | UnQover | African American women | African American men |  |  |  |
| kept (C4.EN) | UnQover | Indian women | Indian men |  |  |  |
| kept (C4.EN) | UnQover | European women | European men |  |  |  |
| kept (C4.EN) | UnQover | Chinese women | Chinese men | 40.00 | 0.00 | 100.00 |
| kept (C4.EN) | UnQover | Muslim women | Muslim men |  |  |  |

## Sentiment by age term, kept documents (SocialSent)

| term | paragraphs | positivity | ci_lo | ci_hi |
|---|---|---|---|---|
| child | 3915 | 75.49 | 73.65 | 78.09 |
| teen | 427 | 65.94 | 59.18 | 72.88 |
| young adult | 216 | 70.39 | 62.35 | 78.35 |
| adult | 697 | 73.62 | 68.85 | 78.06 |
| middle-aged | 9 | 50.00 | 3.21 | 85.71 |
| elderly | 306 | 67.38 | 61.93 | 71.59 |
| person | 8388 | 69.29 | 67.93 | 70.91 |
