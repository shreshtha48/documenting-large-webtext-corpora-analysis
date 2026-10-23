# Arm A experiments (all runs, reproduced or not)

Paper (Dodge et al. 2021, Table 7): Jewish 67.1%, Arab 37.0%, spread 30.5.

| experiment | model | rule | contexts_scored | spearman_vs_table7 | answer_flips_with_sentiment_% | share_option_a | Jewish_% | Jewish_CI | Arab_% | Arab_CI | spread_max_minus_min | sd_noise_corrected | slope_vs_paper | spearman_no_negation_attrs | coverage_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3_full_small_ext | unifiedqa-t5-small | mean | 147000 | -0.002 | 18.9 | 0.5678 | 47.4 | 46.0-48.8 | 47.3 | 45.4-49.3 | 7.9 | 2.07 | -0.009 | -0.109 | True |
| E3_full_small_ext | unifiedqa-t5-small | sum | 147000 | -0.013 | 18.1 | 0.5651 | 47.4 | 46.0-48.7 | 47.4 | 45.5-49.4 | 8.0 | 2.05 | -0.016 | -0.1001 | True |
| E4_size_test_sample | unifiedqa-t5-small | mean | 1516 | -0.055 | 18.4 | 0.5755 | 47.5 | 44.9-50.0 | 48.5 | 45.0-52.0 | 9.6 | 2.14 | -0.013 | -0.0215 | True |
| E4_size_test_sample | unifiedqa-t5-base | mean | 1516 | -0.052 | 25.5 | 0.5762 | 44.6 | 40.7-48.5 | 54.9 | 52.0-57.8 | 19.7 | 4.93 | 0.017 | -0.1162 | True |
| E4_size_test_sample | unifiedqa-t5-large | mean | 1516 | 0.488 | 28.2 | 0.3203 | 55.0 | 50.5-59.3 | 44.6 | 40.5-48.3 | 31.4 | 6.9 | 0.363 | 0.4057 | True |
| E4_size_test_sample | unifiedqa-t5-3b | mean | 1516 | 0.808 | 49.4 | 0.4337 | 65.6 | 60.6-70.8 | 30.9 | 27.2-34.9 | 45.8 | 13.22 | 1.176 | 0.8186 | True |
| E4_size_test_sample | unifiedqa-t5-small | sum | 1516 | -0.002 | 17.5 | 0.5775 | 47.5 | 44.9-50.2 | 48.8 | 45.9-51.8 | 8.4 | 1.79 | -0.003 | 0.0643 | True |
| E4_size_test_sample | unifiedqa-t5-base | sum | 1516 | 0.03 | 26.2 | 0.5788 | 45.3 | 41.1-49.5 | 55.6 | 52.7-58.6 | 21.7 | 5.17 | 0.06 | 0.034 | True |
| E4_size_test_sample | unifiedqa-t5-large | sum | 1516 | 0.483 | 30.1 | 0.3189 | 54.7 | 50.5-59.1 | 43.1 | 39.4-46.9 | 35.9 | 7.92 | 0.415 | 0.5112 | True |
| E4_size_test_sample | unifiedqa-t5-3b | sum | 1516 | 0.792 | 49.8 | 0.4383 | 65.1 | 59.9-70.1 | 31.4 | 27.6-35.7 | 46.3 | 13.31 | 1.197 | 0.8043 | True |
| E6_full_3b | unifiedqa-t5-3b | mean | 147000 | 0.844 | 50.7 | 0.4242 | 65.8 | 62.8-68.9 | 34.4 | 31.5-37.4 | 43.4 | 12.13 | 1.09 | 0.8472 | True |
| E6_full_3b | unifiedqa-t5-3b | sum | 147000 | 0.844 | 50.4 | 0.4244 | 65.7 | 62.7-68.8 | 34.6 | 31.8-37.5 | 43.4 | 12.09 | 1.086 | 0.8472 | True |
