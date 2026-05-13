# Statistical Significance Tests

Paired t-test on macro F1 across seeds (alpha=0.05).
⚠️ 3-seed 반복은 검정력이 낮으므로 해석에 주의가 필요합니다.

| model_a | model_b | metric | n_seeds | mean_diff | t_statistic | p_value | significant | cohens_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_B | A_R | macro_f1 | 3 | 0.0097 | 1.1639 | 0.3645 | False | 0.6720 |
| A_B | B_B | macro_f1 | 3 | -0.0036 | -1.0775 | 0.3939 | False | -0.6221 |
| A_B | B_R | macro_f1 | 3 | 0.0034 | 0.5803 | 0.6204 | False | 0.3350 |
| A_B | C_B | macro_f1 | 3 | -0.0003 | -0.0747 | 0.9473 | False | -0.0431 |
| A_B | C_R | macro_f1 | 3 | 0.0065 | 0.7134 | 0.5496 | False | 0.4119 |
| A_B | D_B | macro_f1 | 3 | -0.0095 | -3.3102 | 0.0804 | False | -1.9111 |
| A_B | D_R | macro_f1 | 3 | 0.0034 | 0.5959 | 0.6117 | False | 0.3440 |
| A_B | TF-IDF + LR | macro_f1 | 3 | 0.0390 | 14.7907 | 0.0045 | True | 8.5394 |
| A_B | TF-IDF + SVM | macro_f1 | 3 | 0.0408 | 15.4671 | 0.0042 | True | 8.9299 |
| A_R | B_B | macro_f1 | 3 | -0.0132 | -1.1401 | 0.3724 | False | -0.6582 |
| A_R | B_R | macro_f1 | 3 | -0.0062 | -2.5190 | 0.1280 | False | -1.4544 |
| A_R | C_B | macro_f1 | 3 | -0.0099 | -0.8729 | 0.4747 | False | -0.5040 |
| A_R | C_R | macro_f1 | 3 | -0.0032 | -0.4832 | 0.6767 | False | -0.2789 |
| A_R | D_B | macro_f1 | 3 | -0.0191 | -2.0506 | 0.1768 | False | -1.1839 |
| A_R | D_R | macro_f1 | 3 | -0.0062 | -1.7210 | 0.2274 | False | -0.9936 |
| A_R | TF-IDF + LR | macro_f1 | 3 | 0.0294 | 4.0506 | 0.0559 | False | 2.3386 |
| A_R | TF-IDF + SVM | macro_f1 | 3 | 0.0312 | 4.2967 | 0.0501 | False | 2.4807 |
| B_B | B_R | macro_f1 | 3 | 0.0070 | 0.7580 | 0.5276 | False | 0.4376 |
| B_B | C_B | macro_f1 | 3 | 0.0033 | 1.6838 | 0.2342 | False | 0.9722 |
| B_B | C_R | macro_f1 | 3 | 0.0101 | 0.8697 | 0.4762 | False | 0.5021 |
| B_B | D_B | macro_f1 | 3 | -0.0059 | -1.6005 | 0.2506 | False | -0.9241 |
| B_B | D_R | macro_f1 | 3 | 0.0070 | 0.7785 | 0.5177 | False | 0.4495 |
| B_B | TF-IDF + LR | macro_f1 | 3 | 0.0426 | 8.3749 | 0.0140 | True | 4.8352 |
| B_B | TF-IDF + SVM | macro_f1 | 3 | 0.0444 | 8.7259 | 0.0129 | True | 5.0379 |
| B_R | C_B | macro_f1 | 3 | -0.0037 | -0.4037 | 0.7255 | False | -0.2331 |
| B_R | C_R | macro_f1 | 3 | 0.0031 | 0.4404 | 0.7027 | False | 0.2543 |
| B_R | D_B | macro_f1 | 3 | -0.0129 | -1.7612 | 0.2203 | False | -1.0168 |
| B_R | D_R | macro_f1 | 3 | 0.0000 | 0.0041 | 0.9971 | False | 0.0024 |
| B_R | TF-IDF + LR | macro_f1 | 3 | 0.0356 | 6.6983 | 0.0216 | True | 3.8673 |
| B_R | TF-IDF + SVM | macro_f1 | 3 | 0.0374 | 7.0343 | 0.0196 | True | 4.0612 |
| C_B | C_R | macro_f1 | 3 | 0.0068 | 0.6571 | 0.5786 | False | 0.3794 |
| C_B | D_B | macro_f1 | 3 | -0.0092 | -3.9103 | 0.0596 | False | -2.2576 |
| C_B | D_R | macro_f1 | 3 | 0.0037 | 0.3997 | 0.7280 | False | 0.2307 |
| C_B | TF-IDF + LR | macro_f1 | 3 | 0.0393 | 9.2126 | 0.0116 | True | 5.3189 |
| C_B | TF-IDF + SVM | macro_f1 | 3 | 0.0411 | 9.6310 | 0.0106 | True | 5.5605 |
| C_R | D_B | macro_f1 | 3 | -0.0160 | -2.0011 | 0.1834 | False | -1.1553 |
| C_R | D_R | macro_f1 | 3 | -0.0030 | -0.3576 | 0.7549 | False | -0.2064 |
| C_R | TF-IDF + LR | macro_f1 | 3 | 0.0325 | 4.9482 | 0.0385 | True | 2.8568 |
| C_R | TF-IDF + SVM | macro_f1 | 3 | 0.0343 | 5.2196 | 0.0348 | True | 3.0136 |
| D_B | D_R | macro_f1 | 3 | 0.0129 | 1.6672 | 0.2374 | False | 0.9626 |
| D_B | TF-IDF + LR | macro_f1 | 3 | 0.0485 | 23.3709 | 0.0018 | True | 13.4932 |
| D_B | TF-IDF + SVM | macro_f1 | 3 | 0.0503 | 24.2311 | 0.0017 | True | 13.9898 |
| D_R | TF-IDF + LR | macro_f1 | 3 | 0.0356 | 6.0258 | 0.0265 | True | 3.4790 |
| D_R | TF-IDF + SVM | macro_f1 | 3 | 0.0374 | 6.3280 | 0.0241 | True | 3.6535 |
| TF-IDF + LR | TF-IDF + SVM | macro_f1 | 3 | 0.0018 | inf | 0.0000 | True | 0.0000 |