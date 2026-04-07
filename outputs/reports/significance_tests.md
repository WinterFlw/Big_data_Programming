# Statistical Significance Tests

Paired t-test on macro F1 across seeds (alpha=0.05).
⚠️ 3-seed 반복은 검정력이 낮으므로 해석에 주의가 필요합니다.

| model_a | model_b | metric | n_seeds | mean_diff | t_statistic | p_value | significant | cohens_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT+MLP | BERT+VADER | macro_f1 | 3 | 0.0022 | 0.4857 | 0.6752 | False | 0.2804 |
| BERT+MLP | BERT-base | macro_f1 | 3 | 0.0086 | 1.8764 | 0.2014 | False | 1.0833 |
| BERT+MLP | RoBERTa+VADER | macro_f1 | 3 | -0.0010 | -0.1213 | 0.9145 | False | -0.0700 |
| BERT+MLP | TF-IDF + LR | macro_f1 | 3 | 0.0424 | 10.3965 | 0.0091 | True | 6.0024 |
| BERT+MLP | TF-IDF + SVM | macro_f1 | 3 | 0.0442 | 10.8341 | 0.0084 | True | 6.2551 |
| BERT+VADER | BERT-base | macro_f1 | 3 | 0.0064 | 0.9529 | 0.4412 | False | 0.5502 |
| BERT+VADER | RoBERTa+VADER | macro_f1 | 3 | -0.0031 | -0.6899 | 0.5615 | False | -0.3983 |
| BERT+VADER | TF-IDF + LR | macro_f1 | 3 | 0.0402 | 23.3140 | 0.0018 | True | 13.4603 |
| BERT+VADER | TF-IDF + SVM | macro_f1 | 3 | 0.0420 | 24.3485 | 0.0017 | True | 14.0576 |
| BERT-base | RoBERTa+VADER | macro_f1 | 3 | -0.0096 | -0.8559 | 0.4822 | False | -0.4941 |
| BERT-base | TF-IDF + LR | macro_f1 | 3 | 0.0338 | 6.5121 | 0.0228 | True | 3.7597 |
| BERT-base | TF-IDF + SVM | macro_f1 | 3 | 0.0356 | 6.8559 | 0.0206 | True | 3.9583 |
| RoBERTa+VADER | TF-IDF + LR | macro_f1 | 3 | 0.0434 | 6.9261 | 0.0202 | True | 3.9988 |
| RoBERTa+VADER | TF-IDF + SVM | macro_f1 | 3 | 0.0452 | 7.2112 | 0.0187 | True | 4.1634 |
| TF-IDF + LR | TF-IDF + SVM | macro_f1 | 3 | 0.0018 | inf | 0.0000 | True | 0.0000 |