# Statistical Significance Tests

Paired t-test on macro F1 across seeds (alpha=0.05).
⚠️ 3-seed 반복은 검정력이 낮으므로 해석에 주의가 필요합니다.

| model_a | model_b | metric | n_seeds | mean_diff | t_statistic | p_value | significant | cohens_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT+MLP | BERT+VADER | macro_f1 | 3 | 0.0016 | 0.6801 | 0.5666 | False | 0.3927 |
| BERT+MLP | BERT-base | macro_f1 | 3 | 0.0065 | 2.5750 | 0.1235 | False | 1.4867 |
| BERT+MLP | RoBERTa+VADER | macro_f1 | 3 | -0.0054 | -8.2034 | 0.0145 | True | -4.7363 |
| BERT+MLP | TF-IDF + LR | macro_f1 | 3 | 0.0440 | 30.2029 | 0.0011 | True | 17.4376 |
| BERT+MLP | TF-IDF + SVM | macro_f1 | 3 | 0.0417 | 28.5990 | 0.0012 | True | 16.5116 |
| BERT+VADER | BERT-base | macro_f1 | 3 | 0.0049 | 2.4004 | 0.1384 | False | 1.3859 |
| BERT+VADER | RoBERTa+VADER | macro_f1 | 3 | -0.0070 | -3.9875 | 0.0575 | False | -2.3022 |
| BERT+VADER | TF-IDF + LR | macro_f1 | 3 | 0.0424 | 17.6055 | 0.0032 | True | 10.1645 |
| BERT+VADER | TF-IDF + SVM | macro_f1 | 3 | 0.0401 | 16.6351 | 0.0036 | True | 9.6043 |
| BERT-base | RoBERTa+VADER | macro_f1 | 3 | -0.0119 | -5.0792 | 0.0366 | True | -2.9325 |
| BERT-base | TF-IDF + LR | macro_f1 | 3 | 0.0375 | 27.6485 | 0.0013 | True | 15.9629 |
| BERT-base | TF-IDF + SVM | macro_f1 | 3 | 0.0351 | 25.9242 | 0.0015 | True | 14.9674 |
| RoBERTa+VADER | TF-IDF + LR | macro_f1 | 3 | 0.0494 | 30.9800 | 0.0010 | True | 17.8863 |
| RoBERTa+VADER | TF-IDF + SVM | macro_f1 | 3 | 0.0470 | 29.5132 | 0.0011 | True | 17.0394 |
| TF-IDF + LR | TF-IDF + SVM | macro_f1 | 3 | -0.0023 | -inf | 0.0000 | True | 0.0000 |