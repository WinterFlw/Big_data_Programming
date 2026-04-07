# Final Experiment Bundle

## Benchmark Summary

| model | macro_f1_display | macro_precision_display | macro_recall_display | accuracy_display |
| --- | --- | --- | --- | --- |
| RoBERTa+VADER | 0.6834 ± 0.0089 | 0.6846 ± 0.0072 | 0.6899 ± 0.0067 | 0.6892 ± 0.0095 |
| BERT+MLP | 0.6824 ± 0.0058 | 0.6812 ± 0.0058 | 0.6841 ± 0.0058 | 0.6914 ± 0.0060 |
| BERT+VADER | 0.6803 ± 0.0024 | 0.6806 ± 0.0033 | 0.6813 ± 0.0027 | 0.6876 ± 0.0006 |
| BERT-base | 0.6738 ± 0.0073 | 0.6731 ± 0.0072 | 0.6751 ± 0.0069 | 0.6834 ± 0.0050 |
| TF-IDF + LR | 0.6400 ± 0.0000 | 0.6403 ± 0.0000 | 0.6402 ± 0.0000 | 0.6489 ± 0.0000 |
| TF-IDF + SVM | 0.6382 ± 0.0000 | 0.6489 ± 0.0000 | 0.6397 ± 0.0000 | 0.6582 ± 0.0000 |

## Freeze Study

| model | macro_f1_display | macro_precision_display | macro_recall_display | accuracy_display |
| --- | --- | --- | --- | --- |
| BERT+VADER (Fine-tuned Encoder) | 0.6807 ± 0.0041 | 0.6808 ± 0.0039 | 0.6831 ± 0.0043 | 0.6883 ± 0.0016 |
| BERT+VADER (Frozen Encoder) | 0.4052 ± 0.0080 | 0.4473 ± 0.0096 | 0.4361 ± 0.0077 | 0.4200 ± 0.0093 |

## XAI Summary

| baseline_model | improved_model | baseline_macro_f1 | improved_macro_f1 | baseline_overlap_mean | improved_overlap_mean | baseline_overlap_ge_60 | improved_overlap_ge_60 | sample_count | fixed_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT-base | RoBERTa+VADER | 0.6792 | 0.6822 | 0.6280 | 0.7240 | 36 | 42 | 50 | 25 |