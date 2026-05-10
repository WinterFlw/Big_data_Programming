# Final Experiment Bundle

## Benchmark Summary

| model | macro_f1_display | macro_precision_display | macro_recall_display | accuracy_display |
| --- | --- | --- | --- | --- |
| D_B | 0.6885 ± 0.0029 | 0.6892 ± 0.0051 | 0.6908 ± 0.0028 | 0.6949 ± 0.0012 |
| B_B | 0.6826 ± 0.0072 | 0.6837 ± 0.0082 | 0.6839 ± 0.0052 | 0.6911 ± 0.0067 |
| C_B | 0.6793 ± 0.0060 | 0.6789 ± 0.0068 | 0.6820 ± 0.0058 | 0.6862 ± 0.0028 |
| A_B | 0.6791 ± 0.0037 | 0.6779 ± 0.0037 | 0.6820 ± 0.0052 | 0.6885 ± 0.0049 |
| B_R | 0.6756 ± 0.0075 | 0.6757 ± 0.0059 | 0.6786 ± 0.0081 | 0.6860 ± 0.0033 |
| D_R | 0.6756 ± 0.0084 | 0.6744 ± 0.0088 | 0.6789 ± 0.0080 | 0.6835 ± 0.0060 |
| C_R | 0.6726 ± 0.0093 | 0.6730 ± 0.0088 | 0.6749 ± 0.0096 | 0.6806 ± 0.0059 |
| A_R | 0.6694 ± 0.0103 | 0.6685 ± 0.0106 | 0.6729 ± 0.0082 | 0.6787 ± 0.0087 |
| TF-IDF + LR | 0.6400 ± 0.0000 | 0.6403 ± 0.0000 | 0.6402 ± 0.0000 | 0.6489 ± 0.0000 |
| TF-IDF + SVM | 0.6382 ± 0.0000 | 0.6489 ± 0.0000 | 0.6397 ± 0.0000 | 0.6582 ± 0.0000 |

## Freeze Study

| model | macro_f1_display | macro_precision_display | macro_recall_display | accuracy_display |
| --- | --- | --- | --- | --- |
| BERT+VADER (Fine-tuned Encoder) | 0.6805 ± 0.0043 | 0.6801 ± 0.0050 | 0.6828 ± 0.0046 | 0.6869 ± 0.0019 |
| BERT+VADER (Frozen Encoder) | 0.4052 ± 0.0080 | 0.4469 ± 0.0099 | 0.4359 ± 0.0078 | 0.4199 ± 0.0094 |

## XAI Summary

| baseline_model | improved_model | baseline_macro_f1 | improved_macro_f1 | baseline_overlap_mean | improved_overlap_mean | baseline_overlap_ge_60 | improved_overlap_ge_60 | sample_count | fixed_error_count | baseline_rationale_shap_mean | improved_rationale_shap_mean | baseline_rationale_lime_mean | improved_rationale_lime_mean | baseline_rationale_ge_50 | improved_rationale_ge_50 | rationale_sample_count | baseline_comprehensiveness | improved_comprehensiveness | baseline_sufficiency | improved_sufficiency | masking_sample_count | baseline_slur_free_accuracy | improved_slur_free_accuracy | baseline_slur_prob_drop | improved_slur_prob_drop | slur_free_sample_count | baseline_ci | improved_ci | baseline_mss | improved_mss | baseline_loo | improved_loo | baseline_interaction_strength | improved_interaction_strength | baseline_rollout_entropy | improved_rollout_entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_B | D_B | 0.6765 | 0.6897 | 0.2000 | 0.2000 | 0 | 0 | 50 | 25 | 1.0000 | 1.0000 | 0.6411 | 0.6246 | 40 | 40 | 40 | 0.3730 | 0.3815 | 0.0000 | 0.0000 | 50 | 0.3550 | 0.3542 | 0.3682 | 0.3716 | 1262 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.3730 | 0.3815 | None | None | 2.5831 | 2.5897 |