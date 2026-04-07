# Encoder Freeze Study

Tuned hyperparameters reused from `BERT+VADER` benchmark settings.

| model | macro_f1_display | macro_precision_display | macro_recall_display | accuracy_display |
| --- | --- | --- | --- | --- |
| BERT+VADER (Fine-tuned Encoder) | 0.6807 ± 0.0041 | 0.6808 ± 0.0039 | 0.6831 ± 0.0043 | 0.6883 ± 0.0016 |
| BERT+VADER (Frozen Encoder) | 0.4052 ± 0.0080 | 0.4473 ± 0.0096 | 0.4361 ± 0.0077 | 0.4200 ± 0.0093 |