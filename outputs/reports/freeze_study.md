# Encoder Freeze Study

Tuned hyperparameters reused from `BERT+VADER` benchmark settings.

| model | macro_f1_display | macro_precision_display | macro_recall_display | accuracy_display |
| --- | --- | --- | --- | --- |
| BERT+VADER (Fine-tuned Encoder) | 0.6805 ± 0.0043 | 0.6801 ± 0.0050 | 0.6828 ± 0.0046 | 0.6869 ± 0.0019 |
| BERT+VADER (Frozen Encoder) | 0.4052 ± 0.0080 | 0.4469 ± 0.0099 | 0.4359 ± 0.0078 | 0.4199 ± 0.0094 |