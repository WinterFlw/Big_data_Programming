# Encoder Freeze Study

Tuned hyperparameters reused from `BERT+VADER` benchmark settings.

| model | macro_f1_display | macro_precision_display | macro_recall_display | accuracy_display |
| --- | --- | --- | --- | --- |
| BERT+VADER (Fine-tuned Encoder) | 0.6790 ± 0.0075 | 0.6781 ± 0.0077 | 0.6815 ± 0.0063 | 0.6908 ± 0.0060 |
| BERT+VADER (Frozen Encoder) | 0.3243 ± 0.0099 | 0.5855 ± 0.0553 | 0.4024 ± 0.0068 | 0.4645 ± 0.0057 |