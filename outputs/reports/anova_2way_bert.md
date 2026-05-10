# BERT Family 2-way ANOVA

Macro F1 ~ attention loss × VADER within BERT conditions.

| term | metric | df_num | df_den | sum_sq | f_value | p_value | significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| use_attention_loss | macro_f1 | 1 | 8 | 0.0000 | 0.4566 | 0.5182 | False |
| use_vader | macro_f1 | 1 | 8 | 0.0000 | 0.0026 | 0.9606 | False |
| use_attention_loss * use_vader | macro_f1 | 1 | 8 | 0.0000 | 0.5749 | 0.4701 | False |