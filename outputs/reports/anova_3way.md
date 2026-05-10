# 3-way ANOVA

Macro F1 ~ family × attention loss × VADER.

| term | metric | df_num | df_den | sum_sq | f_value | p_value | significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| family | macro_f1 | 1 | 16 | 0.0001 | 1.7420 | 0.2055 | False |
| use_attention_loss | macro_f1 | 1 | 16 | 0.0000 | 0.2358 | 0.6338 | False |
| use_vader | macro_f1 | 1 | 16 | 0.0000 | 0.0013 | 0.9712 | False |
| family * use_attention_loss | macro_f1 | 1 | 16 | 0.0000 | 0.0662 | 0.8002 | False |
| family * use_vader | macro_f1 | 1 | 16 | 0.0000 | 0.0783 | 0.7832 | False |
| use_attention_loss * use_vader | macro_f1 | 1 | 16 | 0.0000 | 0.2969 | 0.5933 | False |
| family * use_attention_loss * use_vader | macro_f1 | 1 | 16 | 0.0000 | 0.3623 | 0.5556 | False |