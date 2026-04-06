# XAI Summary

| baseline_model | improved_model | baseline_macro_f1 | improved_macro_f1 | baseline_overlap_mean | improved_overlap_mean | baseline_overlap_ge_60 | improved_overlap_ge_60 | sample_count | fixed_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT-base | RoBERTa+VADER | 0.6770 | 0.6846 | 0.7167 | 0.6750 | 18 | 20 | 24 | 12 |

## Case Summary

| sample_id | category | baseline_top_tokens | improved_top_tokens | baseline_overlap_at_5 | improved_overlap_at_5 |
| --- | --- | --- | --- | --- | --- |
| 27 | fixed_error | facebook , hillary , everything , law , leftist | leftists , immigrants , illegal , swamp , the  | 0.2000 | 0.6000 |
| 39 | fixed_error | gg, fa, you , user, < | bitch, would , azi , ute , even  | 1.0000 | 0.4000 |
| 45 | fixed_error | trash , di, african , pe, it | ghetto , it, trash , rican , af | 0.8000 | 0.6000 |
| 57 | fixed_error | ghetto , boot, w, hine, s  | ghetto , dark , constantly , the , friend  | 1.0000 | 0.8000 |
| 71 | fixed_error | negro , wouldn , ain , dad , us | the , negro , claim , i , look  | 0.6000 | 0.6000 |
| 74 | fixed_error | bitch, wit , pretty , es, fat | ches, bit, wit , uated , at | 1.0000 | 1.0000 |
| 76 | fixed_error | immigrants , post, jews , are , whites  | immigrants , ists , that , whites , alt  | 0.6000 | 0.6000 |
| 79 | fixed_error | immigrants , user, > , blamed, be  | immigrants , > , user, will , < | 1.0000 | 1.0000 |