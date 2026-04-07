# XAI Summary

| baseline_model | improved_model | baseline_macro_f1 | improved_macro_f1 | baseline_overlap_mean | improved_overlap_mean | baseline_overlap_ge_60 | improved_overlap_ge_60 | sample_count | fixed_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT-base | RoBERTa+VADER | 0.6792 | 0.6822 | 0.6280 | 0.7240 | 36 | 42 | 50 | 25 |

## Case Summary

| sample_id | category | baseline_top_tokens | improved_top_tokens | baseline_overlap_at_5 | improved_overlap_at_5 |
| --- | --- | --- | --- | --- | --- |
| 38 | fixed_error | people, then , e , smart , other  | or , ians , smart , smart , fluoride  | 0.4000 | 0.8000 |
| 44 | fixed_error | hits, number, made , > , e  | that , oe , h, > , number | 1.0000 | 1.0000 |
| 52 | fixed_error | 🤡, ho, myself , never , am  | h, a , oe , trust , me  | 0.6000 | 0.8000 |
| 65 | fixed_error | obama , countries , bombs , and , arranging  | bombs , hates , loves , place , a  | 0.4000 | 0.4000 |
| 70 | fixed_error | 😂 , next, 😂 , 😂 , homosexual  | homosexual , cancer , cancer , is , there  | 0.4000 | 0.6000 |
| 79 | fixed_error | blamed, user, brown , immigrants , be  | immigrants , > , <, user, brown  | 1.0000 | 1.0000 |
| 80 | fixed_error | mine, <, user, do , but  | gay , do , that , <, user | 0.8000 | 1.0000 |
| 82 | fixed_error | wall, yo , pot, a, na  | oes , was, h, hy , they  | 1.0000 | 0.6000 |