# XAI Summary (v2.1 4축 검증 프레임워크)

| baseline_model | improved_model | baseline_macro_f1 | improved_macro_f1 | baseline_overlap_mean | improved_overlap_mean | baseline_overlap_ge_60 | improved_overlap_ge_60 | sample_count | fixed_error_count | baseline_rationale_shap_mean | improved_rationale_shap_mean | baseline_rationale_lime_mean | improved_rationale_lime_mean | baseline_rationale_ge_50 | improved_rationale_ge_50 | rationale_sample_count | baseline_comprehensiveness | improved_comprehensiveness | baseline_sufficiency | improved_sufficiency | masking_sample_count | baseline_slur_free_accuracy | improved_slur_free_accuracy | baseline_slur_prob_drop | improved_slur_prob_drop | slur_free_sample_count | baseline_ci | improved_ci | baseline_mss | improved_mss | baseline_loo | improved_loo | baseline_interaction_strength | improved_interaction_strength | baseline_rollout_entropy | improved_rollout_entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_B | D_B | 0.6765 | 0.6897 | 0.4040 | 0.4160 | 16 | 15 | 50 | 25 | 0.3644 | 0.4402 | 0.5898 | 0.5852 | 14 | 19 | 40 | 0.2986 | 0.3497 | 0.2400 | 0.2102 | 50 | 0.3376 | 0.3368 | 0.3744 | 0.3841 | 1262 | 0.5013 | 0.5234 | 6.3000 | 5.6800 | 0.1206 | 0.1105 | 0.1439 | 0.0382 | 2.5831 | 2.5897 |

## Case Summary

| sample_id | category | baseline_top_tokens | improved_top_tokens | baseline_overlap_at_5 | improved_overlap_at_5 |
| --- | --- | --- | --- | --- | --- |
| 5 | fixed_error | mu, zzi, e, but, recommendations | e, mu, zzi, favour, s | 0.2000 | 0.4000 |
| 7 | fixed_error | i, creamy, negro, am, a | negro, wanna, ass, creamy, ur | 0.4000 | 0.2000 |
| 44 | fixed_error | hits, ho, >, number, made | ho, <, made, hits, that | 0.6000 | 0.4000 |
| 52 | fixed_error | 🤡, a, trusted, ho, the | ho, ho, trust, e, 🤡 | 0.4000 | 0.2000 |
| 63 | fixed_error | dicated, is, islam, europeans, de | de, white, port, islam, is | 0.0000 | 0.6000 |
| 82 | fixed_error | wall, g, will, ho, they | es, yo, ho, y, y | 0.4000 | 0.2000 |
| 100 | fixed_error | peaceful, being, message, mainstream, islam | islam, peaceful, liberals, muslims, lies | 0.2000 | 1.0000 |
| 105 | fixed_error | islam, attack, terror, is, and | stupid, you, las, radical, will | 0.4000 | 0.4000 |

## Human Rationale Alignment (축 1-2: 설명 타당성)

| 지표 | A_B | D_B |
|------|:---:|:---:|
| SHAP Top-5 vs Human Rationale (mean) | 0.3644 | 0.4402 |
| LIME Top-5 vs Human Rationale (mean) | 0.5898 | 0.5852 |
| SHAP overlap >= 0.5 (count) | 14 | 19 |
| Rationale 보유 샘플 수 | 40 | 40 |


## Masking Verification (축 3: 설명 충실도)

| 지표 | A_B | D_B |
|------|:---:|:---:|
| Comprehensiveness (mean) | 0.2986 | 0.3497 |
| Sufficiency (mean) | 0.24 | 0.2102 |
| 마스킹 검증 샘플 수 | 50 | 50 |


## Slur-Free Prediction (맥락 이해 능력)

| 지표 | A_B | D_B |
|------|:---:|:---:|
| Slur-Free Accuracy | 0.3376 | 0.3368 |
| Mean Prob Drop | 0.3744 | 0.3841 |
| 대상 샘플 수 | 1262 | 1262 |


## XAI 4-Axis Automatic Metrics (v2.1)

| 지표 | A_B | D_B |
|------|:---:|:---:|
| CI (Concentration Index) | 0.5013 | 0.5234 |
| MSS | 6.3 | 5.68 |
| LOO mean drop | 0.1206 | 0.1105 |
| Interaction Strength | 0.1439 | 0.0382 |
| Attention Rollout Entropy | 2.5831 | 2.5897 |
