# v2 Final Report

## Run

- run_id: `v2_15seed`
- manifest_hash: `c75bc0254e3ca26c`
- output_root: `outputs/experiments/v2_15seed`

## Benchmark Status

- total units: 120
- completed: 120
- failed: 0
- planned: 0

## Model

The v2 model family evaluates rationale-aware attention loss and VADER sentiment features through an 8-condition ablation matrix across BERT and RoBERTa backbones.

## Benchmark Summary

| Condition | Backbone | N seeds | Macro F1 (mean ± std) | 95% CI |
|---|---|---|---|---|
| A_B | BERT | 15 | 0.6798 ± 0.0076 | [0.6762, 0.6832] |
| B_B | BERT | 15 | 0.6858 ± 0.0060 | [0.6829, 0.6884] |
| C_B | BERT | 15 | 0.6825 ± 0.0088 | [0.6778, 0.6863] |
| D_B | BERT | 15 | 0.6836 ± 0.0041 | [0.6816, 0.6856] |
| A_R | RoBERTa | 15 | 0.6653 ± 0.0072 | [0.6617, 0.6685] |
| B_R | RoBERTa | 15 | 0.6763 ± 0.0040 | [0.6743, 0.6782] |
| C_R | RoBERTa | 15 | 0.6698 ± 0.0065 | [0.6666, 0.6730] |
| D_R | RoBERTa | 15 | 0.6743 ± 0.0069 | [0.6707, 0.6775] |

## Primary Paired Tests (macro_f1)

The headline analysis uses same-seed paired tests, mean difference, confidence intervals, and effect size. Adjusted p-values are shown as a supplementary guardrail when multiple comparisons are displayed.

| Comparison | n pairs | Mean diff | paired p | adjusted p | effect size | sig@0.05 |
|---|---|---|---|---|---|---|
| A_B vs D_B | 15 | 0.0039 | 0.0485 | 0.5818 | 0.5580 | False |
| A_B vs B_B | 15 | 0.0060 | 0.0001 | 0.0027 | 1.3506 | True |
| A_B vs C_B | 15 | 0.0027 | 0.3721 | 1.0000 | 0.2381 | False |
| B_B vs D_B | 15 | -0.0022 | 0.2136 | 1.0000 | -0.3365 | False |
| C_B vs D_B | 15 | 0.0011 | 0.6141 | 1.0000 | 0.1332 | False |
| A_R vs D_R | 15 | 0.0090 | 0.0026 | 0.0417 | 0.9434 | True |
| D_B vs D_R | 15 | -0.0093 | 0.0005 | 0.0090 | -1.1689 | True |

## Supplementary ANOVA — BERT family (2-way)

Factors: attention loss × VADER.

| family | metric | factor | sum_sq | df | F | p_value | eta_squared | partial_eta_squared |
|---|---|---|---|---|---|---|---|---|
| BERT | macro_f1 | C(attention_loss) | 0.0002 | 1.0000 | 4.1283 | 0.0469 | 0.0665 | 0.0687 |
| BERT | macro_f1 | C(vader) | 0.0000 | 1.0000 | 0.0272 | 0.8696 | 0.0004 | 0.0005 |
| BERT | macro_f1 | C(attention_loss):C(vader) | 0.0001 | 1.0000 | 1.9120 | 0.1722 | 0.0308 | 0.0330 |
| BERT | macro_f1 | Residual | 0.0026 | 56.0000 | - | - | - | - |

## Supplementary ANOVA — RoBERTa family (2-way)

| family | metric | factor | sum_sq | df | F | p_value | eta_squared | partial_eta_squared |
|---|---|---|---|---|---|---|---|---|
| RoBERTa | macro_f1 | C(attention_loss) | 0.0009 | 1.0000 | 23.1094 | 0.0000 | 0.2759 | 0.2921 |
| RoBERTa | macro_f1 | C(vader) | 0.0000 | 1.0000 | 0.5847 | 0.4477 | 0.0070 | 0.0103 |
| RoBERTa | macro_f1 | C(attention_loss):C(vader) | 0.0002 | 1.0000 | 4.0556 | 0.0488 | 0.0484 | 0.0675 |
| RoBERTa | macro_f1 | Residual | 0.0022 | 56.0000 | - | - | - | - |

## Supplementary ANOVA — Cross-family (3-way)

Factors: backbone × attention loss × VADER.

| metric | factor | sum_sq | df | F | p_value | eta_squared | partial_eta_squared |
|---|---|---|---|---|---|---|---|
| macro_f1 | C(backbone) | 0.0039 | 1.0000 | 91.8449 | 0.0000 | 0.3896 | 0.4506 |
| macro_f1 | C(attention_loss) | 0.0010 | 1.0000 | 22.5118 | 0.0000 | 0.0955 | 0.1674 |
| macro_f1 | C(vader) | 0.0000 | 1.0000 | 0.4070 | 0.5248 | 0.0017 | 0.0036 |
| macro_f1 | C(backbone):C(attention_loss) | 0.0001 | 1.0000 | 3.0530 | 0.0833 | 0.0129 | 0.0265 |
| macro_f1 | C(backbone):C(vader) | 0.0000 | 1.0000 | 0.1558 | 0.6938 | 0.0007 | 0.0014 |
| macro_f1 | C(attention_loss):C(vader) | 0.0002 | 1.0000 | 5.6631 | 0.0190 | 0.0240 | 0.0481 |
| macro_f1 | C(backbone):C(attention_loss):C(vader) | 0.0000 | 1.0000 | 0.1155 | 0.7346 | 0.0005 | 0.0010 |
| macro_f1 | Residual | 0.0048 | 112.0000 | - | - | - | - |

## XAI Evidence Summary

XAI is treated as post-hoc verification. Primary XAI compares A_B and D_B across all seeds, while deep XAI uses median-performing checkpoints for detailed cases.

### Summary Cards

- **Top benchmark condition**: B_B (0.685780115774346 ± 0.005993749356719612) (source: `benchmark/benchmark_summary.csv`)
- **Rationale F1@5 (mean across seeds)**: 0.0857 (source: `xai/primary/seed_level_metrics.csv`)
- **Most stable explanation (top-k Jaccard)**: D_B mean=0.5797883597883597 (source: `xai/primary/seed_stability.csv`)

### Statistically-supported Claims

| ID | Strength | Statement | Source |
|---|---|---|---|
| claim_001 | strong | A_B vs B_B shows higher macro F1 (mean diff +0.0060, supplementary adjusted p=0.0027). | `benchmark/paired_tests_holm.csv` |
| claim_002 | moderate | A_R vs D_R shows higher macro F1 (mean diff +0.0090, supplementary adjusted p=0.0417). | `benchmark/paired_tests_holm.csv` |
| claim_003 | strong | D_B vs D_R shows lower macro F1 (mean diff -0.0093, supplementary adjusted p=0.0090). | `benchmark/paired_tests_holm.csv` |

## Seed Stability

Top-k Jaccard and rank correlation across seed checkpoints — high stability means explanations stay consistent under seed variation.

| Condition | Metric | Mean | Std | CI Low | CI High |
|---|---|---|---|---|---|
| A_B | topk_jaccard_5 | 0.4093 | 0.2713 | 0.3122 | 0.5063 |
| A_B | rank_corr | 0.5952 | 0.5884 | 0.3645 | 0.8258 |
| D_B | topk_jaccard_5 | 0.5798 | 0.2617 | 0.4862 | 0.6734 |
| D_B | rank_corr | 0.5992 | 0.6316 | 0.3652 | 0.8332 |

## Limitations

- [warning] low_pair_count: n_pairs=2 for metric shap_lime_overlap_at_5 (recommendation: Treat XAI paired test as exploratory.)
- [warning] low_pair_count: n_pairs=2 for metric rationale_f1_at_5 (recommendation: Treat XAI paired test as exploratory.)
- [warning] low_pair_count: n_pairs=2 for metric comprehensiveness (recommendation: Treat XAI paired test as exploratory.)
- [warning] low_pair_count: n_pairs=2 for metric sufficiency (recommendation: Treat XAI paired test as exploratory.)
- [warning] low_pair_count: n_pairs=2 for metric loo_drop (recommendation: Treat XAI paired test as exploratory.)

## XAI Evidence Bundle (file links)

Report and dashboard generation should prefer the evidence bundle before reading raw XAI case files.

- `xai/evidence_bundle/xai_claims.json`: present
- `xai/evidence_bundle/xai_dashboard_bundle.json`: present
- `xai/evidence_bundle/xai_interpretation_cards.json`: present
- `xai/evidence_bundle/xai_run_metadata.json`: present
- `xai/evidence_bundle/token_attributions.jsonl`: present

## Reproducibility

- Manifest hash: `c75bc0254e3ca26c`
- Conditions: A_B, B_B, C_B, D_B, A_R, B_R, C_R, D_R
- Seeds: 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182
- Commands to reproduce (from `v2/`):
  - `./run.sh e2e plan --run-id v2_15seed`
  - `./run.sh e2e benchmark --run-id v2_15seed --execute`
  - `./run.sh e2e aggregate --run-id v2_15seed`
  - `./run.sh e2e xai-primary --run-id v2_15seed`
  - `./run.sh e2e xai-bundle --run-id v2_15seed`
  - `./run.sh e2e report --run-id v2_15seed`
