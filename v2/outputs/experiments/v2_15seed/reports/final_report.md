# v2 Final Report

## Run

- run_id: `v2_15seed`
- manifest_hash: `c75bc0254e3ca26c`
- output_root: `outputs/experiments/v2_15seed`

## Benchmark Status

- total units: 120
- completed: 0
- failed: 0
- planned: 120

## Model

The v2 model family evaluates rationale-aware attention loss and VADER sentiment features through an 8-condition ablation matrix across BERT and RoBERTa backbones.

## Benchmark Summary

_no benchmark results yet — populate by running ./run.sh e2e benchmark --execute_

## Primary Paired Tests (macro_f1)

The headline analysis uses same-seed paired tests, mean difference, confidence intervals, and effect size. Adjusted p-values are shown as a supplementary guardrail when multiple comparisons are displayed.

_no paired tests yet for metric `macro_f1` — populate by running ./run.sh e2e aggregate_

## Supplementary ANOVA — BERT family (2-way)

Factors: attention loss × VADER.

_no ANOVA rows yet for `anova_2way_bert` — populate by running ./run.sh e2e aggregate after seeds complete_

## Supplementary ANOVA — RoBERTa family (2-way)

_no ANOVA rows yet for `anova_2way_roberta` — populate by running ./run.sh e2e aggregate after seeds complete_

## Supplementary ANOVA — Cross-family (3-way)

Factors: backbone × attention loss × VADER.

_no ANOVA rows yet for `anova_3way` — populate by running ./run.sh e2e aggregate after seeds complete_

## XAI Evidence Summary

XAI is treated as post-hoc verification. Primary XAI compares A_B and D_B across all seeds, while deep XAI uses median-performing checkpoints for detailed cases.

### Summary Cards

- **No XAI evidence yet**: Run xai-primary / xai-deep / xai-ablation, then re-run xai-bundle. (source: `<bundle placeholder>`)


## Seed Stability

Top-k Jaccard and rank correlation across seed checkpoints — high stability means explanations stay consistent under seed variation.

_no seed stability rows yet — populate by running ./run.sh e2e xai-primary after multiple seeds complete_

## Limitations

- [info] no_seed_metrics: xai/primary/seed_level_metrics.csv has no rows. (recommendation: Mark XAI section as not yet executed.)

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
