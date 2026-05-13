# Common Project Rules for AI Agents

Use this document with Codex, Claude, Gemini, Cursor, Antigravity, or any other coding agent working on HateSpeachStudy v2.

## Mission

Build and validate the `v2_15seed` end-to-end pipeline:

```text
benchmark -> aggregate -> xai-primary -> xai-deep -> xai-ablation -> xai-bundle -> report -> dashboard
```

The goal is not only a performance table. The final claim must combine:

```text
benchmark/statistics: how much the model changes performance
xai-bundle/report/dashboard: what evidence the model leaves behind
```

## Canonical Workspace

```text
Runtime code: v2/runtime/
Pipeline code: v2/pipeline/
Config: v2/configs/v2_15seed.json
Docs: v2/docs/
Canonical outputs: v2/outputs/experiments/v2_15seed/
```

`v1/` is archive/reference only. Do not import it, write new artifacts into it, or treat old outputs as canonical v2 results.

## Read Before Work

```text
v2/README.md
v2/docs/00_reading_order.md
v2/docs/02_e2e_pipeline.md
v2/docs/07_output_and_report_contract.md
v2/docs/14_team_assignment_matrix.md
v2/docs/15_runtime_code_validation_matrix.md
v2/docs/agent_tasks/10_team_dispatch_prompts.md
```

## Cheap Validation First

Run local checks before expensive GPU work:

```bash
python3 -m compileall v2/runtime v2/pipeline v2/scripts/validate_commit_message.py
python3 -m json.tool v2/configs/v2_15seed.json >/tmp/v2_config_check.json
PYTHON_BIN=python3 ./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
PYTHON_BIN=python3 ./v2/run.sh e2e aggregate --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e xai-bundle --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e report --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e dashboard --run-id v2_15seed
git diff --check
```

## Full Run Gate

Do not start the full `8 conditions x 15 seeds = 120` benchmark before all are true:

```text
v2_runtime_import_smoke passed
A_B seed 42 single training smoke passed
A_B/D_B seed 42 paired smoke passed
metrics.json/history.csv/run_config.json/predictions.csv/checkpoint created
aggregate reads smoke metrics and creates paired row
checkpoint_path and predictions_path stay under v2/outputs/experiments/v2_15seed/
report/dashboard stages pass
```

## Output Contract

Important artifacts:

```text
v2/outputs/experiments/v2_15seed/benchmark/benchmark_runs.csv
v2/outputs/experiments/v2_15seed/benchmark/benchmark_summary.csv
v2/outputs/experiments/v2_15seed/benchmark/paired_tests_holm.csv
v2/outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json
v2/outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json
v2/outputs/experiments/v2_15seed/reports/final_report.md
v2/outputs/experiments/v2_15seed/reports/final_report.docx
v2/outputs/experiments/v2_15seed/dashboard/index.html
```

## Interpretation Rules

```text
Use same-seed paired comparisons.
Report mean difference, 95% CI, effect size, and Holm-adjusted p-value together.
Treat smoke runs as execution validation, not statistical evidence.
Treat XAI as post-hoc verification, not causal proof.
Make XAI claims only when they link to source artifacts.
```

## Handoff Format

```text
[v2 agent handoff]
Role:
Files changed:
Commands run:
Artifacts created/updated:
Validation passed:
Known limitations:
Next owner:
```
