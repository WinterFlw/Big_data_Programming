---
name: hatespeech-v2-e2e
description: Portable AI instructions for HateSpeachStudy v2 end-to-end orchestration. Use when an agent must understand or coordinate the v2_15seed pipeline, stage order, v2-only path rules, server run gate, team handoff, or integration across benchmark, aggregate, XAI bundle, report, and dashboard. Works for Codex, Claude, Gemini, Cursor, and other markdown-reading coding agents.
---

# HateSpeachStudy v2 End-to-End Skill

## First Move

Read:

```text
v2/ai_skills/common_project_rules.md
v2/docs/02_e2e_pipeline.md
v2/docs/07_output_and_report_contract.md
v2/docs/14_team_assignment_matrix.md
v2/docs/15_runtime_code_validation_matrix.md
```

## Mission

Coordinate this stage chain:

```text
benchmark -> aggregate -> xai-primary -> xai-deep -> xai-ablation -> xai-bundle -> report -> dashboard
```

Keep the run scoped to:

```text
run_id: v2_15seed
config: v2/configs/v2_15seed.json
output root: v2/outputs/experiments/v2_15seed/
```

## Ownership

Prefer editing:

```text
v2/run.sh
v2/pipeline/cli.py
v2/pipeline/runner.py
v2/pipeline/artifacts.py
v2/pipeline/paths.py
v2/docs/06_execution_runbook.md
v2/docs/11_team_tasking_and_server_run_plan.md
```

Avoid changing runtime model code unless the requested task requires it.

## Required Checks

```bash
python3 -m compileall v2/runtime v2/pipeline v2/scripts/validate_commit_message.py
python3 -m json.tool v2/configs/v2_15seed.json >/tmp/v2_config_check.json
PYTHON_BIN=python3 ./v2/run.sh e2e --help
PYTHON_BIN=python3 ./v2/run.sh e2e status --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
PYTHON_BIN=python3 ./v2/run.sh e2e aggregate --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e xai-bundle --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e report --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e dashboard --run-id v2_15seed
git diff --check
```

## Red Flags

Stop and fix if:

```text
Any stage writes outside v2/outputs/experiments/v2_15seed/.
Any code imports v1 as runtime dependency.
Full benchmark is started before smoke gate.
Report/dashboard reads raw XAI cases before xai_claims.json or xai_dashboard_bundle.json.
Status shows failed units without a retry plan.
```

## Handoff

Use the common `[v2 agent handoff]` block from `common_project_rules.md`.
