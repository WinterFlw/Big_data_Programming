# HateSpeachStudy v2 Agent Context

Use this file as the project-level instruction for Gemini CLI, Gemini Code Assist, or any Gemini-based coding agent.

## Active Workspace

Work from `v2/` as the single source of truth. `v1/` is archive/reference only.

Never use these as canonical v2 inputs or outputs:

```text
v1/outputs
v1/checkpoints
outputs/reports
outputs/xai
outputs/runs
checkpoints
```

## Core Map

```text
Runtime code: v2/runtime/
Pipeline orchestration: v2/pipeline/
Run config: v2/configs/v2_15seed.json
Canonical output root: v2/outputs/experiments/v2_15seed/
Portable agent instructions: v2/ai_skills/
Main command: ./v2/run.sh e2e ...
```

## How To Start

1. Read `v2/ai_skills/common_project_rules.md`.
2. Read `v2/docs/14_team_assignment_matrix.md`.
3. Read `v2/docs/15_runtime_code_validation_matrix.md`.
4. Select exactly one role skill from `v2/ai_skills/*/SKILL.md`.
5. Keep edits inside the role's ownership unless a cross-file change is necessary and explained.

## Required Validation

Prefer cheap local checks before expensive GPU work:

```bash
python3 -m compileall v2/runtime v2/pipeline v2/scripts/validate_commit_message.py
python3 -m json.tool v2/configs/v2_15seed.json >/tmp/v2_config_check.json
PYTHON_BIN=python3 ./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
PYTHON_BIN=python3 ./v2/run.sh e2e aggregate --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e xai-bundle --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e report --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e dashboard --run-id v2_15seed
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
