# HateSpeachStudy v2 Agent Context

Use this file as the project-level instruction for Claude Code or any Claude-based coding agent.

## Project Rule

This repository's active workspace is `v2/`. Treat `v1/` as archive/reference only.

Do not import, write, or treat `v1/outputs`, `v1/checkpoints`, root `outputs`, or root `checkpoints` as canonical v2 artifacts.

## Canonical Paths

```text
Runtime code: v2/runtime/
Pipeline code: v2/pipeline/
Config: v2/configs/v2_15seed.json
Docs: v2/docs/
Portable AI skills: v2/ai_skills/
Canonical outputs: v2/outputs/experiments/v2_15seed/
Main CLI: ./v2/run.sh e2e ...
```

## Required First Reads

Read these before changing code:

```text
v2/ai_skills/common_project_rules.md
v2/docs/14_team_assignment_matrix.md
v2/docs/15_runtime_code_validation_matrix.md
```

Then read the role skill that matches the task:

```text
v2/ai_skills/hatespeech-v2-e2e/SKILL.md
v2/ai_skills/hatespeech-v2-benchmark/SKILL.md
v2/ai_skills/hatespeech-v2-statistics/SKILL.md
v2/ai_skills/hatespeech-v2-xai/SKILL.md
v2/ai_skills/hatespeech-v2-report-dashboard/SKILL.md
v2/ai_skills/hatespeech-v2-review/SKILL.md
```

## Safety Gate

Do not launch the full `8 conditions x 15 seeds = 120` benchmark unless the smoke gate is already satisfied:

```text
v2_runtime_import_smoke passed
A_B seed 42 single training smoke passed
A_B/D_B seed 42 paired smoke passed
metrics/history/config/predictions/checkpoint created under v2 output root
aggregate/report/dashboard stages pass
```

## Completion Format

End substantial work with:

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
