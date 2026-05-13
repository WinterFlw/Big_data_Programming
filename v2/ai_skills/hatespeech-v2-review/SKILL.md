---
name: hatespeech-v2-review
description: Portable AI instructions for reviewing HateSpeachStudy v2 code, docs, and server readiness. Use when an agent is asked to audit the pipeline, find bugs, verify v2-only execution, check role boundaries, review output contracts, validate statistical/XAI claims, or decide whether the project is ready for a limited GPU/server run. Works for Codex, Claude, Gemini, Cursor, and other markdown-reading coding agents.
---

# HateSpeachStudy v2 Review Skill

## First Move

Read:

```text
v2/ai_skills/common_project_rules.md
v2/docs/14_team_assignment_matrix.md
v2/docs/15_runtime_code_validation_matrix.md
v2/docs/07_output_and_report_contract.md
```

## Review Stance

Prioritize bugs and run-blocking risks over style. Lead with findings.

Check:

```text
v2-only imports and paths
run_id output isolation
resume/force behavior
metrics and predictions contract
paired statistics correctness
XAI evidence bundle contract
report/dashboard input priority
server smoke gate readiness
```

## Commands

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

## Path Audit

Search for stale wording or unsafe dependencies:

```bash
rg -n "v1 fallback|v1 checkpoint|legacy train|refuses --execute|3-seed|6 models x 3|top-level outputs|outputs/reports|outputs/xai" v2 README.md
rg -n "sys.path|BASE_DIR|OUTPUT_DIR|CHECKPOINT_DIR|v1" v2/pipeline v2/runtime
```

Not every match is a bug. Archive warnings are acceptable; runtime imports or canonical-output use are not.

## Review Output

Use:

```text
Findings
- [severity] file:line - issue, impact, suggested fix

Open Questions
- ...

Validation
- commands run
- commands not run and why
```

If no issues are found, say so clearly and name remaining residual risk.
