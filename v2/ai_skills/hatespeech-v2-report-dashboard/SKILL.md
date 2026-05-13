---
name: hatespeech-v2-report-dashboard
description: Portable AI instructions for HateSpeachStudy v2 final report and dashboard work. Use when an agent must create or validate final_report.md, final_report.docx, dashboard/index.html, report sections, dashboard rendering, evidence bundle consumption, output links, or final project deliverables. Works for Codex, Claude, Gemini, Cursor, and other markdown-reading coding agents.
---

# HateSpeachStudy v2 Report/Dashboard Skill

## First Move

Read:

```text
v2/ai_skills/common_project_rules.md
v2/docs/07_output_and_report_contract.md
v2/docs/08_xai_report_template.md
v2/docs/11_team_tasking_and_server_run_plan.md
```

## Mission

Generate final presentation artifacts from benchmark/statistics/XAI evidence:

```text
v2/outputs/experiments/v2_15seed/reports/final_report.md
v2/outputs/experiments/v2_15seed/reports/final_report.docx
v2/outputs/experiments/v2_15seed/dashboard/index.html
```

## Ownership

Primary files:

```text
v2/pipeline/reporting.py
v2/pipeline/runner.py
v2/pipeline/cli.py
v2/runtime/experiment_dashboard.py
v2/runtime/dashboard_app.py
v2/docs/08_xai_report_template.md
```

Avoid changing benchmark/statistics/XAI code unless output schema is impossible to consume.

## Report Rules

```text
Use xai_claims.json and xai_dashboard_bundle.json before raw XAI case files.
Keep placeholders conservative when final results do not exist.
Do not write statistically significant language unless paired_tests_holm supports it.
Include limitations and threat-to-validity language.
Make paths relative to v2 canonical output root where practical.
```

## Commands

```bash
PYTHON_BIN=python3 ./v2/run.sh e2e report --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e dashboard --run-id v2_15seed
unzip -t v2/outputs/experiments/v2_15seed/reports/final_report.docx
```

## Red Flags

```text
Report reads old top-level outputs as final truth.
Dashboard ignores xai/evidence_bundle.
DOCX is missing when final_report.md exists.
Dashboard hardcodes old 3-seed language for v2_15seed.
Report turns qualitative XAI cases into broad statistical claims.
```

Use the common `[v2 agent handoff]` block from `common_project_rules.md`.
