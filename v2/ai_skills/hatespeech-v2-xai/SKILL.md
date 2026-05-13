---
name: hatespeech-v2-xai
description: Portable AI instructions for HateSpeachStudy v2 XAI and evidence bundle work. Use when an agent must implement or validate xai-primary, xai-deep, xai-ablation, xai-bundle, sample selection, seed stability, SHAP/LIME integration, faithfulness/context/plausibility metrics, xai_claims.json, xai_dashboard_bundle.json, or XAI report evidence contracts. Works for Codex, Claude, Gemini, Cursor, and other markdown-reading coding agents.
---

# HateSpeachStudy v2 XAI Skill

## First Move

Read:

```text
v2/ai_skills/common_project_rules.md
v2/docs/04_xai_protocol.md
v2/docs/07_output_and_report_contract.md
v2/docs/08_xai_report_template.md
v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md
```

## Mission

Build post-hoc explanation evidence:

```text
xai-primary: A_B vs D_B across all benchmark seeds
xai-deep: detailed median-seed case analysis
xai-ablation: lightweight 8-condition comparison
xai-bundle: deterministic combiner for report/dashboard
```

## Ownership

Primary files:

```text
v2/pipeline/xai.py
v2/pipeline/xai_bundle.py
v2/runtime/experiment_xai.py
v2/docs/04_xai_protocol.md
v2/docs/07_output_and_report_contract.md
```

Avoid changing benchmark/statistics/report unless an artifact contract is wrong.

## Interpretation Rules

```text
XAI is post-hoc verification, not causal proof.
Use the same sample IDs across seeds for seed-stability comparisons.
Do not recalculate SHAP/LIME inside xai-bundle; xai-bundle combines artifacts.
Every claim must link to source artifacts.
Unverified claims must be marked planned/insufficient_evidence.
```

## Required Bundle Outputs

```text
xai/evidence_bundle/evidence_inventory.csv
xai/evidence_bundle/xai_run_metadata.json
xai/evidence_bundle/xai_sample_manifest.csv
xai/evidence_bundle/xai_predictions.csv
xai/evidence_bundle/method_agreement.csv
xai/evidence_bundle/faithfulness_metrics.csv
xai/evidence_bundle/context_metrics.csv
xai/evidence_bundle/plausibility_metrics.csv
xai/evidence_bundle/subgroup_xai_metrics.csv
xai/evidence_bundle/xai_risk_flags.csv
xai/evidence_bundle/xai_claims.json
xai/evidence_bundle/xai_interpretation_cards.json
xai/evidence_bundle/xai_dashboard_bundle.json
xai/evidence_bundle/token_attributions.jsonl
```

## Smoke Commands

```bash
PYTHON_BIN=python3 ./v2/run.sh e2e xai-primary --run-id v2_15seed --dry-run
PYTHON_BIN=python3 ./v2/run.sh e2e xai-bundle --run-id v2_15seed
python3 -m json.tool v2/outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json >/tmp/xai_claims_check.json
python3 -m json.tool v2/outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json >/tmp/xai_dashboard_check.json
```

## Red Flags

```text
XAI output is written outside v2 output root.
Sample set changes by seed.
Report uses cherry-picked cases as statistical proof.
xai-bundle imports heavy SHAP/LIME code unnecessarily.
Dashboard reads raw case files before evidence bundle.
```

Use the common `[v2 agent handoff]` block from `common_project_rules.md`.
