---
name: hatespeech-v2-statistics
description: Portable AI instructions for HateSpeachStudy v2 aggregate and statistical validation work. Use when an agent must work on benchmark_runs.csv, benchmark_summary.csv, paired_tests.csv, paired_tests_holm.csv, same-seed paired tests, Holm correction, confidence intervals, Cohen dz, p-values, partial-result handling, or inference output contracts. Works for Codex, Claude, Gemini, Cursor, and other markdown-reading coding agents.
---

# HateSpeachStudy v2 Statistics Skill

## First Move

Read:

```text
v2/ai_skills/common_project_rules.md
v2/docs/03_validation_and_statistics.md
v2/docs/07_output_and_report_contract.md
v2/docs/15_runtime_code_validation_matrix.md
```

## Mission

Turn completed benchmark run artifacts into statistical evidence:

```text
benchmark_runs.csv
benchmark_summary.csv
paired_tests.csv
paired_tests_holm.csv
```

## Ownership

Primary files:

```text
v2/pipeline/statistics.py
v2/pipeline/schema.py
v2/docs/03_validation_and_statistics.md
v2/docs/07_output_and_report_contract.md
```

Do not change benchmark execution or XAI execution code unless metrics schema is broken.

## Statistical Rules

```text
Compare conditions within the same seed.
Do not use independent-sample tests for same-seed condition comparisons.
Report mean_diff, 95% CI, effect_size, p_value, and p_value_holm together.
Keep partial/empty outputs schema-correct so report/dashboard stages still run.
Do not claim significance from smoke runs.
```

## Smoke Commands

```bash
PYTHON_BIN=python3 ./v2/run.sh e2e aggregate --run-id v2_15seed
PYTHONPATH=v2 python3 - <<'PY'
from pipeline.statistics import compute_paired_tests, apply_holm_correction
manifest = {'statistics': {'paired_tests': ['A_B:D_B']}}
rows = []
for seed, a, d in [(42, .60, .64), (52, .61, .67), (62, .62, .68), (72, .64, .69)]:
    rows.append({'condition':'A_B','seed':seed,'status':'completed','macro_f1':a,'accuracy':a,'weighted_f1':a})
    rows.append({'condition':'D_B','seed':seed,'status':'completed','macro_f1':d,'accuracy':d,'weighted_f1':d})
paired = compute_paired_tests(manifest, rows)
corrected = apply_holm_correction(paired)
assert paired[0]['n_pairs'] == 4
assert corrected[0]['p_value_holm'] != ''
print('paired_statistics_smoke: ok')
PY
```

## Red Flags

```text
p-value is reported without effect size or CI.
Rows compare different seeds.
Failed or planned runs are silently treated as completed.
predictions_path disappears from aggregate output.
Holm correction order is not deterministic.
```

Use the common `[v2 agent handoff]` block from `common_project_rules.md`.
