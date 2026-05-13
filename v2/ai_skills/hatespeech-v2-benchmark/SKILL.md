---
name: hatespeech-v2-benchmark
description: Portable AI instructions for HateSpeachStudy v2 benchmark, training, inference, and runtime adapter work. Use when an agent is asked to validate or modify v2/runtime training code, v2/pipeline/training_adapter.py, benchmark --execute, smoke runs, checkpoints, metrics.json, predictions.csv, resume/force behavior, or the 8-condition x 15-seed benchmark execution plan. Works for Codex, Claude, Gemini, Cursor, and other markdown-reading coding agents.
---

# HateSpeachStudy v2 Benchmark Skill

## First Move

Read:

```text
v2/ai_skills/common_project_rules.md
v2/docs/01_model_definition.md
v2/docs/06_execution_runbook.md
v2/docs/07_output_and_report_contract.md
v2/docs/15_runtime_code_validation_matrix.md
```

## Mission

Make sure `benchmark --execute` runs one condition x seed through v2-local runtime code and writes stable artifacts:

```text
run_config.json
metrics.json
history.csv
predictions.csv
stdout.log
stderr.log
checkpoint under v2/outputs/experiments/v2_15seed/
```

## Ownership

Primary files:

```text
v2/pipeline/training_adapter.py
v2/pipeline/runner.py
v2/pipeline/artifacts.py
v2/pipeline/schema.py
v2/runtime/experiment_core.py
v2/runtime/utils.py
```

Do not change statistics, XAI, or report files unless a contract mismatch requires it.

## Invariants

```text
RUNTIME_DIR must be v2/runtime.
No v1 import is allowed.
All conditions share controlled family hyperparameters.
Same seed must be comparable across all conditions.
Smoke run is execution validation, not final statistical evidence.
```

## Smoke Commands

Cheap checks:

```bash
PYTHON_BIN=python3 ./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
PYTHONPATH=v2 python3 - <<'PY'
from pipeline.training_adapter import _load_runtime_core, _condition_spec, RUNTIME_DIR
core = _load_runtime_core()
spec = _condition_spec(core, 'A_B')
assert spec.model_name == 'bert-base-uncased'
assert str(RUNTIME_DIR).endswith('/v2/runtime')
assert '/v1/' not in str(core.__file__)
print('v2_runtime_import_smoke: ok')
PY
```

First expensive gate:

```bash
PYTHON_BIN=/path/to/venv/bin/python ./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --resume
```

Second expensive gate:

```bash
PYTHON_BIN=/path/to/venv/bin/python ./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --execute --resume
PYTHON_BIN=/path/to/venv/bin/python ./v2/run.sh e2e aggregate --run-id v2_15seed
```

## Completion Checks

Confirm generated paths stay under:

```text
v2/outputs/experiments/v2_15seed/benchmark/
```

Use the common `[v2 agent handoff]` block from `common_project_rules.md`.
