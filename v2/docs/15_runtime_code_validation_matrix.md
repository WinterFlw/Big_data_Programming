# 15. Runtime Code Validation Matrix

> 목적: 모든 실행 코드가 `v2/` 안에서 돌아간다는 원칙을 고정하고, 5명이 전체 코드를 전수 검토하지 않으면서도 학습/추론/XAI/report critical path를 검증하도록 역할을 나눈다.

---

## 1. 기준 원칙

이제 v2 실행의 기준은 아래다.

```text
모든 실행 코드는 v2/ 안에 있어야 한다.
v1/은 archival reference일 뿐, v2 실행 중 import하거나 canonical output으로 사용하지 않는다.
```

따라서 실제 학습/XAI/대시보드 runtime 코드는 아래 폴더에 둔다.

```text
v2/runtime/
  utils.py
  experiment_core.py
  experiment_xai.py
  experiment_dashboard.py
  experiment_eda.py
  run_experiments.py
  dashboard_app.py
  requirements.txt
```

v2 pipeline orchestration 코드는 아래 폴더에 둔다.

```text
v2/pipeline/
  cli.py
  runner.py
  training_adapter.py
  statistics.py
  xai.py
  xai_bundle.py
  reporting.py
  artifacts.py
  manifest.py
  schema.py
  paths.py
```

---

## 2. 검증 범위

전체 코드를 다 읽지 않는다. 대신 서버 실행과 최종 산출물을 망칠 수 있는 경로만 본다.

```text
data/runtime setup
-> benchmark adapter
-> train/evaluate/inference output
-> aggregate/statistics
-> XAI smoke
-> xai-bundle
-> report/dashboard
```

검증할 질문:

```text
v2가 v1을 import하지 않는가?
학습 결과가 v2/outputs/experiments/v2_15seed/ 아래로 정규화되는가?
checkpoint가 v2 내부에 남는가?
predictions.csv가 생성되어 추론 결과를 확인할 수 있는가?
aggregate가 smoke metrics를 읽는가?
XAI가 v2 runtime checkpoint를 load할 수 있는가?
report/dashboard가 xai-bundle을 우선 입력으로 삼는가?
```

---

## 3. 5명 역할 배분

| 사람 | 역할 | 검증 범위 | 주 담당 파일 |
|---|---|---|---|
| 1번 | Runtime Training 담당 | 모델/데이터셋/학습 루프/평가 함수 | `v2/runtime/experiment_core.py`, `v2/runtime/utils.py` |
| 2번 | Adapter/CLI 담당 | v2 CLI가 runtime 학습을 호출하고 output 계약을 맞추는지 | `v2/pipeline/training_adapter.py`, `v2/pipeline/runner.py`, `v2/pipeline/artifacts.py` |
| 3번 | Statistics/Inference Output 담당 | `metrics.json`, `predictions.csv`, paired statistics 검증 | `v2/pipeline/statistics.py`, `v2/pipeline/schema.py` |
| 4번 | XAI Runtime 담당 | checkpoint load, sample 고정, XAI smoke | `v2/runtime/experiment_xai.py`, `v2/pipeline/xai.py`, `v2/pipeline/xai_bundle.py` |
| 5번 | Integration/Report/Server 담당 | 전체 stage 연결, report/dashboard, 서버 gate | `v2/run.sh`, `v2/pipeline/cli.py`, `v2/pipeline/reporting.py`, `v2/runtime/experiment_dashboard.py` |

---

## 4. 1번 Runtime Training 담당

### 목표

`v2/runtime/experiment_core.py`가 v2 내부 경로에서 데이터 준비, 학습, 평가, 예측 저장을 수행할 수 있는지 확인한다.

### 보는 파일

```text
v2/runtime/utils.py
v2/runtime/experiment_core.py
v2/runtime/requirements.txt
```

### 확인할 것

```text
BASE_DIR이 v2/를 가리키는가?
DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR가 v2 내부인가?
train_neural_model이 output_root를 받으면 해당 run_dir에 history/metrics/predictions를 쓰는가?
ConditionSpec 8개가 존재하는가?
TransformerConditionClassifier가 VADER 조건과 attention loss 조건을 분리하는가?
evaluate_neural_model이 y_true/y_pred/y_prob를 반환하는가?
```

### 완료 보고

```text
[Runtime Training 검증]
검토 파일:
확인한 함수:
발견한 위험:
수정 필요 여부:
```

---

## 5. 2번 Adapter/CLI 담당

### 목표

`./run.sh e2e benchmark --execute`가 v2/runtime 학습 코드를 호출하고, 산출물을 v2 contract로 정규화하는지 확인한다.

### 보는 파일

```text
v2/pipeline/training_adapter.py
v2/pipeline/runner.py
v2/pipeline/cli.py
v2/pipeline/artifacts.py
v2/pipeline/schema.py
```

### 확인할 것

```text
training_adapter가 v1/을 import하지 않는가?
RUNTIME_DIR가 v2/runtime인가?
stdout.log, stderr.log가 run_dir에 저장되는가?
run_config.json이 생성되는가?
runtime predictions pickle이 predictions.csv로 변환되는가?
checkpoint가 v2/outputs/experiments/v2_15seed/benchmark/checkpoints/로 복사되는가?
resume 시 completed unit을 skip하는가?
```

### 검증 명령

```bash
PYTHON_BIN=python3 ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
python - <<'PY'
from pipeline.training_adapter import _load_runtime_core, _condition_spec, RUNTIME_DIR
core = _load_runtime_core()
spec = _condition_spec(core, 'A_B')
assert spec.model_name == 'bert-base-uncased'
assert str(RUNTIME_DIR).endswith('/v2/runtime')
print('v2_runtime_import_smoke: ok')
PY
```

---

## 6. 3번 Statistics/Inference Output 담당

### 목표

학습 결과와 추론 결과가 aggregate/statistics로 이어지는지 확인한다.

### 보는 파일

```text
v2/pipeline/statistics.py
v2/pipeline/schema.py
v2/docs/03_validation_and_statistics.md
v2/docs/07_output_and_report_contract.md
```

### 확인할 것

```text
benchmark_runs.csv가 metrics.json을 읽는가?
predictions_path가 비어 있지 않게 전달되는가?
same-seed paired comparison인가?
Holm correction은 보조 adjusted p-value로만 해석되는가?
CI와 effect_size가 p-value와 함께 저장되는가?
빈 결과/부분 결과에서도 aggregate가 실패하지 않는가?
```

### 검증 명령

```bash
PYTHON_BIN=python3 ./run.sh e2e aggregate --run-id v2_15seed
python3 - <<'PY'
from pipeline.statistics import compute_paired_tests, apply_holm_correction
manifest = {'statistics': {'paired_tests': ['A_B:D_B']}}
rows = []
for seed, a, d in [(42, .60, .65), (52, .61, .66), (62, .62, .67)]:
    rows.append({'condition':'A_B','seed':seed,'status':'completed','macro_f1':a,'accuracy':a,'weighted_f1':a})
    rows.append({'condition':'D_B','seed':seed,'status':'completed','macro_f1':d,'accuracy':d,'weighted_f1':d})
paired = compute_paired_tests(manifest, rows)
corrected = apply_holm_correction(paired)
assert paired[0]['n_pairs'] == 3
assert round(float(paired[0]['mean_diff']), 4) == 0.05
assert corrected[0]['p_value_holm'] != ''
print('paired_statistics_smoke: ok')
PY
```

---

## 7. 4번 XAI Runtime 담당

### 목표

XAI가 v2/runtime 모델과 v2 benchmark checkpoint를 대상으로 실행될 수 있는지 확인한다.

### 보는 파일

```text
v2/runtime/experiment_xai.py
v2/pipeline/xai.py
v2/pipeline/xai_bundle.py
v2/docs/04_xai_protocol.md
v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md
```

### 확인할 것

```text
XAI runtime이 archive output을 canonical input으로 읽지 않는가?
checkpoint path가 v2/outputs/experiments/v2_15seed/ 내부를 기준으로 연결 가능한가?
sample set을 seed마다 고정할 수 있는가?
xai-primary/deep/ablation 산출물 스키마가 output contract와 맞는가?
xai-bundle이 raw XAI artifact를 다시 계산하지 않고 통합만 하는가?
```

### 검증 명령

```bash
PYTHON_BIN=python3 ./run.sh e2e xai-primary --run-id v2_15seed --dry-run
PYTHON_BIN=python3 ./run.sh e2e xai-bundle --run-id v2_15seed
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json >/tmp/xai_claims_check.json
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json >/tmp/xai_dashboard_check.json
```

---

## 8. 5번 Integration/Report/Server 담당

### 목표

모든 stage가 v2-local code와 v2-local output만 사용한다는 것을 검증하고, full run gate를 관리한다.

### 보는 파일

```text
v2/run.sh
v2/pipeline/cli.py
v2/pipeline/reporting.py
v2/pipeline/paths.py
v2/runtime/experiment_dashboard.py
v2/runtime/dashboard_app.py
v2/docs/11_team_tasking_and_server_run_plan.md
```

### 확인할 것

```text
run.sh가 v2 workspace root로 이동한 뒤 실행하는가?
PYTHON_BIN으로 서버 venv를 지정할 수 있는가?
report/dashboard가 xai_claims.json과 xai_dashboard_bundle.json을 우선 입력으로 보는가?
docs가 v1 실행 의존을 지시하지 않는가?
full 120 benchmark 시작 전에 smoke gate가 문서화되어 있는가?
```

### 검증 명령

```bash
python3 -m compileall runtime pipeline scripts/validate_commit_message.py
python3 -m json.tool configs/v2_15seed.json >/tmp/v2_config_check.json
PYTHON_BIN=python3 ./run.sh e2e --help
PYTHON_BIN=python3 ./run.sh e2e status --run-id v2_15seed
PYTHON_BIN=python3 ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
PYTHON_BIN=python3 ./run.sh e2e aggregate --run-id v2_15seed
PYTHON_BIN=python3 ./run.sh e2e xai-bundle --run-id v2_15seed
PYTHON_BIN=python3 ./run.sh e2e report --run-id v2_15seed
PYTHON_BIN=python3 ./run.sh e2e dashboard --run-id v2_15seed
```

---

## 9. Full Run Gate

아래가 모두 참일 때만 full 120 benchmark를 시작한다.

```text
v2_runtime_import_smoke 통과
A_B seed 42 단일 학습 성공
A_B/D_B seed 42 paired smoke 성공
metrics.json/history.csv/run_config.json/predictions.csv/checkpoint 생성
aggregate가 smoke 결과를 읽고 paired_tests row 생성
checkpoint와 predictions_path가 v2/outputs/experiments/v2_15seed/ 내부를 가리킴
report/dashboard stage가 실패하지 않음
```

full run 전 실제 명령:

```bash
PYTHON_BIN=/path/to/venv/bin/python ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --resume
PYTHON_BIN=/path/to/venv/bin/python ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --execute --resume
PYTHON_BIN=/path/to/venv/bin/python ./run.sh e2e aggregate --run-id v2_15seed
```
