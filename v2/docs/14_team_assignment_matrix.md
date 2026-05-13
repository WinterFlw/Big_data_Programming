# 14. Team Assignment Matrix

> 목적: 팀원 5명을 기준으로 v2 end-to-end 구현, 검증, 서버 실행 업무를 명확히 나눈다. 이 문서는 “누가 무엇을 언제까지 맡는가”를 정하고, 상세 코드 검증 범위는 `15_runtime_code_validation_matrix.md`가 담당한다.

---

## 1. 공통 기준

팀원에게 먼저 아래 원칙을 공유한다.

```text
v1은 archive다.
실행과 검증은 v2만 기준으로 한다.
학습/추론/XAI runtime은 v2/runtime/에 있다.
pipeline orchestration은 v2/pipeline/에 있다.
모든 새 산출물은 v2/outputs/experiments/v2_15seed/ 아래에 둔다.
```

v2 end-to-end stage는 아래 순서다.

```text
benchmark -> aggregate -> xai-primary -> xai-deep -> xai-ablation -> xai-bundle -> report -> dashboard
```

이번 작업의 핵심은 두 가지다.

```text
benchmark/statistics: 얼마나 나아졌는가
xai-bundle/report/dashboard: 무엇을 더 남기는가
```

---

## 2. 5명 최종 배분

| 사람 | 역할 | 기간 | 핵심 책임 | 주 담당 코드 |
|---|---|---:|---|---|
| 1번 | Runtime Training | D0-D2 | v2 runtime 학습/평가 코드 검증 | `v2/runtime/experiment_core.py`, `v2/runtime/utils.py` |
| 2번 | Adapter/CLI | D0-D3 | `benchmark --execute`와 output 계약 검증 | `v2/pipeline/training_adapter.py`, `runner.py`, `artifacts.py` |
| 3번 | Statistics/Inference Output | D1-D6 | `metrics.json`, `predictions.csv`, paired statistics | `v2/pipeline/statistics.py`, `schema.py` |
| 4번 | XAI Runtime | D2-D8 | XAI smoke, primary/deep/ablation, bundle 연결 | `v2/runtime/experiment_xai.py`, `xai.py`, `xai_bundle.py` |
| 5번 | Integration/Report/Server | D0-D10 | CLI, preflight, report/dashboard, full-run gate | `v2/run.sh`, `cli.py`, `reporting.py` |

운영 원칙:

```text
1번과 2번이 초반 병목이다.
1번/2번 smoke가 끝나기 전까지 full benchmark는 금지한다.
3번/4번/5번은 기다리지 말고 placeholder와 입력 계약 검증을 먼저 끝낸다.
smoke 결과가 나오면 각자 실제 결과 연동 검증으로 넘어간다.
```

---

## 3. 일정

`D0 = 업무 하달일` 기준이다.

```text
D0:
  역할 확정
  v2-only 원칙 공유
  각자 15_runtime_code_validation_matrix.md의 자기 범위 확인

D1:
  1번: runtime 학습 코드 검토
  2번: adapter/CLI 검토
  3번: aggregate/statistics 빈 결과 검증
  4번: XAI 입력/산출 계약 검토
  5번: preflight/report/dashboard scaffold 검증

D2:
  A_B seed 42 단일 smoke 준비
  v2_runtime_import_smoke 통과 확인
  output path가 v2 내부인지 확인

D3:
  A_B seed 42 단일 smoke 실행
  metrics/history/config/predictions/checkpoint 생성 확인

D4:
  A_B/D_B seed 42 paired smoke 실행
  aggregate가 smoke 결과를 읽는지 확인
  full benchmark 가능 여부 판단

D5-D6:
  full 8 conditions x 15 seeds benchmark 실행

D7:
  aggregate/statistics 최종 생성

D8:
  XAI primary/deep/ablation 실행
  xai-bundle 생성

D9:
  report/dashboard 생성

D10:
  최종 검수
  실패 run 재실행
  제출물 정리
```

---

## 4. 1번 Runtime Training

### 책임

`v2/runtime/experiment_core.py`가 v2 내부 경로에서 학습, 평가, 추론 산출물을 만들 수 있는지 확인한다.

### 검증 범위

```text
v2/runtime/utils.py
v2/runtime/experiment_core.py
v2/runtime/requirements.txt
```

### 확인할 것

```text
BASE_DIR이 v2/를 가리키는가?
DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR가 v2 내부인가?
train_neural_model이 output_root를 받으면 run_dir에 산출물을 쓰는가?
evaluate_neural_model이 y_true/y_pred/y_prob를 반환하는가?
ConditionSpec 8개가 존재하는가?
TransformerConditionClassifier가 VADER/attention 조건을 분리하는가?
```

---

## 5. 2번 Adapter/CLI

### 책임

v2 CLI가 runtime 학습 코드를 호출하고, 산출물을 v2 output contract로 정규화하는지 확인한다.

### 검증 범위

```text
v2/pipeline/training_adapter.py
v2/pipeline/runner.py
v2/pipeline/cli.py
v2/pipeline/artifacts.py
v2/pipeline/schema.py
```

### 확인할 것

```text
training_adapter가 v1을 import하지 않는가?
RUNTIME_DIR가 v2/runtime인가?
stdout.log, stderr.log가 run_dir에 저장되는가?
run_config.json이 생성되는가?
predictions pickle이 predictions.csv로 변환되는가?
checkpoint가 benchmark/checkpoints/로 복사되는가?
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
assert '/v1/' not in str(core.__file__)
print('v2_runtime_import_smoke: ok')
PY
```

---

## 6. 3번 Statistics / Inference Output

### 책임

학습 결과와 추론 결과가 aggregate/statistics로 이어지는지 확인한다.

### 검증 범위

```text
v2/pipeline/statistics.py
v2/pipeline/schema.py
v2/docs/03_validation_and_statistics.md
v2/docs/07_output_and_report_contract.md
```

### 확인할 것

```text
benchmark_runs.csv가 metrics.json을 읽는가?
predictions_path가 유지되는가?
same-seed paired comparison인가?
Holm correction이 적용되는가?
CI와 effect_size가 p-value와 함께 저장되는가?
빈 결과/부분 결과에서도 aggregate가 실패하지 않는가?
```

---

## 7. 4번 XAI Runtime

### 책임

XAI가 v2 runtime checkpoint와 v2 output을 기준으로 실행될 수 있는지 확인한다.

### 검증 범위

```text
v2/runtime/experiment_xai.py
v2/pipeline/xai.py
v2/pipeline/xai_bundle.py
v2/docs/04_xai_protocol.md
v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md
```

### 확인할 것

```text
XAI runtime이 archive output을 canonical input으로 쓰지 않는가?
checkpoint path가 v2/outputs/experiments/v2_15seed/ 내부 기준으로 연결 가능한가?
sample set을 seed마다 고정할 수 있는가?
xai-primary/deep/ablation 산출물 스키마가 output contract와 맞는가?
xai-bundle은 raw XAI 재계산이 아니라 통합만 하는가?
```

---

## 8. 5번 Integration / Report / Server

### 책임

전체 stage가 v2-local code와 v2-local output만 사용하는지 확인하고 full-run gate를 관리한다.

### 검증 범위

```text
v2/run.sh
v2/pipeline/cli.py
v2/pipeline/reporting.py
v2/pipeline/paths.py
v2/runtime/experiment_dashboard.py
v2/runtime/dashboard_app.py
v2/docs/11_team_tasking_and_server_run_plan.md
```

### 서버 업로드 전 검증 명령

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

아래가 모두 통과해야 full 120 benchmark를 시작한다.

```text
v2_runtime_import_smoke 통과
A_B seed 42 단일 smoke 성공
A_B/D_B seed 42 paired smoke 성공
metrics.json/history.csv/run_config.json/predictions.csv/checkpoint 생성
aggregate가 smoke 결과를 읽고 paired_tests row 생성
checkpoint_path와 predictions_path가 v2/outputs/experiments/v2_15seed/ 내부를 가리킴
report/dashboard stage가 실패하지 않음
```

실제 smoke 명령:

```bash
PYTHON_BIN=/path/to/venv/bin/python ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --resume
PYTHON_BIN=/path/to/venv/bin/python ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --execute --resume
PYTHON_BIN=/path/to/venv/bin/python ./run.sh e2e aggregate --run-id v2_15seed
```

---

## 10. 완료 보고 양식

```text
[v2 작업 완료]
담당:
검증 범위:
수정한 파일:
실행한 명령어:
생성/변경된 산출물:
통과한 검증:
남은 위험:
다음 사람이 이어받을 부분:
```
