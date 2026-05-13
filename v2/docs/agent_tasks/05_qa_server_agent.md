# 05. QA and Server Agent Brief

> 역할: 서버 실행 전후 검증, 상태 관리, 실패 run 재실행 계획을 담당한다.

---

## 1. 에이전트에게 줄 첫 지시문

```text
당신은 HateSpeachStudy v2_15seed 파이프라인의 QA/Server 담당 에이전트입니다.
목표는 제한된 GPU 서버 실행 기회를 낭비하지 않도록 preflight, dry-run, smoke test, status, failed-run recovery를 점검하는 것입니다.
full benchmark나 full XAI를 임의로 실행하지 말고, 먼저 검증 명령과 smoke run 기준을 확인하세요.
```

---

## 2. 반드시 읽을 문서

```text
docs/06_execution_runbook.md
docs/11_team_tasking_and_server_run_plan.md
docs/07_output_and_report_contract.md
docs/agent_tasks/00_common_agent_rules.md
```

---

## 3. 소유 파일

우선 수정 가능:

```text
pipeline/artifacts.py
pipeline/manifest.py
pipeline/cli.py
```

필요 시 수정 가능:

```text
pipeline/runner.py
configs/v2_15seed.json
```

가급적 수정하지 않을 파일:

```text
runtime/experiment_core.py
runtime/experiment_xai.py
pipeline/statistics.py
pipeline/xai.py
pipeline/xai_bundle.py
pipeline/reporting.py
```

---

## 4. 구현 목표

```text
manifest validation 강화
execution_status.csv 신뢰성 강화
completed/planned/failed 상태 판정
failed_runs.csv 생성
completed_runs.csv 생성
resume/force/only-failed 정책 문서화 또는 CLI 옵션화
split hash와 manifest hash 기록
```

---

## 5. 서버 업로드 전 검증

아래 명령이 로컬에서 통과해야 한다.

```bash
python3 -m compileall pipeline scripts/validate_commit_message.py
python3 -m json.tool configs/v2_15seed.json >/tmp/v2_config_check.json
./run.sh e2e --help
./run.sh e2e plan --run-id v2_15seed --force
./run.sh e2e status --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

---

## 6. 서버 첫 실행 순서

```bash
git status --short
python3 --version
nvidia-smi
python3 -m compileall pipeline scripts/validate_commit_message.py
./run.sh e2e status --run-id v2_15seed
```

학습 연결 후 smoke:

```bash
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --execute --resume
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e status --run-id v2_15seed
```

---

## 7. 중단 기준

즉시 중단:

```text
split hash가 달라짐
metrics schema가 condition마다 다름
checkpoint가 run_id 외부에만 저장됨
VADER feature가 A/B 조건에 들어감
attention loss가 rationale 없는 샘플에 잘못 적용됨
```

failed unit만 재실행:

```text
단일 CUDA OOM
서버 세션 끊김
단일 seed run 실패
일부 XAI sample 실패
```

---

## 8. 완료 기준

아래 산출물이 신뢰 가능해야 한다.

```text
outputs/experiments/v2_15seed/execution_status.csv
outputs/experiments/v2_15seed/plan_status.json
outputs/experiments/v2_15seed/failed_runs.csv
outputs/experiments/v2_15seed/completed_runs.csv
```

`./run.sh e2e status --run-id v2_15seed`에서 아래 관계가 항상 성립해야 한다.

```text
completed + failed + planned = total
total = 120
```
