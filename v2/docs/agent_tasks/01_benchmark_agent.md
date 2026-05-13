# 01. Benchmark Agent Brief

> 역할: `condition x seed` 단위 실제 학습 실행을 v2 pipeline에 연결한다.

---

## 1. 에이전트에게 줄 첫 지시문

```text
당신은 HateSpeachStudy v2_15seed 파이프라인의 Benchmark 담당 에이전트입니다.
목표는 ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute 명령이 실제 학습 1개를 수행하게 만드는 것입니다.
기존 experiment_core.py의 학습 로직을 재사용하되, 모든 v2 산출물은 outputs/experiments/v2_15seed/ 아래에 저장되어야 합니다.
역할 밖 파일 수정은 최소화하고, 수정 이유를 명확히 기록하세요.
```

---

## 2. 반드시 읽을 문서

```text
docs/01_model_definition.md
docs/02_e2e_pipeline.md
docs/06_execution_runbook.md
docs/07_output_and_report_contract.md
docs/10_code_implementation_notes.md
docs/11_team_tasking_and_server_run_plan.md
docs/agent_tasks/00_common_agent_rules.md
```

---

## 3. 소유 파일

우선 수정 가능:

```text
pipeline/runner.py
pipeline/artifacts.py
pipeline/schema.py
```

필요 시 수정 가능:

```text
experiment_core.py
configs/v2_15seed.json
```

가급적 수정하지 않을 파일:

```text
pipeline/statistics.py
pipeline/xai.py
pipeline/reporting.py
```

---

## 4. 구현 목표

`benchmark --execute`를 아래 흐름으로 연결한다.

```text
1. manifest load
2. selected condition/seed parse
3. RunUnit 생성
4. 완료된 unit은 --resume 시 skip
5. condition metadata로 model/dataset/hyperparams 결정
6. 기존 train_neural_model 또는 adapter 호출
7. run_dir 내부에 metrics/history/config/log 저장
8. checkpoint는 run_id 내부 benchmark/checkpoints에 저장
9. execution_status.csv 갱신
```

---

## 5. 완료 기준

아래 명령이 성공해야 한다.

```bash
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute
./run.sh e2e status --run-id v2_15seed
./run.sh e2e aggregate --run-id v2_15seed
```

생성 파일:

```text
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/metrics.json
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/history.csv
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/run_config.json
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/stdout.log
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/stderr.log
outputs/experiments/v2_15seed/benchmark/checkpoints/a_b_seed_42.pt
```

---

## 6. Smoke test 단계

1개 run smoke:

```bash
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --resume
```

2개 조건 paired smoke:

```bash
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --execute --resume
```

전체 실행은 팀장 승인 전 하지 않는다.

---

## 7. 절대 금지

```text
full 120-run benchmark를 임의 실행하지 않는다.
기존 outputs/runs를 v2 canonical result로 쓰지 않는다.
기존 checkpoints/를 v2 checkpoint root로 쓰지 않는다.
condition별 hyperparameter를 임의로 다르게 주지 않는다.
```

