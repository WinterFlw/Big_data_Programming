# 06. Integration Lead Agent Brief

> 역할: 여러 담당자의 변경사항을 충돌 없이 합치고, 전체 v2 pipeline이 끊기지 않는지 확인한다.

---

## 1. 에이전트에게 줄 첫 지시문

```text
당신은 HateSpeachStudy v2_15seed 파이프라인의 Integration Lead 에이전트입니다.
목표는 Benchmark, Statistics, XAI, Report, QA 담당자의 변경사항을 통합하고, ./run.sh e2e 명령 체계가 깨지지 않도록 검증하는 것입니다.
직접 기능을 크게 새로 만들기보다, 각 담당자의 변경이 산출물 계약과 충돌하지 않는지 확인하고 필요한 최소 수정만 수행하세요.
```

---

## 2. 반드시 읽을 문서

```text
docs/00_reading_order.md
docs/02_e2e_pipeline.md
docs/07_output_and_report_contract.md
docs/10_code_implementation_notes.md
docs/11_team_tasking_and_server_run_plan.md
docs/agent_tasks/00_common_agent_rules.md
```

---

## 3. 소유 파일

우선 검토 대상:

```text
run.sh
run_experiments.py
configs/v2_15seed.json
pipeline/
docs/
```

직접 수정은 최소화한다.

---

## 4. 통합 체크리스트

```text
./run.sh e2e --help가 동작하는가?
./run.sh e2e status --run-id v2_15seed가 동작하는가?
./run.sh e2e benchmark --dry-run이 동작하는가?
각 stage가 outputs/experiments/v2_15seed/만 canonical output으로 쓰는가?
CSV schema가 07_output_and_report_contract.md와 맞는가?
각 담당자가 자기 파일 밖 변경을 설명했는가?
```

---

## 5. 통합 검증 명령

```bash
python3 -m compileall run_experiments.py pipeline
python3 -m json.tool configs/v2_15seed.json >/tmp/v2_config_check.json
./run.sh e2e --help
./run.sh e2e plan --run-id v2_15seed --force
./run.sh e2e status --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

---

## 6. 통합 완료 기준

```text
CLI가 깨지지 않는다.
manifest와 execution_status가 생성된다.
120 planned units가 확인된다.
aggregate/report/dashboard가 결과가 없어도 실패하지 않는다.
실제 학습 연결 후 smoke run 1-2개가 통과한다.
```

---

## 7. 통합 보고 양식

```text
[v2 integration report]
Merged roles:
Files touched:
Commands run:
Pass:
Fail:
Blocking issue:
Ready for server smoke: yes/no
```

