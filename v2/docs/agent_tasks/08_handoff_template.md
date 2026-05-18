# 08. Agent Handoff Template

> 역할: 팀원이 에이전트 작업 결과를 공유할 때 쓰는 공통 인수인계 양식이다.

---

## 1. 완료 보고 템플릿

```text
[v2 agent handoff]
Role:
Owner:
Date:

Goal:

Files changed:

Commands run:

Artifacts created/updated:

Validation passed:

Commit message:

Known limitations:

Risks:

Next owner:

Notes for server run:
```

---

## 2. Benchmark 완료 예시

```text
[v2 agent handoff]
Role: Benchmark
Owner: 홍길동
Date: 2026-05-13

Goal:
condition x seed 단위 학습 adapter 연결

Files changed:
pipeline/runner.py
pipeline/training_adapter.py

Commands run:
python3 -m compileall pipeline scripts/validate_commit_message.py
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute
./run.sh e2e aggregate --run-id v2_15seed

Artifacts created/updated:
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/metrics.json
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/history.csv

Validation passed:
A_B seed 42 completed
aggregate reads the run

Known limitations:
Full 120-run benchmark not executed

Risks:
RoBERTa conditions not smoke-tested yet

Next owner:
Statistics 담당

Notes for server run:
Start with A_B,D_B seed 42 smoke before full benchmark
```

---

## 3. Statistics 완료 예시

```text
[v2 agent handoff]
Role: Statistics
Owner:
Date:

Goal:
15 seed 요약과 핵심 A_B vs D_B paired test 구현

Files changed:
pipeline/statistics.py
pipeline/schema.py

Commands run:
./run.sh e2e aggregate --run-id v2_15seed

Artifacts created/updated:
benchmark_runs.csv
benchmark_summary.csv
paired_tests.csv
paired_tests_holm.csv (adjusted p-value는 보조 확인용)

Validation passed:
A_B vs D_B row generated when both conditions have same-seed metrics

Known limitations:
Full 15-seed result not available yet

Risks:
Small n fallback behavior should be reviewed

Next owner:
Report 담당

Notes for server run:
Run aggregate immediately after benchmark smoke
```

---

## 4. 실패 보고 템플릿

```text
[v2 failure report]
Stage:
Condition:
Seed:
Command:
Log path:
Last successful artifact:
Observed error:
Suspected cause:
Suggested action:
Need full stop: yes/no
```

---

## 5. 서버 실행 승인 요청 템플릿

```text
[v2 server run approval request]
Requested stage:
Command:
Expected runtime:
Expected artifacts:
Smoke tests passed:
Rollback/retry plan:
Risk:
Approval needed from:
```
