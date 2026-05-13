# 11. Team Tasking and Server Run Plan

> 목적: 서버 실행 기회가 제한된 상황에서 팀원이 무엇을 언제까지 구현하고, 어떤 검증을 통과한 뒤 GPU 서버에 올릴지 정한다. 이 문서는 팀장 업무하달서이자 서버 실행 작전서다.

---

## 1. 팀 전체 공지문

팀원에게는 아래 문장을 그대로 공유한다.

```text
이번 목표는 기존 v2.1 결과를 조금 고치는 것이 아니라, v2_15seed라는 새 end-to-end 실험 라인을 완성하는 것입니다.

모든 새 결과는 outputs/experiments/v2_15seed/ 아래에 저장합니다.
기존 outputs/reports, outputs/xai, outputs/runs를 직접 덮어쓰지 않습니다.

서버 실행 기회가 제한되어 있으므로, 서버에 올리기 전에 로컬에서 CLI, manifest, dry-run, smoke test를 반드시 통과시켜야 합니다.
서버에서는 실험을 새로 디버깅하는 것이 아니라, 이미 검증된 실행 계획을 돌리는 것을 목표로 합니다.
```

공통으로 읽을 문서:

```text
docs/00_reading_order.md
docs/01_model_definition.md
docs/02_e2e_pipeline.md
docs/06_execution_runbook.md
docs/07_output_and_report_contract.md
docs/10_code_implementation_notes.md
docs/11_team_tasking_and_server_run_plan.md
docs/12_code_commenting_guide.md
docs/13_commit_message_policy.md
docs/agent_tasks/README.md
```

공통으로 실행할 확인 명령:

```bash
./run.sh e2e status --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
```

---

## 2. 전체 일정 구조

서버 실행 전까지의 흐름은 아래처럼 나눈다.

```text
Phase 0: 문서/역할 확정
Phase 1: 코드 연결
Phase 2: 로컬 smoke test
Phase 3: 서버 preflight
Phase 4: 서버 benchmark 실행
Phase 5: aggregate/statistics
Phase 6: XAI 실행
Phase 7: xai-bundle 통합
Phase 8: report/dashboard 생성
Phase 9: 결과 회수와 제출 전 검수
```

핵심 원칙:

```text
서버에서 처음 발견될 문제를 줄인다.
서버에서는 full run보다 smoke run을 먼저 실행한다.
실패 run은 전체 재실행이 아니라 failed unit만 재실행한다.
모든 stage는 resume 가능해야 한다.
```

---

## 3. 역할 분담 요약

| 역할 | 담당 영역 | 주요 파일 | 최종 산출물 |
|---|---|---|---|
| A. Benchmark 담당 | 실제 학습 실행 연결 | `pipeline/runner.py`, `pipeline/training_adapter.py`, `runtime/experiment_core.py` 참고 | condition x seed 학습 실행 |
| B. Statistics 담당 | 15 seed 통계 검정 | `pipeline/statistics.py` | summary/test CSV |
| C. XAI 담당 | XAI sample/metric/stability | `pipeline/xai.py`, `runtime/experiment_xai.py` 참고 | XAI CSV/JSON/case |
| D. XAI Bundle 담당 | full XAI evidence bundle | `pipeline/xai_bundle.py` | xai_claims, dashboard bundle |
| E. Report 담당 | 최종 보고서/대시보드 | `pipeline/reporting.py` | final_report, dashboard |
| F. QA/Server 담당 | preflight/status/retry | `pipeline/artifacts.py`, `pipeline/manifest.py` | 서버 실행 체크리스트 |

팀원별 기간, write set, read-only set, 완료 기준은 `docs/14_team_assignment_matrix.md`를 기준으로 한다.

팀원이 3명일 때는 아래처럼 합친다.

```text
Person 1: Benchmark + Server 실행
Person 2: Statistics + QA
Person 3: XAI Core + XAI Bundle + Report
```

팀원이 4명일 때:

```text
Person 1: Benchmark
Person 2: Statistics
Person 3: XAI Core + XAI Bundle
Person 4: Report + QA/Server
```

팀원이 5명일 때:

```text
Person 1: Benchmark
Person 2: Statistics
Person 3: XAI
Person 4: XAI Bundle + Report
Person 5: QA/Server
```

에이전트를 사용할 경우 각 담당자는 아래 문서를 자기 에이전트에게 전달한다.

```text
Benchmark: docs/agent_tasks/01_benchmark_agent.md
Statistics: docs/agent_tasks/02_statistics_agent.md
XAI: docs/agent_tasks/03_xai_agent.md
XAI Evidence Bundle: docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md
Report/Dashboard: docs/agent_tasks/04_report_dashboard_agent.md
QA/Server: docs/agent_tasks/05_qa_server_agent.md
Integration Lead: docs/agent_tasks/06_integration_lead_agent.md
Review: docs/agent_tasks/07_review_agent.md
Handoff: docs/agent_tasks/08_handoff_template.md
Dispatch Prompts: docs/agent_tasks/10_team_dispatch_prompts.md
```

---

## 4. A 담당: Benchmark 실행 연결

### 4.1 업무 지시

```text
pipeline.runner.benchmark(..., execute=True)를 실제 학습 실행으로 연결해주세요.
실행 단위는 condition x seed입니다.
기존 v2/runtime/experiment_core.py의 학습 로직을 참고하거나 adapter로 재사용하되, 모든 산출물은 outputs/experiments/v2_15seed/ 내부에 저장되도록 격리해야 합니다.
```

### 4.2 담당 파일

```text
pipeline/runner.py
pipeline/training_adapter.py
pipeline/schema.py
pipeline/artifacts.py
runtime/experiment_core.py 참고
```

### 4.3 완료 기준

아래 명령이 실제 학습 1개를 돌릴 수 있어야 한다.

```bash
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute
```

생성되어야 하는 파일:

```text
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/metrics.json
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/history.csv
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/run_config.json
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/stdout.log
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/stderr.log
```

체크포인트는 아래처럼 run_id 내부에 저장한다.

```text
outputs/experiments/v2_15seed/benchmark/checkpoints/a_b_seed_42.pt
```

### 4.4 금지 사항

```text
기존 outputs/runs를 canonical output으로 쓰지 않는다.
기존 checkpoints/를 v2 최종 checkpoint 위치로 쓰지 않는다.
조건별로 다른 hyperparameter를 임의 적용하지 않는다.
같은 seed에서 condition마다 다른 데이터 split을 쓰지 않는다.
```

---

## 5. B 담당: Statistics 집계

### 5.1 업무 지시

```text
120개 benchmark run 결과를 읽어서 condition별 요약, same-seed paired test, Holm 보정 결과를 생성해주세요.
p-value만 보고하지 말고 mean difference, 95% CI, effect size를 같이 출력해야 합니다.
```

### 5.2 담당 파일

```text
pipeline/statistics.py
pipeline/schema.py
docs/03_validation_and_statistics.md
docs/07_output_and_report_contract.md
```

### 5.3 완료 기준

아래 명령이 통계 산출물을 만든다.

```bash
./run.sh e2e aggregate --run-id v2_15seed
```

생성되어야 하는 파일:

```text
outputs/experiments/v2_15seed/benchmark/benchmark_runs.csv
outputs/experiments/v2_15seed/benchmark/benchmark_summary.csv
outputs/experiments/v2_15seed/benchmark/paired_tests.csv
outputs/experiments/v2_15seed/benchmark/paired_tests_holm.csv
```

필수 비교:

```text
A_B vs D_B
A_B vs B_B
A_B vs C_B
B_B vs D_B
C_B vs D_B
A_R vs D_R
D_B vs D_R
```

필수 컬럼:

```text
comparison
metric
n_pairs
mean_diff
ci_low
ci_high
p_value
p_value_holm
effect_size
significant_0_05
```

---

## 6. C 담당: XAI 실행

### 6.1 업무 지시

```text
XAI는 모델 설계 근거가 아니라 사후 검증입니다.
Primary XAI는 A_B와 D_B를 15 seed 전체에서 비교하고, 같은 sample set을 모든 seed checkpoint에 동일하게 적용해야 합니다.
Deep XAI는 median-performing seed를 사용해 정성 사례를 만듭니다.
이 담당자는 xai-primary, xai-deep, xai-ablation까지 책임지고, xai-bundle은 D 담당자에게 넘깁니다.
```

### 6.2 담당 파일

```text
pipeline/xai.py
runtime/experiment_xai.py 참고
docs/04_xai_protocol.md
docs/08_xai_report_template.md
```

### 6.3 완료 기준

아래 명령이 XAI 산출물을 만든다.

```bash
./run.sh e2e xai-primary --run-id v2_15seed --resume
./run.sh e2e xai-deep --run-id v2_15seed
./run.sh e2e xai-ablation --run-id v2_15seed
```

생성되어야 하는 파일:

```text
outputs/experiments/v2_15seed/xai/samples/primary_samples.csv
outputs/experiments/v2_15seed/xai/primary/seed_level_metrics.csv
outputs/experiments/v2_15seed/xai/primary/paired_xai_tests.csv
outputs/experiments/v2_15seed/xai/primary/seed_stability.csv
outputs/experiments/v2_15seed/xai/deep/case_summary.csv
outputs/experiments/v2_15seed/xai/ablation/xai_ablation_metrics.csv
outputs/experiments/v2_15seed/xai/xai_summary.json
```

필수 metric:

```text
SHAP-LIME Overlap@5
Rationale Precision@5
Rationale Recall@5
Rationale F1@5
Comprehensiveness
Sufficiency
Leave-one-out Drop
Top-k Jaccard across seeds
Rank correlation across seeds
```

---

## 7. D 담당: XAI Evidence Bundle

### 7.1 업무 지시

```text
xai-primary, xai-deep, xai-ablation 결과를 읽어 full XAI evidence bundle을 생성해주세요.
이 stage는 SHAP/LIME을 다시 계산하지 않고, report/dashboard가 우선 소비할 계약 파일을 만드는 통합 stage입니다.
```

### 7.2 담당 파일

```text
pipeline/xai_bundle.py
docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md
docs/07_output_and_report_contract.md
```

### 7.3 완료 기준

아래 명령이 evidence bundle 산출물을 만든다.

```bash
./run.sh e2e xai-bundle --run-id v2_15seed
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_run_metadata.json >/tmp/xai_meta_check.json
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json >/tmp/xai_claims_check.json
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_interpretation_cards.json >/tmp/xai_cards_check.json
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json >/tmp/xai_dashboard_check.json
```

생성되어야 하는 파일:

```text
outputs/experiments/v2_15seed/xai/evidence_bundle/evidence_inventory.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_run_metadata.json
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_sample_manifest.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_predictions.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/method_agreement.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/faithfulness_metrics.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/context_metrics.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/plausibility_metrics.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/subgroup_xai_metrics.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_risk_flags.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_interpretation_cards.json
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json
outputs/experiments/v2_15seed/xai/evidence_bundle/token_attributions.jsonl
outputs/experiments/v2_15seed/xai/evidence_bundle/README.md
```

---

## 8. E 담당: Report와 Dashboard

### 8.1 업무 지시

```text
v2 run_id 내부의 benchmark/statistics/XAI 산출물만 읽어서 final_report와 dashboard를 생성해주세요.
보고서는 결과 해석이 가능해야 하고, dashboard는 실행 상태와 주요 표를 빠르게 확인할 수 있어야 합니다.
가능하면 report와 dashboard는 raw deep case보다 `xai/evidence_bundle/xai_claims.json`과 `xai/evidence_bundle/xai_dashboard_bundle.json`을 우선 소비해야 합니다.
```

### 8.2 담당 파일

```text
pipeline/reporting.py
docs/07_output_and_report_contract.md
docs/08_xai_report_template.md
```

### 8.3 완료 기준

아래 명령이 최종 산출물을 만든다.

```bash
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

생성되어야 하는 파일:

```text
outputs/experiments/v2_15seed/reports/final_report.md
outputs/experiments/v2_15seed/reports/final_report.docx
outputs/experiments/v2_15seed/dashboard/index.html
```

보고서 필수 내용:

```text
모델 정의
8조건 ablation matrix
15 seed 반복 이유
성능 요약
paired test와 Holm 보정
XAI primary 결과
XAI seed stability
정성 사례
한계와 threat to validity
재현 명령어
```

---

## 9. F 담당: QA와 서버 실행 관리

### 9.1 업무 지시

```text
서버에 올리기 전에 실행 환경, manifest, 데이터 split, dry-run, smoke test를 확인해주세요.
서버에서는 status를 계속 기록하고, 실패 run이 생기면 전체 재실행이 아니라 실패 unit만 재실행하도록 관리해주세요.
```

### 9.2 담당 파일

```text
pipeline/artifacts.py
pipeline/manifest.py
pipeline/cli.py
docs/06_execution_runbook.md
```

### 9.3 완료 기준

아래 명령이 신뢰 가능한 상태를 출력해야 한다.

```bash
./run.sh e2e plan --run-id v2_15seed
./run.sh e2e status --run-id v2_15seed
```

status에서 확인할 것:

```text
total = 120
completed + failed + planned = total
완료 run은 resume 시 skip 가능
failed run은 별도 목록으로 추출 가능
manifest hash와 split hash 기록
```

---

## 10. 서버 업로드 전 체크리스트

서버에 올리기 전 로컬에서 아래가 모두 통과해야 한다.

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

실제 학습 연결 후에는 아래 smoke test도 통과해야 한다.

```bash
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
```

통과 기준:

```text
A_B seed 42 run이 완료된다.
metrics.json이 생성된다.
benchmark_runs.csv에 해당 run이 completed로 반영된다.
report가 생성된다.
기존 outputs/runs나 checkpoints를 덮어쓰지 않는다.
```

---

## 11. 서버에서의 실행 순서

서버 접속 후 첫 명령:

```bash
git status --short
python3 --version
nvidia-smi
python3 -m compileall pipeline scripts/validate_commit_message.py
./run.sh e2e status --run-id v2_15seed
```

서버 smoke run:

```bash
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --execute --resume
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e status --run-id v2_15seed
```

smoke run이 통과하면 전체 benchmark:

```bash
./run.sh e2e benchmark --run-id v2_15seed --resume
```

benchmark 완료 후:

```bash
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e xai-primary --run-id v2_15seed --resume
./run.sh e2e xai-deep --run-id v2_15seed
./run.sh e2e xai-ablation --run-id v2_15seed
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

---

## 12. 서버 실행 중 중단 기준

아래 상황이면 전체 실행을 멈춘다.

```text
데이터 split hash가 로컬/서버 또는 실행 간 달라짐
같은 seed에서 condition별 sample order가 달라짐
VADER feature가 A/B 조건에 들어감
attention loss가 rationale 없는 샘플에 잘못 적용됨
metrics schema가 condition마다 다름
초기 smoke run에서 metrics.json이 생성되지 않음
NaN loss가 반복 발생함
checkpoint가 run_id 외부에만 저장됨
```

아래 상황이면 전체 중단이 아니라 failed unit만 재실행한다.

```text
일시적 CUDA out of memory
서버 세션 끊김
단일 seed run 실패
XAI 일부 sample 실패
```

---

## 13. 실패 보고 양식

실패가 생기면 팀 채팅방에 아래 형식으로 보고한다.

```text
[v2 실패 보고]
stage:
condition:
seed:
명령어:
실패 로그 경로:
마지막 정상 산출물:
추정 원인:
필요 조치:
전체 중단 필요 여부: yes/no
```

예시:

```text
[v2 실패 보고]
stage: benchmark
condition: D_B
seed: 92
명령어: ./run.sh e2e benchmark --run-id v2_15seed --conditions D_B --seeds 92 --execute --resume
실패 로그 경로: outputs/experiments/v2_15seed/benchmark/runs/d_b/seed_92/stderr.log
마지막 정상 산출물: history.csv epoch 3까지 생성
추정 원인: CUDA OOM
필요 조치: batch_size 임시 조정 가능 여부 확인
전체 중단 필요 여부: no
```

---

## 14. 팀원 완료 보고 양식

각 담당자는 작업 완료 시 아래 형식으로 보고한다.

```text
[v2 작업 완료]
담당:
수정한 파일:
실행한 명령어:
생성/변경된 산출물:
통과한 검증:
남은 위험:
다음 사람이 이어받을 부분:
```

예시:

```text
[v2 작업 완료]
담당: Statistics
수정한 파일: pipeline/statistics.py
실행한 명령어: ./run.sh e2e aggregate --run-id v2_15seed
생성/변경된 산출물: benchmark_summary.csv, paired_tests_holm.csv
통과한 검증: A_B vs D_B paired test row 생성 확인
남은 위험: 120개 full run 결과가 아직 없어서 빈 run 처리만 검증됨
다음 사람이 이어받을 부분: reporting.py에서 paired_tests_holm.csv 읽기
```

---

## 15. 서버 시간 절약 전략

서버 시간을 아끼기 위해 아래 원칙을 따른다.

```text
로컬에서 가능한 것은 로컬에서 끝낸다.
서버에서는 GPU가 필요한 benchmark/XAI만 돌린다.
첫 서버 실행은 full run이 아니라 A_B/D_B seed 42 smoke run이다.
full benchmark 전에는 반드시 aggregate가 smoke result를 읽는지 확인한다.
XAI는 benchmark가 끝난 뒤 median seed와 sample set이 확정된 다음 실행한다.
```

가능하면 benchmark와 XAI를 같은 서버 세션에서 바로 이어서 돌리지 않는다.
benchmark 결과를 먼저 회수해서 통계적으로 말이 되는지 확인한 뒤 XAI를 시작한다.

---

## 16. 최종 제출 전 검수

제출 전 확인 파일:

```text
outputs/experiments/v2_15seed/manifest.json
outputs/experiments/v2_15seed/execution_status.csv
outputs/experiments/v2_15seed/benchmark/benchmark_runs.csv
outputs/experiments/v2_15seed/benchmark/benchmark_summary.csv
outputs/experiments/v2_15seed/benchmark/paired_tests_holm.csv
outputs/experiments/v2_15seed/xai/xai_summary.json
outputs/experiments/v2_15seed/reports/final_report.md
outputs/experiments/v2_15seed/reports/final_report.docx
outputs/experiments/v2_15seed/dashboard/index.html
```

최종 검수 질문:

```text
120개 benchmark run이 모두 completed인가?
A_B vs D_B 비교가 같은 seed paired design으로 계산되었는가?
p-value와 함께 CI/effect size가 보고되었는가?
XAI sample set이 seed 간 고정되었는가?
보고서에서 XAI를 인과적 증명처럼 과장하지 않았는가?
재현 명령어가 보고서에 들어갔는가?
```
