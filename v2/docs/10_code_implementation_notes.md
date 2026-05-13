# 10. Code Implementation Notes

> 목적: v2 문서 설계를 코드의 큰 틀로 옮긴 위치와, 팀원이 분업할 때 맡을 수 있는 파일 책임을 정리한다.

---

## 1. 새 실행 진입점

v2 전용 명령은 아래 형태로 실행한다.

```bash
./run.sh e2e plan --run-id v2_15seed
./run.sh e2e status --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --dry-run
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e xai-primary --run-id v2_15seed
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

runtime 단독 점검 명령은 보조용으로 남긴다. 팀 작업과 서버 실행의 기준은
항상 `e2e` 명령이다.

```bash
./run.sh data
./run.sh benchmark
./run.sh xai
./run.sh dashboard
```

---

## 2. 추가된 코드 구조

```text
configs/v2_15seed.json
runtime/
  utils.py
  experiment_core.py
  experiment_xai.py
  experiment_dashboard.py
  experiment_eda.py
  run_experiments.py
  dashboard_app.py
pipeline/
  __init__.py
  paths.py
  schema.py
  manifest.py
  artifacts.py
  statistics.py
  xai.py
  xai_bundle.py
  reporting.py
  runner.py
  cli.py
```

---

## 3. 파일별 책임

| 파일 | 책임 | 담당자가 구현할 다음 내용 |
|---|---|---|
| `configs/v2_15seed.json` | 15 seed v2 실행 설정 | 실제 GPU 환경에 맞춘 batch/epoch 조정 |
| `pipeline/cli.py` | `./run.sh e2e ...` CLI | 옵션 추가, help 정리 |
| `pipeline/runner.py` | stage orchestration | smoke/full run gate와 실패 복구 고도화 |
| `pipeline/manifest.py` | manifest 로드/검증/저장 | manifest schema 강화 |
| `pipeline/artifacts.py` | run unit과 status 관리 | 실패 run 감지 고도화 |
| `pipeline/statistics.py` | benchmark 집계와 paired 통계 | full 결과에서 CI/effect size/p-value 검수 |
| `pipeline/xai.py` | XAI 산출물 골격 | SHAP/LIME 실행 함수 연결, sample selection, seed stability |
| `pipeline/xai_bundle.py` | XAI evidence bundle 계약 | primary/deep/ablation 결과를 `xai_claims.json`, `xai_dashboard_bundle.json`으로 통합 |
| `pipeline/reporting.py` | report/dashboard 생성 골격 | `xai_claims.json`, `xai_dashboard_bundle.json` 우선 연결 |
| `pipeline/schema.py` | condition metadata와 CSV schema | 산출물 스키마 확정 |

---

## 4. 현재 구현 상태

현재 코드는 아래를 수행할 수 있다.

```text
manifest 생성
output directory 생성
condition x seed 실행 계획 생성
execution_status.csv 생성
benchmark --execute adapter 연결
metrics/history/config/predictions/checkpoint v2 output 정규화
benchmark aggregate용 빈/부분 CSV 생성
paired test/Holm/effect size/CI 계산
XAI stage별 산출물 파일 골격 생성
xai/evidence_bundle/ 계약 반영
xai-bundle CLI stage 생성
final_report.md 초안 생성
dashboard/index.html 초안 생성
```

`runner.benchmark(..., execute=True)`는 `pipeline/training_adapter.py`를 통해 `runtime/experiment_core.py`의 v2-local 학습 로직을 호출한다.

주의:

```text
실제 학습 smoke는 아직 이 문서 작업에서 수행하지 않았다.
서버 또는 로컬 GPU 환경에서 A_B seed 42를 먼저 실행해 학습 시간, 메모리, checkpoint load를 검증해야 한다.
```

---

## 5. 다음 코드 연결 순서

권장 구현 순서:

```text
1. A_B seed 42 단일 smoke 학습 실행
2. A_B/D_B seed 42 paired smoke 학습 실행
3. aggregate가 smoke metrics를 읽어 paired row를 생성하는지 확인
4. full 120 benchmark 실행
5. xai.py primary/deep/ablation runner 구현
6. xai_bundle.py evidence bundle builder가 실제 XAI artifact를 읽도록 구현
7. reporting.py 최종 report/docx/dashboard가 evidence bundle을 우선 읽도록 연결
```

가장 먼저 할 일은 benchmark adapter smoke 검증이다.

---

## 6. 분업 제안

효율적인 분업 단위:

```text
Person A: benchmark adapter + resume
Person B: statistics aggregate + paired tests
Person C: XAI sample selection + seed stability
Person D: XAI evidence bundle + XAI runtime 검증
Person E: report/dashboard + QA/preflight/status checks
```

각 담당자는 자기 파일을 중심으로 작업하고, 산출물 계약은 `07_output_and_report_contract.md`를 따른다.

서버 실행 기회가 제한된 상황의 구체적인 업무하달, 서버 preflight, 실패 보고 양식은 `11_team_tasking_and_server_run_plan.md`를 따른다.

팀원별 기간과 코드 책임 범위는 `14_team_assignment_matrix.md`를 따른다.

팀원이 에이전트를 사용해 각자 작업할 경우에는 `agent_tasks/` 아래 역할별 지시서를 사용한다.

코드 주석은 `12_code_commenting_guide.md`를 따르고, 커밋 메시지는 `13_commit_message_policy.md`를 따른다.
