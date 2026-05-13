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
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

기존 v2.1 명령은 그대로 둔다.

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
pipeline/
  __init__.py
  paths.py
  schema.py
  manifest.py
  artifacts.py
  statistics.py
  xai.py
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
| `pipeline/runner.py` | stage orchestration | 실제 training adapter 연결 |
| `pipeline/manifest.py` | manifest 로드/검증/저장 | manifest schema 강화 |
| `pipeline/artifacts.py` | run unit과 status 관리 | 실패 run 감지 고도화 |
| `pipeline/statistics.py` | benchmark 집계 골격 | paired t-test, Wilcoxon, Holm 보정 구현 |
| `pipeline/xai.py` | XAI 산출물 골격 | SHAP/LIME 실행 함수 연결 |
| `pipeline/reporting.py` | report/dashboard 생성 골격 | 실제 결과표/그림 삽입 |
| `pipeline/schema.py` | condition metadata와 CSV schema | 산출물 스키마 확정 |

---

## 4. 현재 구현 상태

현재 코드는 아래를 수행할 수 있다.

```text
manifest 생성
output directory 생성
condition x seed 실행 계획 생성
execution_status.csv 생성
benchmark aggregate용 빈/부분 CSV 생성
XAI stage별 산출물 파일 골격 생성
final_report.md 초안 생성
dashboard/index.html 초안 생성
```

아직 실제 학습 실행은 연결하지 않았다. 의도적으로 `runner.benchmark(..., execute=True)`는 `NotImplementedError`를 내도록 두었다.

이유:

```text
기존 experiment_core.py의 학습 함수는 존재하지만,
checkpoint path와 output root를 v2 run_id 내부로 완전히 격리하는 작업이 먼저 필요하다.
```

---

## 5. 다음 코드 연결 순서

권장 구현 순서:

```text
1. pipeline.runner.benchmark execute adapter 구현
2. experiment_core.train_neural_model checkpoint output_root 분리
3. condition x seed 단위 resume/skip 구현
4. statistics.py paired test/Holm 보정 구현
5. xai.py primary/deep/ablation runner 구현
6. reporting.py 최종 report/docx/dashboard 연결
```

가장 먼저 할 일은 benchmark 실행 adapter다.

---

## 6. 분업 제안

효율적인 분업 단위:

```text
Person A: benchmark adapter + resume
Person B: statistics aggregate + paired tests
Person C: XAI sample selection + seed stability
Person D: report/dashboard generation
Person E: QA/preflight/status checks
```

각 담당자는 자기 파일을 중심으로 작업하고, 산출물 계약은 `07_output_and_report_contract.md`를 따른다.

서버 실행 기회가 제한된 상황의 구체적인 업무하달, 서버 preflight, 실패 보고 양식은 `11_team_tasking_and_server_run_plan.md`를 따른다.

팀원이 에이전트를 사용해 각자 작업할 경우에는 `agent_tasks/` 아래 역할별 지시서를 사용한다.

코드 주석은 `12_code_commenting_guide.md`를 따르고, 커밋 메시지는 `13_commit_message_policy.md`를 따른다.
