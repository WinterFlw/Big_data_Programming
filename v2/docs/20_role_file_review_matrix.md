# 20. 역할별 파일 리뷰 책임 매트릭스

> 목적: "코드 리뷰 담당 1명" 구조를 피하고, 각 팀원이 자기 파트의 1차 코드 리뷰를 책임지도록 폴더/파일 단위로 경계를 고정한다.

---

## 1. 기본 원칙

```text
1번은 전체 코드리뷰 독박이 아니다.
각 담당자는 자기 파트 코드와 산출물을 1차 리뷰한다.
1번은 E2E 연결부, output path, smoke gate, full run GO/STOP만 취합한다.
담당 범위 밖 파일을 수정해야 하면 보고 양식에 이유와 영향 범위를 적는다.
```

역할 흐름:

```text
1번 E2E Gate 총괄
2번 학습 실행 / 실험 관리
3번 결과 분석 / 통계 해석
4번 XAI 설명 / evidence bundle
5번 발표자료 / 최종 보고서 제작
```

---

## 2. 파일 책임 요약표

| 사람 | 역할 | 1차 리뷰 폴더/파일 | 1번이 확인하는 연결부 |
|---|---|---|---|
| 1번 | E2E Gate 총괄 | `v2/run.sh`, `v2/pipeline/cli.py`, `v2/pipeline/runner.py`, `v2/pipeline/manifest.py`, `v2/pipeline/paths.py`, `v2/pipeline/artifacts.py`, `v2/configs/v2_15seed.json`, `v2/scripts/daily.sh`, `v2/scripts/gate_check.py` | 전체 stage 순서, run_id, output root, smoke/full gate |
| 2번 | 학습 실행 / 실험 관리 | `v2/pipeline/training_adapter.py`, `v2/runtime/experiment_core.py`, `v2/runtime/run_experiments.py`, `v2/runtime/utils.py`, `v2/runtime/requirements.txt`, `v2/configs/v2_15seed.json`의 training 관련 필드 | benchmark stage가 정규 output을 남기는지 |
| 3번 | 결과 분석 / 통계 해석 | `v2/pipeline/statistics.py`, `v2/pipeline/schema.py`의 benchmark/stat columns, `v2/outputs/experiments/v2_15seed/benchmark/*.csv`, `v2/docs/03_validation_and_statistics.md`, `v2/docs/07_output_and_report_contract.md` | aggregate stage가 report 입력 CSV를 만드는지 |
| 4번 | XAI 설명 / evidence bundle | `v2/pipeline/xai.py`, `v2/pipeline/xai_sampling.py`, `v2/pipeline/xai_bundle.py`, `v2/runtime/experiment_xai.py`, `v2/outputs/experiments/v2_15seed/xai/`, `v2/docs/04_xai_protocol.md`, `v2/docs/08_xai_report_template.md` | xai-bundle stage가 report/dashboard 입력 JSON을 만드는지 |
| 5번 | 발표자료 / 최종 보고서 제작 | `v2/pipeline/reporting.py`, `v2/runtime/dashboard_app.py`, `v2/runtime/experiment_dashboard.py`, `v2/outputs/experiments/v2_15seed/reports/`, `v2/outputs/experiments/v2_15seed/dashboard/`, `v2/docs/v2_end_to_end_team_brief.docx`, 발표자료 초안 | report/dashboard stage가 최종 산출물을 만드는지 |

---

## 3. 1번의 범위: 전체 코드리뷰가 아니라 Gate 총괄

1번이 직접 깊게 볼 파일:

```text
v2/run.sh
v2/pipeline/cli.py
v2/pipeline/runner.py
v2/pipeline/manifest.py
v2/pipeline/paths.py
v2/pipeline/artifacts.py
v2/configs/v2_15seed.json
v2/scripts/daily.sh
v2/scripts/gate_check.py
```

1번이 하지 않는 것:

```text
v2/runtime/experiment_core.py 내부 학습 로직 세부 리뷰
v2/pipeline/statistics.py 통계 수식 세부 리뷰
v2/pipeline/xai.py SHAP/LIME 세부 리뷰
v2/pipeline/reporting.py 발표 문장 세부 작성
```

1번의 판단 질문:

```text
./v2/run.sh가 어느 위치에서 실행돼도 v2 root로 들어가는가?
run_id가 모든 stage에서 v2_15seed로 유지되는가?
benchmark -> aggregate -> xai-bundle -> report -> dashboard가 같은 output root를 보는가?
failed_runs.csv / completed_runs.csv / execution_status.csv가 gate 판단에 충분한가?
full 120 run을 시작해도 되는 smoke 근거가 있는가?
```

---

## 4. 각 담당자의 1차 리뷰 기준

### 2번 학습 실행 / 실험 관리

1차 리뷰 파일:

```text
v2/pipeline/training_adapter.py
v2/runtime/experiment_core.py
v2/runtime/run_experiments.py
v2/runtime/utils.py
v2/runtime/requirements.txt
v2/configs/v2_15seed.json
```

확인할 산출물:

```text
v2/outputs/experiments/v2_15seed/benchmark/runs/<condition>/seed_<seed>/
v2/outputs/experiments/v2_15seed/benchmark/checkpoints/
v2/outputs/experiments/v2_15seed/completed_runs.csv
v2/outputs/experiments/v2_15seed/failed_runs.csv
```

### 3번 결과 분석 / 통계 해석

1차 리뷰 파일:

```text
v2/pipeline/statistics.py
v2/pipeline/schema.py
v2/docs/03_validation_and_statistics.md
v2/docs/07_output_and_report_contract.md
```

확인할 산출물:

```text
v2/outputs/experiments/v2_15seed/benchmark/benchmark_runs.csv
v2/outputs/experiments/v2_15seed/benchmark/benchmark_summary.csv
v2/outputs/experiments/v2_15seed/benchmark/paired_tests.csv
v2/outputs/experiments/v2_15seed/benchmark/paired_tests_holm.csv
v2/outputs/experiments/v2_15seed/benchmark/anova_*.csv
```

### 4번 XAI 설명 / evidence bundle

1차 리뷰 파일:

```text
v2/pipeline/xai.py
v2/pipeline/xai_sampling.py
v2/pipeline/xai_bundle.py
v2/runtime/experiment_xai.py
v2/docs/04_xai_protocol.md
v2/docs/08_xai_report_template.md
```

확인할 산출물:

```text
v2/outputs/experiments/v2_15seed/xai/primary/
v2/outputs/experiments/v2_15seed/xai/deep/
v2/outputs/experiments/v2_15seed/xai/ablation/
v2/outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json
v2/outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json
v2/outputs/experiments/v2_15seed/xai/evidence_bundle/token_attributions.jsonl
```

### 5번 발표자료 / 최종 보고서 제작

1차 리뷰 파일:

```text
v2/pipeline/reporting.py
v2/runtime/dashboard_app.py
v2/runtime/experiment_dashboard.py
v2/docs/08_xai_report_template.md
v2/docs/v2_end_to_end_team_brief.docx
v2/docs/role_guides/*.docx
```

확인할 산출물:

```text
v2/outputs/experiments/v2_15seed/reports/final_report.md
v2/outputs/experiments/v2_15seed/reports/final_report.docx
v2/outputs/experiments/v2_15seed/dashboard/index.html
발표자료/PPT 초안
발표 스크립트
```

---

## 5. 보고 양식

각 담당자는 아래 형식으로 보고한다.

```text
[v2 역할별 코드리뷰 보고]
담당 역할:
1차 리뷰한 폴더/파일:
실행한 명령:
확인한 산출물:
발견한 문제:
수정한 파일:
내 범위 밖으로 넘어가는 이슈:
1번 Gate 총괄에게 넘길 판단:
```

1번은 위 보고를 모아 아래처럼 판단한다.

```text
[Full Run Gate 판단]
2번 학습 smoke:
3번 aggregate/stat smoke:
4번 xai-bundle smoke:
5번 report/dashboard smoke:
남은 P0/P1 이슈:
GO/STOP:
```
