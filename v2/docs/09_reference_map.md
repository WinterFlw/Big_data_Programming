# 09. Reference Map

> 목적: 기존 문서와 새 v2 문서의 관계를 정리한다. 여러 문서가 동시에 존재할 때 어떤 문서를 기준으로 볼지 혼란을 줄이기 위한 지도다.

---

## 1. Canonical 문서

새 v2 실험 설계의 기준 문서는 이 폴더다.

```text
docs/
```

특히 아래 파일을 canonical로 본다.

```text
README.md
01_model_definition.md
02_e2e_pipeline.md
03_validation_and_statistics.md
04_xai_protocol.md
06_execution_runbook.md
07_output_and_report_contract.md
10_code_implementation_notes.md
11_team_tasking_and_server_run_plan.md
```

기존 문서와 내용이 충돌하면, 새 v2 재설계와 배치 실행에 대해서는 `docs/`의 내용을 우선한다.

---

## 2. 기존 문서와의 관계

기존 문서들은 배경과 이전 결정 사항을 이해하는 데 쓴다.

| 문서 | 역할 | v2에서의 사용 방식 |
|---|---|---|
| `docs/파이프라인_명세서_v2.md` | 기존 파이프라인 명세 | 기존 구현 이해용 |
| `docs/엔드투엔드_설명출력_가이드.md` | 기존 end-to-end 출력 가이드 | v2 출력 설계 참고용 |
| `docs/seed15_XAI_통계_실험계획.md` | 15 seed와 XAI 통계 계획 | v2 통계/XAI 설계의 선행 메모 |
| `README.md` | 프로젝트 전체 README | 실행 진입점과 프로젝트 개요 확인 |
| `progress.md` | 진행상황 기록 | 작업 히스토리 확인 |

---

## 3. 문서별 책임 분리

문서 책임은 아래처럼 나눈다.

```text
README.md:
  프로젝트 전체 진입점

docs/README.md:
  v2 실험 설계 진입점

docs/00_reading_order.md:
  읽는 순서와 독자별 경로

docs/01_model_definition.md:
  모델 정의와 ablation 설계

docs/02_e2e_pipeline.md:
  파이프라인 stage와 산출물 위치

docs/03_validation_and_statistics.md:
  통계 검정과 seed 반복 근거

docs/04_xai_protocol.md:
  XAI 분석 원칙과 지표

docs/05_improvements_and_open_checks.md:
  구현 전 개선사항과 검증할 것

docs/06_execution_runbook.md:
  실제 배치 실행 절차

docs/07_output_and_report_contract.md:
  파일 스키마와 보고서 계약

docs/08_xai_report_template.md:
  XAI 보고서 작성 템플릿

docs/09_reference_map.md:
  기존 문서와 새 문서의 관계

docs/10_code_implementation_notes.md:
  v2 코드 골격과 파일별 구현 책임

docs/11_team_tasking_and_server_run_plan.md:
  팀 업무하달, 제한된 서버 실행 기회 운영, 실패 보고 양식

docs/agent_tasks/:
  팀원이 에이전트에게 전달할 역할별 작업 지시서
```

---

## 4. 구현 시 참조 순서

코드 구현을 시작할 때는 아래 순서로 참조한다.

```text
1. 02_e2e_pipeline.md
2. manifest_template.json
3. 07_output_and_report_contract.md
4. 06_execution_runbook.md
5. 05_improvements_and_open_checks.md
6. 10_code_implementation_notes.md
7. 11_team_tasking_and_server_run_plan.md
```

이 순서를 쓰는 이유:

```text
먼저 실행 구조를 정한다.
그 다음 설정값을 고정한다.
그 다음 산출물 스키마를 맞춘다.
그 다음 실제 실행/복구 절차를 구현한다.
마지막으로 빠진 검증 항목을 체크한다.
```

---

## 5. 보고서 작성 시 참조 순서

최종 보고서를 쓸 때는 아래 순서로 참조한다.

```text
1. 01_model_definition.md
2. 03_validation_and_statistics.md
3. 04_xai_protocol.md
4. 08_xai_report_template.md
5. 07_output_and_report_contract.md
```

보고서의 핵심 논리:

```text
모델 가설이 있다.
통제된 ablation으로 가설을 검증한다.
15 seed 반복으로 stochasticity를 반영한다.
paired statistical test로 조건 차이를 평가한다.
XAI로 판단 패턴과 rationale alignment를 사후 점검한다.
```

---

## 6. 앞으로 추가할 수 있는 문서

코드 구현 후 추가하면 좋은 문서:

```text
12_result_interpretation_guide.md
13_submission_checklist.md
```

`12_result_interpretation_guide.md`에는 실제 결과가 나온 뒤 어떤 결론을 낼 수 있는지 정리한다.
`13_submission_checklist.md`에는 제출 전 확인할 파일과 표/그림 누락 여부를 적는다.
