# 00. Reading Order and Document Map

> 목적: v2 모델 문서 묶음을 처음 보는 사람이 어떤 순서로 읽어야 하는지 정한다. 이 문서는 연구 설계, 코드 구현, 배치 실행, 최종 보고서 작성의 입구 역할을 한다.

---

## 1. 가장 먼저 읽을 순서

기본 순서는 아래와 같다.

```text
0. README.md
1. 00_reading_order.md
2. 01_model_definition.md
3. 02_e2e_pipeline.md
4. 03_validation_and_statistics.md
5. 04_xai_protocol.md
6. 05_improvements_and_open_checks.md
7. 06_execution_runbook.md
8. 07_output_and_report_contract.md
9. 08_xai_report_template.md
10. 09_reference_map.md
11. 10_code_implementation_notes.md
12. 11_team_tasking_and_server_run_plan.md
13. 12_code_commenting_guide.md
14. 13_commit_message_policy.md
15. 14_team_assignment_matrix.md
16. 15_runtime_code_validation_matrix.md
17. 16_portable_ai_agent_skills_guide.md
18. 17_korean_reading_file_index.md
19. 20_role_file_review_matrix.md
20. ../ai_skills/README.md
21. ../ai_skills/common_project_rules.md
22. agent_tasks/README.md
23. agent_tasks/09_e2e_xai_evidence_bundle_agent.md
```

이 순서의 의도:

```text
무엇을 하는가 -> 어떻게 돌리는가 -> 어떻게 검증하는가 -> 어떻게 설명하는가 -> 무엇을 산출하는가
```

---

## 2. 30분 안에 방향만 잡는 경로

발표 준비, 회의 전 브리핑, 지도교수 설명 전에는 아래만 읽는다.

```text
README.md
17_korean_reading_file_index.md
01_model_definition.md
02_e2e_pipeline.md
04_xai_protocol.md
```

읽고 답할 수 있어야 하는 질문:

```text
우리 v2 모델은 무엇인가?
왜 8조건 ablation인가?
왜 15 seed인가?
XAI는 모델 설계 도구인가, 사후 검증 도구인가?
최종 결과는 어디에 쌓이는가?
```

---

## 3. 실제 코드를 짤 때 읽는 경로

구현자는 아래 순서로 읽는다.

```text
01_model_definition.md
02_e2e_pipeline.md
05_improvements_and_open_checks.md
06_execution_runbook.md
07_output_and_report_contract.md
manifest_template.json
```

구현자가 특히 확인할 것:

```text
run_id 기반 output root가 적용되는가?
condition x seed 단위로 resume 가능한가?
aggregate stage가 학습 없이 독립 실행 가능한가?
XAI sample set이 seed마다 동일하게 고정되는가?
최종 report/dashboard가 run_id 내부 산출물만 읽는가?
```

---

## 4. 통계 검증 담당자가 읽는 경로

성능 비교와 p-value를 정리할 때는 아래 순서로 읽는다.

```text
01_model_definition.md
03_validation_and_statistics.md
07_output_and_report_contract.md
```

핵심 검정 구조:

```text
같은 seed 안에서 A_B와 D_B를 비교한다.
조건 차이는 paired difference로 본다.
평균 차이만 보지 않고 CI와 effect size를 같이 본다.
Holm 보정과 ANOVA는 여러 조건을 볼 때의 보조/부록 분석으로 둔다.
```

최소 보고 단위:

```text
mean
std
95% CI
paired p-value
Cohen's dz or paired effect size
supplementary adjusted p-value when many comparisons are shown
```

---

## 5. XAI 담당자가 읽는 경로

XAI 분석자는 아래 순서로 읽는다.

```text
04_xai_protocol.md
08_xai_report_template.md
07_output_and_report_contract.md
agent_tasks/09_e2e_xai_evidence_bundle_agent.md
```

XAI에서 가장 중요한 원칙:

```text
XAI는 사후 검증이다.
XAI는 성능 개선 원인의 절대 증명이 아니다.
XAI는 모델 판단 패턴이 선행 가설과 일관적인지 확인하는 보조 근거다.
TF-IDF 대비 강점은 full XAI evidence bundle로 방어한다.
```

확인해야 할 질문:

```text
중요 토큰이 인간 rationale과 겹치는가?
중요 토큰 제거 시 예측 확률이 흔들리는가?
seed가 달라도 설명 패턴이 유지되는가?
오류 개선 사례가 전체 통계 결론으로 과대 해석되지 않는가?
```

---

## 6. 배치 실행 담당자가 읽는 경로

실제로 GPU에서 돌릴 사람은 아래 순서로 읽는다.

```text
02_e2e_pipeline.md
06_execution_runbook.md
manifest_template.json
07_output_and_report_contract.md
```

실행 전 확인:

```text
데이터 split이 고정되어 있는가?
15개 seed가 manifest에 기록되어 있는가?
8개 condition이 모두 들어 있는가?
output root가 outputs/experiments/v2_15seed/인가?
resume 시 완료된 run을 건너뛰는가?
실패 run만 재시작할 수 있는가?
```

---

## 7. 최종 보고서 작성자가 읽는 경로

보고서 작성자는 아래 순서로 읽는다.

```text
01_model_definition.md
03_validation_and_statistics.md
04_xai_protocol.md
07_output_and_report_contract.md
08_xai_report_template.md
09_reference_map.md
```

보고서에 반드시 들어갈 내용:

```text
모델 정의
실험 조건표
15 seed 반복 이유
성능 요약표
핵심 paired test 결과와 보조 adjusted p-value
XAI primary 결과
XAI seed stability 결과
한계와 threat to validity
재현 가능한 실행 경로
```

---

## 8. 팀장/서버 실행 담당자가 읽는 경로

팀장, 서버 예약 담당자, 최종 실행 책임자는 아래 순서로 읽는다.

```text
02_e2e_pipeline.md
06_execution_runbook.md
10_code_implementation_notes.md
11_team_tasking_and_server_run_plan.md
14_team_assignment_matrix.md
15_runtime_code_validation_matrix.md
20_role_file_review_matrix.md
role_guides/README.md
07_output_and_report_contract.md
```

읽고 바로 결정해야 하는 것:

```text
누가 코드 리뷰/파이프라인 검증을 맡는가?
누가 학습 실행/실험 관리를 맡는가?
누가 결과 분석/통계 해석을 맡는가?
누가 XAI 설명/evidence bundle을 맡는가?
누가 발표자료/최종 보고서 제작을 맡는가?
서버에 올리기 전 smoke test 통과 기준은 무엇인가?
서버에서 전체 실행을 시작해도 되는 중단/진행 기준은 무엇인가?
실패 run을 누가 확인하고 재실행하는가?
```

---

## 9. 에이전트를 쓰는 팀원이 읽는 경로

팀원이 개인 에이전트를 붙여 작업할 때는 아래 순서로 읽는다.

```text
../CLAUDE.md 또는 ../GEMINI.md
17_korean_reading_file_index.md
16_portable_ai_agent_skills_guide.md
../ai_skills/README.md
../ai_skills/common_project_rules.md
../ai_skills/<자기 역할>/SKILL.md
agent_tasks/README.md
agent_tasks/00_common_agent_rules.md
agent_tasks/<자기 역할 문서>.md
agent_tasks/10_team_dispatch_prompts.md
agent_tasks/08_handoff_template.md
12_code_commenting_guide.md
13_commit_message_policy.md
```

역할 문서 선택:

```text
Benchmark: agent_tasks/01_benchmark_agent.md
Statistics: agent_tasks/02_statistics_agent.md
XAI: agent_tasks/03_xai_agent.md
Report/Dashboard: agent_tasks/04_report_dashboard_agent.md
QA/Server: agent_tasks/05_qa_server_agent.md
Integration: agent_tasks/06_integration_lead_agent.md
Review: agent_tasks/07_review_agent.md
Dispatch Prompts: agent_tasks/10_team_dispatch_prompts.md
```

---

## 10. 용어 빠른 정리

| 용어 | 의미 |
|---|---|
| `condition` | ablation 실험의 한 조건. 예: A_B, D_B |
| `seed` | 학습 난수 초기값. v2 기본은 15개 |
| `paired design` | 같은 seed에서 조건을 비교하는 설계 |
| `run_id` | 실험 묶음 식별자. 기본값은 `v2_15seed` |
| `manifest` | 실험 설정과 실행 계획을 기록하는 파일 |
| `aggregate` | 개별 run 결과를 모아 통계표를 만드는 단계 |
| `Primary XAI` | 핵심 모델 비교를 15 seed 전체에서 수행하는 XAI |
| `Deep XAI` | median seed에서 더 많은 샘플을 자세히 분석하는 XAI |
| `Ablation XAI` | 8조건 전체를 가볍게 비교하는 XAI |
| `xai-bundle` | XAI raw artifact를 report/dashboard용 evidence bundle로 통합하는 stage |
| `seed stability` | seed가 바뀌어도 설명 패턴이 유지되는지 보는 지표 |

---

## 11. 이 문서 묶음의 기준 문장

v2 실험의 기준 문장은 다음이다.

```text
본 연구의 v2 파이프라인은 HateXplain 기반 혐오표현 분류에서 rationale-aware attention loss와 sentiment feature의 효과를 8조건 ablation 및 15 seed 반복 실험으로 검증하고, XAI를 통해 예측 성능뿐 아니라 인간 rationale 정렬성, faithfulness, seed-level explanation stability를 사후 분석한다.
```
