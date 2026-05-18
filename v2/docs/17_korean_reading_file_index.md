# 17. 한글 문서 읽기 목록

> 목적: 팀원이 프로젝트에 들어왔을 때 어떤 한글 문서부터 읽어야 하는지 명확히 정한다. 이 문서는 `v2/` 기준 필독 문서와 `v1/` archive 참고 문서를 분리한다.

---

## 1. 먼저 알아둘 기준

```text
v2/ 문서 = 현재 작업 기준
v1/ 문서 = 과거 자료, 발표/설계 참고용
```

팀원이 실제 구현, 검증, 서버 실행을 맡는다면 반드시 `v2/` 문서를 먼저 읽는다. `v1/` 문서는 배경 이해나 과거 논리 확인이 필요할 때만 본다.

업무 도입 가능성 점검 결과:

```text
이 문서는 "무엇을 읽을지"를 알려주는 인덱스로는 충분하다.
다만 팀원이 바로 업무에 들어가려면 역할별 첫 행동과 검증 명령까지 한 장에 있어야 한다.
따라서 업무 하달 시에는 이 문서와 함께 한글_필독문서_업무도입_가이드.docx를 같이 공유한다.
```

---

## 2. 30분 안에 읽을 필수 한글 문서

처음 합류한 팀원은 아래 순서로 읽는다.

| 순서 | 파일 | 읽는 이유 |
|---:|---|---|
| 1 | `v2/README.md` | v2가 현재 기준 작업공간이라는 점과 빠른 실행법 확인 |
| 2 | `v2/docs/00_reading_order.md` | 전체 문서 묶음의 지도 |
| 3 | `v2/docs/01_model_definition.md` | v2 모델, 8조건 ablation, 입력/손실 구조 이해 |
| 4 | `v2/docs/02_e2e_pipeline.md` | benchmark부터 dashboard까지 stage 흐름 이해 |
| 5 | `v2/docs/14_team_assignment_matrix.md` | 팀원 5명 역할과 책임 범위 확인 |
| 6 | `v2/docs/15_runtime_code_validation_matrix.md` | 전체 코드를 전수 검토하지 않고 critical path를 검증하는 법 |
| 7 | `v2/docs/20_role_file_review_matrix.md` | 역할별 1차 코드리뷰 폴더/파일 경계 확인 |
| 8 | `v2/docs/v2_end_to_end_team_brief.docx` | 팀 공유용 Word 브리프. 회의 전 빠르게 읽기 좋음 |

이 7개만 읽어도 아래 질문에 답할 수 있어야 한다.

```text
v2 모델은 무엇인가?
왜 15 seed인가?
어떤 산출물이 어디에 쌓이는가?
내 역할은 어떤 파일을 책임지는가?
full benchmark를 시작하기 전 무엇을 검증해야 하는가?
```

---

## 3. 실제 실행 담당자가 읽을 한글 문서

서버에서 batch를 돌리거나 smoke/full run을 관리하는 사람은 아래를 읽는다.

| 파일 | 핵심 내용 |
|---|---|
| `v2/docs/06_execution_runbook.md` | preflight, dry-run, smoke, full run, 실패 복구 |
| `v2/docs/11_team_tasking_and_server_run_plan.md` | 제한된 서버 기회에서 누가 무엇을 언제 돌릴지 |
| `v2/docs/15_runtime_code_validation_matrix.md` | runtime/adapter/statistics/XAI/report 검증 분담 |
| `v2/docs/20_role_file_review_matrix.md` | 각자 1차 리뷰할 폴더/파일과 1번 Gate 총괄 범위 |
| `v2/docs/07_output_and_report_contract.md` | 산출물 위치와 CSV/JSON 계약 |
| `v2/docs/agent_tasks/10_team_dispatch_prompts.md` | 팀장 업무 하달 문장 |

특히 서버 실행 전에는 아래 문장을 기준으로 판단한다.

```text
A_B seed 42 smoke가 안 끝났으면 full 120 benchmark를 시작하지 않는다.
```

---

## 4. 코드 구현 담당자가 읽을 한글 문서

| 역할 | 먼저 읽을 문서 |
|---|---|
| 학습/benchmark | `v2/docs/10_code_implementation_notes.md`, `v2/docs/15_runtime_code_validation_matrix.md`, `v2/docs/agent_tasks/01_benchmark_agent.md` |
| 통계/aggregate | `v2/docs/03_validation_and_statistics.md`, `v2/docs/07_output_and_report_contract.md`, `v2/docs/agent_tasks/02_statistics_agent.md` |
| XAI | `v2/docs/04_xai_protocol.md`, `v2/docs/08_xai_report_template.md`, `v2/docs/agent_tasks/03_xai_agent.md` |
| XAI evidence bundle | `v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md`, `v2/docs/07_output_and_report_contract.md` |
| report/dashboard | `v2/docs/08_xai_report_template.md`, `v2/docs/07_output_and_report_contract.md`, `v2/docs/agent_tasks/04_report_dashboard_agent.md` |
| 리뷰/preflight | `v2/docs/15_runtime_code_validation_matrix.md`, `v2/docs/agent_tasks/07_review_agent.md`, `v2/ai_skills/hatespeech-v2-review/SKILL.md` |

---

## 5. AI 도구를 쓰는 팀원이 읽을 한글 문서

Claude, Gemini, Cursor, Antigravity, Codex를 쓰는 팀원은 아래 순서로 읽힌다.

| 순서 | 파일 | 용도 |
|---:|---|---|
| 1 | `v2/docs/AI_협업_도구_설치_및_사용_가이드.md` | AI CLI/IDE 설치와 사용 기준 |
| 2 | `v2/docs/16_portable_ai_agent_skills_guide.md` | Claude/Gemini/Cursor/Antigravity 공용 Skill 사용법 |
| 3 | `v2/ai_skills/README.md` | 역할별 portable skill 선택 |
| 4 | `v2/ai_skills/common_project_rules.md` | 모든 AI가 따라야 할 v2 공통 규칙 |
| 5 | `v2/ai_skills/<역할>/SKILL.md` | 담당 역할별 파일 범위와 검증 명령 |
| 6 | `v2/CLAUDE.md` 또는 `v2/GEMINI.md` | Claude/Gemini용 프로젝트 진입 규칙 |

팀원에게는 아래처럼 말하면 된다.

```text
AI 도구를 쓰기 전에 v2/docs/16_portable_ai_agent_skills_guide.md를 먼저 읽고,
자기 역할에 맞는 v2/ai_skills/hatespeech-v2-*/SKILL.md를 AI에게 읽혀라.
```

---

## 6. 보고서/발표 담당자가 읽을 한글 문서

| 파일 | 용도 |
|---|---|
| `v2/docs/08_xai_report_template.md` | XAI 결과를 보고서에 어떻게 쓸지 |
| `v2/docs/07_output_and_report_contract.md` | final_report, dashboard가 읽어야 할 산출물 계약 |
| `v2/docs/03_validation_and_statistics.md` | p-value, CI, effect size 해석 기준 |
| `v2/docs/04_xai_protocol.md` | XAI를 사후 검증으로 해석하는 기준 |
| `v2/docs/v2_end_to_end_team_brief.docx` | 회의/팀 공유용 요약 |

주의:

```text
p-value만으로 결론을 쓰지 않는다.
XAI case 몇 개를 성능 개선의 인과 증명처럼 쓰지 않는다.
xai_claims.json과 xai_dashboard_bundle.json을 우선 근거로 삼는다.
```

---

## 7. v1 archive에서 참고할 만한 한글 문서

아래 문서는 현재 실행 기준이 아니다. 다만 연구 배경, 발표 문장, 과거 설계 논리를 확인할 때 참고할 수 있다.

| 파일 | 참고 용도 |
|---|---|
| `v1/docs/seed15_XAI_통계_실험계획.md` | 15 seed와 XAI 통계 설계의 초기 논리 |
| `v1/docs/파이프라인_검증_및_팀분업_가이드.md` | 과거 파이프라인 검증과 팀 분업 아이디어 |
| `v1/docs/엔드투엔드_설명출력_가이드.md` | end-to-end 출력 설명 방식 참고 |
| `v1/docs/파이프라인_전체_정리.md` | 과거 전체 파이프라인 구조 파악 |
| `v1/docs/파이프라인_상세분석.md` | 이전 코드/산출물 분석 참고 |
| `v1/docs/맥락_개선_설계.md` | 맥락 개선 아이디어의 초기 설계 |
| `v1/docs/XAI_3축_검증_프레임워크_제안서.md` | XAI 검증 축 아이디어 |
| `v1/docs/참고자료_종합_가이드.md` | 기존 참고 자료 묶음 |
| `v1/docs/발표_와꾸_v2.md` | 발표 구조 참고 |
| `v1/docs/표절검증_최종보고서_v2.md` | 제출물 문장/검증 방식 참고 |

v1 문서를 볼 때 지킬 원칙:

```text
아이디어는 참고한다.
실행 경로는 따라 하지 않는다.
산출물 위치는 v2 기준으로 다시 해석한다.
```

---

## 8. Word 파일로 읽을 문서

| 파일 | 언제 읽는가 |
|---|---|
| `v2/docs/v2_end_to_end_team_brief.docx` | 팀 전체 브리핑, 회의 전 공유 |
| `v2/docs/한글_필독문서_업무도입_가이드.docx` | 한글 문서 목록을 업무 행동으로 연결할 때. 역할별 첫 업무와 검증 명령 포함 |
| `v2/docs/AI_협업_도구_설치_및_사용_가이드.docx` | 팀원이 AI 도구 설치/사용법을 확인할 때 |
| `v2/docs/role_guides/01_code_review_pipeline_validation.docx` | 1번 코드 리뷰/파이프라인 검증 담당에게 배포 |
| `v2/docs/role_guides/02_training_execution_experiment_management.docx` | 2번 학습 실행/실험 관리 담당에게 배포 |
| `v2/docs/role_guides/03_result_analysis_statistics.docx` | 3번 결과 분석/통계 해석 담당에게 배포. paired t-test 중심, Holm은 보조 설명 |
| `v2/docs/role_guides/04_xai_explanation_evidence_bundle.docx` | 4번 XAI 설명/evidence bundle 담당에게 배포 |
| `v2/docs/role_guides/05_presentation_report_final_integration.docx` | 5번 발표자료/최종 보고서 담당에게 배포 |
| `v2/docs/20_role_file_review_matrix.md` | 팀장/1번이 코드리뷰 독박을 막고 파일 책임을 나눌 때 |
| `v2/outputs/experiments/v2_15seed/reports/final_report.docx` | report stage 결과 확인용. 아직 실제 full 결과 기반 최종본은 아님 |
| `v1/docs/seed15_XAI_통계_실험계획.docx` | 과거 15 seed/XAI 계획 참고 |
| `v1/docs/맥락이해_기반_혐오표현탐지_개선_제안서.docx` | 연구 제안서 문장 참고 |
| `v1/XAI_12지표_상세해설.docx` | XAI 지표 설명 참고 |
| `v1/v2.1_상세명세서.docx` | 과거 v2.1 상세 명세 참고 |

---

## 9. 읽지 않아도 되는 문서

처음 합류한 팀원이 바로 읽지 않아도 되는 것:

```text
v1/outputs/reports/*.md
v1/outputs/xai/*.md
v1/outputs/tuning/*.md
v1/progress.md
```

이 파일들은 과거 실행 결과나 진행 기록이다. 현재 v2 구현/검증에는 직접 필요하지 않다.

---

## 10. 최종 추천 읽기 코스

### 팀원 공통 1시간 코스

```text
v2/README.md
v2/docs/00_reading_order.md
v2/docs/01_model_definition.md
v2/docs/02_e2e_pipeline.md
v2/docs/14_team_assignment_matrix.md
v2/docs/15_runtime_code_validation_matrix.md
v2/docs/v2_end_to_end_team_brief.docx
v2/docs/한글_필독문서_업무도입_가이드.docx
```

### 서버 실행 담당 1시간 코스

```text
v2/docs/06_execution_runbook.md
v2/docs/11_team_tasking_and_server_run_plan.md
v2/docs/15_runtime_code_validation_matrix.md
v2/docs/07_output_and_report_contract.md
v2/ai_skills/hatespeech-v2-review/SKILL.md
```

### AI 도구 사용자 30분 코스

```text
v2/docs/AI_협업_도구_설치_및_사용_가이드.md
v2/docs/16_portable_ai_agent_skills_guide.md
v2/ai_skills/README.md
v2/ai_skills/common_project_rules.md
v2/ai_skills/<자기 역할>/SKILL.md
```

### 발표/보고서 담당 1시간 코스

```text
v2/docs/03_validation_and_statistics.md
v2/docs/04_xai_protocol.md
v2/docs/07_output_and_report_contract.md
v2/docs/08_xai_report_template.md
v1/docs/발표_와꾸_v2.md
v1/docs/참고자료_종합_가이드.md
```
