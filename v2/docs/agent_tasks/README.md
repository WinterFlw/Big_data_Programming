# Agent Task Pack for v2 Pipeline

> 목적: 팀원이 각자 AI 에이전트를 사용해도 작업 범위가 충돌하지 않도록, 역할별 지시문과 완료 기준을 문서화한다.

---

## 1. 이 폴더의 역할

이 폴더는 팀원이 에이전트에게 그대로 전달할 수 있는 작업 지시서 묶음이다.

핵심 목표:

```text
역할별 책임 범위를 명확히 한다.
각 에이전트가 읽어야 할 문서를 제한한다.
각 에이전트의 수정 가능 파일을 제한한다.
완료 기준과 검증 명령어를 고정한다.
서버 실행 기회가 제한된 상황에서 무리한 full run을 방지한다.
```

---

## 2. 사용 순서

모든 팀원은 먼저 아래 공통 문서를 읽는다.

```text
00_common_agent_rules.md
```

그 다음 자기 역할에 맞는 문서를 하나 고른다.

```text
01_benchmark_agent.md
02_statistics_agent.md
03_xai_agent.md
04_report_dashboard_agent.md
05_qa_server_agent.md
06_integration_lead_agent.md
07_review_agent.md
08_handoff_template.md
```

---

## 3. 역할별 선택 가이드

| 상황 | 읽을 문서 |
|---|---|
| 실제 학습 실행 연결 | `01_benchmark_agent.md` |
| p-value, CI, effect size 구현 | `02_statistics_agent.md` |
| SHAP/LIME/XAI stability 구현 | `03_xai_agent.md` |
| 최종 report/dashboard 생성 | `04_report_dashboard_agent.md` |
| 서버 실행 전후 검증 | `05_qa_server_agent.md` |
| 여러 팀원 작업 통합 | `06_integration_lead_agent.md` |
| 다른 팀원 코드 검토 | `07_review_agent.md` |
| 작업 완료 보고 | `08_handoff_template.md` |

---

## 4. 에이전트에게 주는 기본 문장

각 팀원은 에이전트에게 아래 문장을 먼저 준다.

```text
우리는 HateSpeachStudy의 v2_15seed end-to-end 파이프라인을 구현 중입니다.
기존 v2.1 결과를 덮어쓰지 않고, 모든 새 산출물은 outputs/experiments/v2_15seed/ 아래에 저장해야 합니다.
내 역할은 아래 역할 문서에 정의된 범위만 담당하는 것입니다.
역할 밖 파일을 수정해야 한다면 먼저 이유를 설명하고, 최소 변경으로 제안해주세요.
```

그 다음 자기 역할 문서 전체를 붙인다.

---

## 5. 공통 금지 사항

```text
기존 outputs/reports, outputs/xai, outputs/runs를 canonical output으로 쓰지 않는다.
대용량 full benchmark를 임의 실행하지 않는다.
다른 담당자의 파일을 이유 없이 수정하지 않는다.
통계 결과가 없는데 보고서 문장을 확정적으로 쓰지 않는다.
XAI를 성능 개선의 인과적 증명처럼 표현하지 않는다.
```

