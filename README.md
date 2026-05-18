# HateSpeachStudy

HateXplain 기반 혐오표현 탐지 연구 프로젝트입니다. 현재 저장소의 기준 작업 공간은 `v2/`이며, 과거 1차 파이프라인과 산출물은 `v1/`로 분리했습니다.

## 현재 상태

| 영역 | 위치 | 상태 |
|---|---|---|
| 현재 v2 작업 | `v2/` | 15 seed 실험, 통계 검증, XAI 보강, 서버 배치 실행을 위한 새 기준 폴더 |
| v2 실행 코드 | `v2/runtime/`, `v2/pipeline/` | 학습/추론/XAI runtime과 end-to-end orchestration 코드 |
| 과거 v1 기록 | `v1/` | 1차 baseline 코드, 대시보드, 기존 결과, 발표/문서 산출물 보관 |
| GitHub 첫 화면 | `README.md` | 현재 구조와 실행 진입점만 안내 |

루트에는 새 작업 기준만 남깁니다. 새 구현, 문서화, 팀원 분업, 서버 실행 계획은 `v2/`에서 진행합니다.

## v2 빠른 시작

```bash
git clone https://github.com/WinterFlw/Big_data_Programming.git
cd Big_data_Programming

./v2/run.sh e2e status --run-id v2_15seed
./v2/run.sh e2e plan --run-id v2_15seed --force
./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
```

현재 v2 코드는 실행 계획, manifest, 상태 파일, benchmark 실행 adapter, 통계/XAI/report 산출물 골격을 만드는 단계까지 준비되어 있습니다. 학습/추론/XAI 실행 코드는 `v2/runtime/`에 있고, orchestration은 `v2/pipeline/`에서 처리합니다. 특히 v2의 차별점은 단순 정확도 경쟁이 아니라, `xai-bundle` stage를 통해 SHAP/LIME/faithfulness/plausibility/context 결과를 하나의 **full XAI evidence bundle**로 묶어 report와 dashboard가 직접 읽게 만드는 데 있습니다. 다음 작업은 서버 또는 로컬 GPU 환경에서 `A_B seed 42` smoke 학습을 실제로 돌려 adapter와 산출물 계약을 검증하는 것입니다.

## v2 문서 읽는 순서

처음 합류한 팀원은 아래 순서로 읽으면 됩니다.

1. `v2/README.md`
2. `v2/docs/00_reading_order.md`
3. `v2/docs/01_model_definition.md`
4. `v2/docs/02_e2e_pipeline.md`
5. `v2/docs/03_validation_and_statistics.md`
6. `v2/docs/04_xai_protocol.md`
7. `v2/docs/06_execution_runbook.md`
8. `v2/docs/11_team_tasking_and_server_run_plan.md`
9. `v2/docs/14_team_assignment_matrix.md`
10. `v2/docs/15_runtime_code_validation_matrix.md`
11. `v2/docs/agent_tasks/README.md`
12. `v2/docs/agent_tasks/10_team_dispatch_prompts.md`
13. `v2/docs/AI_협업_도구_설치_및_사용_가이드.md`
14. `v2/docs/16_portable_ai_agent_skills_guide.md`
15. `v2/docs/17_korean_reading_file_index.md`
16. `v2/ai_skills/README.md`

## v2 실험 개요

v2 목표는 과거 baseline 결과를 그대로 확장하는 것이 아니라, 서버 실행을 전제로 새 end-to-end 실험을 재설계하는 것입니다.

| 항목 | 내용 |
|---|---|
| 데이터 | HateXplain 기반 3-class 혐오표현 탐지 |
| 반복 수 | 15 seeds |
| 조건 수 | 8개 ablation 조건 |
| 핵심 비교 | `A_B` baseline 대비 `D_B` 결합 모델 및 RoBERTa 계열 확장 |
| 통계 | paired test, Holm correction, confidence interval, effect size |
| XAI | SHAP, Integrated Gradients, LIME, attention rollout, case review, evidence bundle |
| 산출물 | `v2/outputs/experiments/v2_15seed/` |

조건명은 다음 구조를 따릅니다.

```text
A_B: BERT baseline
B_B: BERT + attention/rationale supervision
C_B: BERT + VADER/context feature
D_B: BERT + attention/rationale supervision + VADER/context feature
A_R/B_R/C_R/D_R: RoBERTa 계열 동일 ablation
```

## 팀원 분업

역할별 지시서는 `v2/docs/agent_tasks/`에 있습니다.
팀원별 기간과 코드 책임 범위는 `v2/docs/14_team_assignment_matrix.md`를 기준으로 합니다.
v2 내부 runtime code 검증 범위는 `v2/docs/15_runtime_code_validation_matrix.md`를 기준으로 합니다.

| 역할 | 시작 문서 |
|---|---|
| Benchmark 담당 | `v2/docs/agent_tasks/01_benchmark_agent.md` |
| Statistics 담당 | `v2/docs/agent_tasks/02_statistics_agent.md` |
| XAI 담당 | `v2/docs/agent_tasks/03_xai_agent.md` |
| XAI Evidence Bundle 담당 | `v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md` |
| Report/Dashboard 담당 | `v2/docs/agent_tasks/04_report_dashboard_agent.md` |
| QA/Server 담당 | `v2/docs/agent_tasks/05_qa_server_agent.md` |
| Integration Lead | `v2/docs/agent_tasks/06_integration_lead_agent.md` |
| Review 담당 | `v2/docs/agent_tasks/07_review_agent.md` |
| 팀장 하달문 | `v2/docs/agent_tasks/10_team_dispatch_prompts.md` |

AI 도구 설치와 사용법은 `v2/docs/AI_협업_도구_설치_및_사용_가이드.docx`와 Markdown 원본을 참고합니다.
Claude/Gemini/Cursor/Antigravity까지 공통으로 쓸 AI 작업 지시서는 `v2/CLAUDE.md`, `v2/GEMINI.md`, `v2/ai_skills/`를 참고합니다.

## v1 기록 확인

과거 1차 파이프라인은 `v1/`에 보관했습니다. 기존 대시보드를 확인해야 할 때만 아래처럼 들어갑니다.

```bash
cd v1
pip install -r requirements.txt
python3 dashboard_app.py
```

v1은 현재 연구의 기준 구현이 아닙니다. 새 실험, 새 문서, 새 팀 작업은 모두 `v2/`에서 진행합니다.

## 커밋 규칙

v2 작업은 커밋 메시지 훅을 사용합니다.

```bash
./v2/scripts/install_commit_msg_hook.sh
```

커밋 메시지는 아래 형식입니다.

```text
docs(v2): update experiment runbook

Why:
- ...

What:
- ...

Validation:
- ...
```

상세 규칙은 `v2/docs/13_commit_message_policy.md`를 따릅니다.
