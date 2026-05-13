# v2 Model End-to-End Redesign

> 목적: 기존 v2.1 실험을 보정하는 수준을 넘어서, **15 seed 반복 성능 검증 + XAI seed stability + 최종 보고/대시보드 출력**까지 포함하는 새 end-to-end 실험 라인을 정의한다.

---

## 1. 한 줄 요약

v2 모델은 HateXplain 텍스트만 입력으로 사용하고, `Rationale-aware Attention Loss`와 `VADER sentiment feature`를 통제된 8조건 ablation으로 검증하는 혐오표현 탐지 파이프라인이다. 새 실험 라인은 15개 random seed에서 같은 조건을 반복 실행하여 학습 stochasticity를 추정하고, XAI도 단일 checkpoint 해석이 아니라 seed 안정성까지 검증한다.

---

## 2. 이 폴더의 역할

이 폴더는 코드 구현 전 기준점이다. 다음 질문에 답한다.

1. 우리의 v2 모델은 정확히 무엇인가?
2. 기존 파이프라인에서 무엇을 바꿀 것인가?
3. 15 seed batch를 어떻게 실행하고 관리할 것인가?
4. 통계적으로 무엇을 검정할 것인가?
5. XAI는 무엇을 의미하며, 어떤 지표로 검증할 것인가?
6. 최종 산출물은 어디에 어떤 형식으로 저장할 것인가?

---

## 3. 먼저 읽을 문서

처음 들어오는 사람은 아래 순서로 읽는다.

```text
00_reading_order.md
01_model_definition.md
02_e2e_pipeline.md
03_validation_and_statistics.md
04_xai_protocol.md
06_execution_runbook.md
07_output_and_report_contract.md
10_code_implementation_notes.md
11_team_tasking_and_server_run_plan.md
```

빠르게 전체 방향만 잡을 때는 `00 -> 01 -> 02 -> 04`까지만 읽어도 된다.
실제 배치를 돌릴 사람은 `06_execution_runbook.md`를 반드시 같이 읽는다.
최종 보고서를 쓸 사람은 `07_output_and_report_contract.md`와 `08_xai_report_template.md`를 같이 본다.

---

## 4. 문서 구성

| 파일 | 내용 |
|---|---|
| `00_reading_order.md` | 읽는 순서, 독자별 경로, 핵심 질문, 용어 정리 |
| `01_model_definition.md` | v2 모델 정의, 8조건 ablation, 입력/손실/하이퍼파라미터 통제 |
| `02_e2e_pipeline.md` | 새 run_id 기반 end-to-end 파이프라인, batch 실행 구조, 산출물 트리 |
| `03_validation_and_statistics.md` | 15 seed 통계 설계, paired test, ANOVA, CI, bootstrap, 성공 기준 |
| `04_xai_protocol.md` | XAI 정의, Primary/Deep/Ablation/Case XAI, seed stability, 샘플링 원칙 |
| `05_improvements_and_open_checks.md` | 개선사항, 바꿀 것, 검증할 것, 리스크와 완료 기준 |
| `06_execution_runbook.md` | 실제 실행 순서, preflight, resume, 실패 복구, 완료 판정 |
| `07_output_and_report_contract.md` | 최종 산출물 구조, 파일 스키마, 보고서/대시보드 계약 |
| `08_xai_report_template.md` | XAI 결과를 논문/보고서에 넣는 표와 문장 템플릿 |
| `09_reference_map.md` | 기존 문서와 v2 문서의 연결 관계, 무엇을 canonical로 볼지 |
| `10_code_implementation_notes.md` | v2 코드 골격, 파일별 책임, 분업 가이드 |
| `11_team_tasking_and_server_run_plan.md` | 팀 업무하달, 서버 실행 전후 체크리스트, 제한된 GPU 기회 운영계획 |
| `12_code_commenting_guide.md` | v2 코드 주석 기준과 에이전트 주석 지시 |
| `13_commit_message_policy.md` | 엄격한 커밋 메시지 형식과 hook 적용법 |
| `agent_tasks/` | 팀원이 에이전트에게 줄 역할별 지시서와 인수인계 템플릿 |
| `manifest_template.json` | 새 실험 라인의 실행 설정 템플릿 |

---

## 5. 최종 실행 목표

새 실험 라인의 기본 실행 목표는 다음과 같다.

```text
Benchmark:
8 conditions x 15 seeds = 120 transformer training runs

Freeze Study:
2 variants x 15 seeds = 30 runs, optional

Primary XAI:
A_B vs D_B x 15 seeds x 200 stratified samples

Deep XAI:
A_B vs D_B x median-performing seed x 500 stratified samples

Ablation XAI:
8 conditions x median-performing seed x 50 samples

Final Output:
Markdown report + DOCX report + dashboard + machine-readable artifacts
```

---

## 6. 기본 run_id

기본 run_id는 다음으로 둔다.

```text
v2_15seed
```

모든 새 산출물은 기존 `outputs/reports`, `outputs/xai`를 바로 덮어쓰지 않고 아래 폴더에 저장한다.

```text
outputs/experiments/v2_15seed/
```

이렇게 분리해야 기존 3-seed 산출물, v1 baseline 산출물, 새 15-seed 산출물이 섞이지 않는다.

---

## 7. 핵심 설계 원칙

### 6.1 모델 입력 단일 소스

모델 입력은 텍스트다. `label`, `rationale`, `target`, `source`, `agreement`는 모델 입력으로 사용하지 않는다.

```text
post_tokens -> tokenizer -> BERT/RoBERTa
post_tokens -> VADER -> 4d sentiment feature, C/D 조건만
```

### 6.2 Ablation 변수 통제

조건별로 다른 learning rate/dropout을 쓰지 않는다. family별 공통 hyperparameter를 사용한다.

```text
BERT family: A_B, B_B, C_B, D_B 모두 동일 hyperparameter
RoBERTa family: A_R, B_R, C_R, D_R 모두 동일 hyperparameter
```

### 6.3 동일 seed paired design

같은 seed를 모든 조건에 공통으로 물린다. 조건 간 비교는 동일 seed 내 paired difference로 계산한다.

### 6.4 XAI seed stability

XAI는 한 checkpoint의 그림 몇 장이 아니라, 같은 sample set을 여러 seed checkpoint에 넣었을 때 설명 지표가 유지되는지 검증한다.

---

## 8. 다음 구현 작업

1. `run.sh`에 `e2e` 명령 추가
2. `run_experiments.py`에 `e2e` subcommand 추가
3. run_id 기반 output root 도입
4. batch benchmark runner 구현
5. 15 seed 통계 집계 함수 추가
6. XAI primary/deep/ablation 실행 함수 분리
7. 새 report/dashboard generator 연결
