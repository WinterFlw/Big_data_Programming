# progress.md — 파이프라인 진행 상태

> 마지막 업데이트: 2026-05-03
> 이 파일은 하네스 시스템에서 **Planner 역할의 현재 위치 파악 문서**입니다.
> AI 에이전트는 작업 시작 전 반드시 이 파일을 읽고 현재 위치를 파악하세요.
> 작업 완료 후 Evaluator 검증을 통과하면, 이 파일을 업데이트합니다.

---

## 현재 위치 (2026-05-03)

명세서 v2.1 기준 코드 갱신 완료. 1차 파이프라인 산출물은 baseline 기록으로만 보유하며, v2 결과는 처음부터 재학습/재생성 필요.

- 단일 출처 명세서: `docs/파이프라인_명세서_v2.md`
- 척추 메시지: "단어 단서뿐 아니라 맥락 단서까지 함께 학습한 모델"이 베이스 대비 분류 성능과 판단 투명성에서 향상됨을 입증
- 베이스는 단어에 과의존(H1) → 개선 모델은 단어 신호 유지 + 맥락 단서 추가 학습(H3)

---

## v2.1 핵심 변경사항 (이전 (3) 초안 대비)

| # | 변경 항목 | 변경 내용 |
|---|---------|---------|
| 1 | Slur-Masking augmentation | 학습에서 제거 ([MASK] shortcut 위험·하드웨어·시간 부담) |
| 2 | Ablation 구조 | 4조건 → 8조건 (BERT × 4 + RoBERTa × 4) |
| 3 | Stratified split | label 단일, source-별 분리 보고로 보강 |
| 4 | XAI 4축 재구성 | Attribution / Faithfulness / Context Learning(자동) / Plausibility(보조) |
| 5 | 자동 메트릭 신규 | CI / IS / MSS / Attention Rollout (인간 라벨 의존 0) |
| 6 | α 그리드 조건 | B_B 조건에서 결정 후 D_B/B_R/D_R 동일 적용 |
| 7 | VADER 표현 | "Cheng 2022 선행연구 기반 사전 가설" 일관 |
| 8 | Confidence weighting | 사용 안 함 (agreement 0.814로 효과 미미) |
| 9 | L_target multi-task | D_B 부가 실험으로 제한 |
| 10 | 모델 입력 원칙 | 텍스트만 (메타는 학습 supervision으로만) |

---

## 1차 파이프라인 (baseline 보유, 2026-04-11 기준)

```
[완료] data ──→ [완료] vader ──→ [완료] eda ──→ [완료] tune ──→ [완료] benchmark
                                                                      │
                                                              [완료] freeze-study
                                                                      │
                                                              [완료] xai (+ Human Rationale)
                                                                      │
                                                              [완료] dashboard (18탭 + Playground)
```

- 1차 결과: Macro F1 0.6822, SHAP Overlap 0.688, LIME Overlap 0.6917
- 18탭 대시보드 운영 중
- v2 결과로 baseline 비교 후 갱신 예정

---

## v2 일정 (오늘 5/3 → 6/10)

```
[Phase 1: 4/30 ~ 5/10] (10일)
  - 데이터 파이프라인 갱신 (rationale 마스크 유지, augmentation 제거)
  - BERT-base baseline (A_B) 1회 학습 → sanity check
  - SHAP/LIME 초기 분석 → H1 진단 (단어 과의존 입증)

[Phase 2: 5/11 ~ 5/24] (14일)
  - α 그리드 서치 (B_B 조건)
  - BERT 4조건 (A_B/B_B/C_B/D_B) × 3 seed 학습
  - XAI 4축 분석 (BERT 4조건)

[Phase 3: 5/25 ~ 6/3] (10일)
  - RoBERTa 4조건 학습
  - XAI 4축 분석 (RoBERTa)
  - Freeze Study + L_target 부가 실험 (D_B with/without)

[Phase 4: 6/4 ~ 6/9] (6일)
  - 통합 분석 (Two-way ANOVA, 3-way ANOVA, paired t-test)
  - source-별 / target-별 subgroup 분리 보고
  - 발표 자료 (26p 평가 문서 톤) 통합
  - 리허설

[6/10] 최종 발표
```

총 추정 시간: 학습 30~60시간 + XAI 분석 60~90시간 = 약 90~150시간 (병렬 가능)

---

## 발표 자료 26p — 5인 분담 (확정)

| 팀 | 인원 | 페이지 | 분량 |
|----|------|------|----|
| 선행연구팀 | 차종민 | p1, p4, p5, p6 | 4p |
| 데이터팀 | 김정훈 | p7, p8 | 2p |
| 데이터팀 | 박종화 | p9~p11, p17, p18 | 5p |
| 파이프라인팀 | 정수현(팀장) | p2, p3, p12~p16, p20, p21 | 9p |
| 문서·통합팀 | 조은 | p19, p22~p26 | 6p |

작성 톤: 평가 문서 와꾸 (헤드라인 + 본문 8~15 bullet + 근거·수치 + 시사점). 페이지당 250~400자, 자체 완결적.

---

## 즉시 실행 항목 (Phase 1 시작 가이드)

### 1순위 (즉시 작성 가능, 외부 의존 없음)
- [ ] 차종민: p1, p4, p5, p6 (선행연구만 정리하면 끝) — Park(2018), Kennedy(2020), Dixon(2018), ElSherief(2021) 추가 인용
- [ ] 김정훈: p7, p8 (데이터·전처리 명세서)
- [ ] 박종화: p9, p10, p11 (EDA 결과 이미 있음)
- [ ] 정수현: p2, p3, p12~p16 (설계 텍스트만 작성)

### 2순위 (다른 팀 결과 일부 의존)
- [ ] 박종화: p17, p18 (평가지표 — 정수현 용어 통일 후)
- [ ] 조은: p23, p22, p25, p26 (반성적·일정 페이지)

### 3순위 (실험 결과 의존, 코드 v2 진행 후)
- [ ] 조은: p19, p24 (실험 결과 받아서 시각화)
- [ ] 정수현: p20, p21 (중간 분석 + 일정 진행 — 팀장 직접 관장)

### 코드 갱신 (Phase 1 즉시 시작)
- [x] data 파이프라인: Slur-Masking augmentation 제거, rationale 마스크 그대로 유지
- [x] experiment_core: 8조건 ablation + 공통 MLP head + Attn Loss + D_B target aux 부가 경로
- [x] experiment_xai: CI / IS / MSS / Attention Rollout 포함 v2.1 4축 사후 검증
- [x] status: v1 산출물을 v2 ready로 오판하지 않도록 stale 판정 추가
- [x] VADER feature 재생성 (`./run.sh vader --force`)
- [x] 구형 v1 checkpoint/model bundle/backup 및 stale benchmark/tuning/xai/dashboard 산출물 제거
- [x] dashboard Playground를 `models/` 고정 경로에서 `best_models.json` 기반 v2 checkpoint 로더로 전환
- [ ] α/β 그리드 서치 (`./run.sh tune --force`)
- [ ] 8조건 + D_B target aux 재학습 (`./run.sh benchmark`)
- [ ] v2.1 XAI 4축 재실행 (`./run.sh xai`)

---

## 알려진 이슈

| 이슈 | 심각도 | 상태 | 해결 방법 |
|------|:------:|:----:|---------|
| v1 결과(Macro F1 0.6822)와 v2 결과 비교 필요 | 중 | 진행 예정 | Phase 4에서 통합 보고 |
| (3) 학교 제출본은 v1 기준 (수정 불가) | 중 | 인지됨 | 발표 자료 p25에 "기존 계획 대비 조정 사항" 명시 |
| SHAP MPS 비호환 | 낮 | 코드에 반영됨 | CPU only 강제 |
| 3-seed 반복은 통계 검정력 낮음 | 낮 | 인지됨 | 한계점 서술, 향후 5~10 seed |
| `cat` → `bat` alias 문제 | 낮 | 우회 완료 | HEREDOC 금지, 직접 -m "..." |
| IS (Shapley interaction) 계산 부담 | 중 | 인지됨 | 50쌍 × 200 샘플로 축소 |
| L_target multi-task negative transfer 위험 | 중 | 인지됨 | D_B 부가 실험으로 분리, 결과 안 좋으면 한계로 솔직히 보고 |
| v1 checkpoint/benchmark/xai 산출물 | 높음 | stale | v2.1 구조와 달라 재학습·재생성 필요 |

---

## 운영 규칙 (CLAUDE.md 준수)

- 용어: "과학적 검증 프레임워크" / "전이학습 기반 full fine-tuning" / "VADER는 선행연구 기반 사전 가설" 일관 사용
- 금지 표현: "혐오는 단어가 아닌 맥락"(이분법 과장), "순환적 프레임워크", "피드백 루프", "XAI 진단 결과 기반 피처 설계"
- 정확한 톤: "단어 단서뿐 아니라 맥락 단서까지 함께 학습", "베이스의 단어 과의존 → 단어 신호 유지 + 맥락 추가 학습"
- 실행 진입점: `run.sh` → `run_experiments.py`만 사용
- Git: HEREDOC 커밋 금지, 한국어 커밋 (type: 제목)
- 6모듈 구조 유지: 새 모듈 생성 금지, experiment_xai.py에 함수 추가만

---

## 세이브 포인트

| 시점 | git commit | 설명 |
|------|-----------|------|
| 최신 | `69ef9f4` | rationale overlap 버그 수정 |
| rationale 구현 | `c8885d1` | Human Rationale 비교 + 대시보드 반영 |
| class weighting | `f032b60` | class weighting + label smoothing 적용 |
| Attention 수정 | `1017ddf` | Playground Heatmap return_dict=True |
| README 재작성 | `bcbc8a8` | Quick Start + 실험 결과 + 체크포인트 가이드 |
| 대시보드 확장 | `4d26a6c` | 6개 신규 기능 (Explorer, Comparison 등) |
| Playground XAI | `22c7355` | Attention Heatmap + LIME + MLP 분기 |
| 초기 커밋 | `4bed3a9` | 파이프라인 첫 커밋 |

> v2 코드 갱신 완료 후 새 커밋 시점 추가 예정 (예: "feat: v2.1 8조건 ablation + XAI 4축").
