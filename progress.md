# progress.md — 파이프라인 진행 상태

> 마지막 업데이트: 2026-04-11
> 이 파일은 하네스 시스템에서 **Planner 역할의 현재 위치 파악 문서**입니다.
> AI 에이전트는 작업 시작 전 반드시 이 파일을 읽고 현재 위치를 파악하세요.
> 작업 완료 후 Evaluator 검증을 통과하면, 이 파일을 업데이트합니다.

---

## 파이프라인 완료 현황

```
[완료] data ──→ [완료] vader ──→ [완료] eda ──→ [완료] tune ──→ [완료] benchmark
                                                                      │
                                                              [완료] freeze-study
                                                                      │
                                                              [완료] xai (+ Human Rationale)
                                                                      │
                                                              [완료] dashboard (18탭 + Playground)
```

| 단계 | 상태 | 산출물 | 비고 |
|------|:----:|--------|------|
| data | 완료 | `data_splits.pkl` (3.9MB) | 13,433건, 70/10/20 stratified split |
| vader | 완료 | `vader_features.pkl` (307KB) | 4차원 감성 피처 (pos, neg, neu, compound) |
| eda | 완료 | `reports/eda/` 전체 | 텍스트 길이, VADER 분포, 타겟, 어휘 중첩 |
| tune | 완료 | `tuning/` 전체 | lr → batch → dropout → epochs 순차 탐색 |
| benchmark | 완료 | `benchmark_summary.csv` + `significance_tests.csv` | 6모델 x 3시드 + paired t-test |
| freeze-study | 완료 | `freeze_study.csv` | Frozen +109% 차이 확인 |
| xai | 완료 | `xai/` 전체 | SHAP/LIME + Overlap@5 + Human Rationale 비교 |
| dashboard | 완료 | `dashboard_app.py` (FastAPI, 18탭) | Playground + 이미지 팝업 + PDF Export |

---

## 최근 구현 내역 (2026-04-10 ~ 04-11)

### 대시보드 대규모 확장 (18탭)

- [x] **6개 신규 기능** — Data Explorer, Model Comparison, Report 자동생성, 이미지 팝업 Lightbox, PDF Export, 인쇄 CSS
- [x] **Playground XAI 강화** — Attention Heatmap + LIME 설명 + 4모델 실시간 추론
- [x] **Pipeline Deep-Dive** — 8단계 연구 방법론 스토리 (가설→XAI 사후검증)
- [x] **E2E Pipeline Analysis** — 데이터 볼륨 추적, 차원 변환, 시간 프로파일링
- [x] **Error Analysis** — 오분류 패턴, VADER 맹점, Ablation 다이어그램
- [x] **References** — 7개 핵심 문헌 + 재현성 가이드 + 학습 시간

### 모델 학습 개선

- [x] **Class weighting + Label smoothing** 적용 및 전체 모델 재학습
- [x] 50개 샘플 XAI 분석 (xai_sample_size 확대)

### Human Rationale 비교 (설명 타당성 평가)

- [x] `_load_human_rationales()` — dataset.json에서 majority vote rationale 로딩
- [x] `_compute_rationale_overlap()` — Model Top-5 vs Human rationale 토큰 overlap 계산
- [x] `_plot_rationale_comparison()` — Baseline vs Improved 박스플롯 시각화
- [x] `run_xai()` 파이프라인에 통합 (Step 6-1)
- [x] `xai_summary.json/md`에 rationale 메트릭 6개 추가
- [x] 대시보드 XAI 탭에 Human Rationale Alignment 카드 + 차트 추가

### 인프라 / 배포

- [x] `outputs/` (38MB) + `data/` (12MB) git 포함 — pull 즉시 대시보드 작동
- [x] `models/` (1.7GB) Google Drive 배포 — Playground용 최적 체크포인트 4개
- [x] Playground Attention Heatmap `return_dict=True` 수정
- [x] rationale overlap 1.0 초과 버그 수정 (human 토큰 기준 커버율로 변경)

### 문서 정비

- [x] README 전면 재작성 (Quick Start, 실험 결과, 체크포인트 다운로드 가이드)
- [x] 대시보드 실행 명령어 상세 (포트 변경, 백그라운드, 의존성 테이블)

---

## 다음 할 일

### 즉시 실행 필요
- [ ] **`./run.sh xai` 재실행** — rationale overlap 버그 수정 후 결과 갱신 필요

### 향후 개선 가능
- [ ] 시드 5~10회로 확장하여 통계적 검정력 강화
- [ ] HateBERT / TweetBERT 등 도메인 특화 모델 비교
- [ ] feature-level SHAP으로 VADER 4차원의 기여도 직접 측정
- [ ] Overlap@K 민감도 분석 (K=3, 5, 10)
- [ ] 한국어/다국어 혐오표현 데이터셋 확장

---

## 이전 세션 완료 작업 (2026-04-06 이전)

- [x] EDA 모듈 신설 (`experiment_eda.py`)
- [x] Ablation 모델 추가 (`TransformerMLPClassifier`)
- [x] 통계 검정 추가 (`compute_pairwise_significance()`)
- [x] 서술 정직화 ("순환적 프레임워크" → "과학적 검증 프레임워크")
- [x] Overlap@5 fuzzy 매칭 버그 수정
- [x] compat/ 래퍼 14개 삭제 + 호환 alias 제거
- [x] 하네스 시스템 구축 (CLAUDE.md, progress.md, architecture.md)
- [x] 전체 코드베이스 다정한 한국어 주석 추가

---

## 알려진 이슈

| 이슈 | 심각도 | 상태 | 해결 방법 |
|------|:------:|:----:|---------|
| rationale 수치 갱신 필요 | 중 | 미해결 | `./run.sh xai` 재실행 |
| SHAP MPS 비호환 | 낮 | 코드에 반영됨 | CPU fallback 구현 완료 |
| 3-seed 반복은 통계 검정력 낮음 | 낮 | 인지됨 | 한계점으로 서술, 향후 5~10 seed |
| `cat` → `bat` alias 문제 | 낮 | 우회 완료 | HEREDOC 대신 직접 -m "..." |
| 폰트 경고 (CLOWN FACE glyph) | 낮 | 무시 가능 | DejaVu Sans에 이모지 없음, 차트에 영향 없음 |

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
