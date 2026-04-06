# progress.md — 파이프라인 진행 상태

> 마지막 업데이트: 2026-04-06
> 이 파일은 하네스 시스템에서 **Planner 역할의 현재 위치 파악 문서**입니다.
> AI 에이전트는 작업 시작 전 반드시 이 파일을 읽고 현재 위치를 파악하세요.
> 작업 완료 후 Evaluator 검증을 통과하면, 이 파일을 업데이트합니다.

---

## 파이프라인 완료 현황

```
[완료] data ──→ [완료] vader ──→ [완료] eda ──→ [부분] tune ──→ [미실행] benchmark
                                                                      │
                                                              [미실행] freeze-study
                                                                      │
                                                              [미실행] xai
                                                                      │
                                                              [미실행] dashboard (재생성 필요)
```

| 단계 | 상태 | 산출물 | 비고 |
|------|:----:|--------|------|
| data | ✅ 완료 | `data_splits.pkl` (3.9MB) | 13,433건, 70/10/20 split |
| vader | ✅ 완료 | `vader_features.pkl` (307KB) | 4차원 감성 피처 |
| eda | ✅ 완료 | `reports/eda/` 전체 | 시각화 + 통계 |
| tune | ⚠️ 부분 | `tuning/bert_base/` 12개 설정 완료 | bert_vader 1개에서 중단. 통합 로그 미생성 |
| benchmark | ❌ 미실행 | — | tune 완료 후 실행 필요 |
| freeze-study | ❌ 미실행 | — | benchmark 후 실행 |
| xai | ❌ 미실행 | — | benchmark 체크포인트 필요 |
| dashboard | ⚠️ 갱신 필요 | `dashboard/index.html` 존재 | 데이터 부족 상태로 생성됨 |

---

## 다음 할 일 (우선순위순)

### 즉시 실행 가능
- [ ] **tune 재실행** — `./run.sh tune --force` 로 bert_vader, bert_mlp, roberta_vader 튜닝 완료
- [ ] **benchmark 실행** — `./run.sh benchmark` (tune 없이도 기본값으로 실행 가능)
- [ ] 또는 한 번에: `./run.sh all --with-tuning`

### benchmark 완료 후
- [ ] **freeze-study 실행** — `./run.sh freeze-study`
- [ ] **xai 실행** — `./run.sh xai` (SHAP은 CPU only, 시간 소요 큼)
- [ ] **dashboard 재생성** — `./run.sh dashboard`

### 결과 분석
- [ ] benchmark 결과 확인: `outputs/reports/benchmark_summary.csv`
- [ ] 통계 검정 확인: `outputs/reports/significance_tests.csv`
- [ ] XAI Before/After 비교: `outputs/xai/xai_summary.json`
- [ ] ablation 결과 해석: BERT+MLP vs BERT+VADER 차이 분석

### 보고서
- [ ] Cheng(2022) + HateXplain 2.0 논문 PDF 확보
- [ ] Related Work 섹션 작성 (차별점 명시)
- [ ] 최종 보고서 통합 편집

---

## 최근 완료된 작업

### 2026-04-06
- [x] docs/ 문서 전체 고도화 (EDA, ablation, 통계검정, 서술 정직화)
- [x] 파이프라인 검증 및 팀분업 가이드 작성
- [x] 참고자료 종합 가이드 작성 (논문 23편 매핑)
- [x] compat/ 래퍼 14개 삭제 + 호환 alias 제거
- [x] 깨진 유니코드 수정, run.sh help 보완
- [x] 하네스 시스템 구축 (CLAUDE.md, progress.md, architecture.md)
- [x] 하네스 고도화: 4역할 구조 (Planner/Generator/Evaluator/Orchestrator) + 생성-검증-재생성 루프 + Adaptive 제어 수준

### 이전 세션
- [x] EDA 모듈 신설 (`experiment_eda.py`)
- [x] Ablation 모델 추가 (`TransformerMLPClassifier`)
- [x] 통계 검정 추가 (`compute_pairwise_significance()`)
- [x] 서술 정직화 ("순환적 프레임워크" → "과학적 검증 프레임워크")
- [x] Overlap@5 fuzzy 매칭 버그 수정
- [x] XAI 차트 하드코딩 수정
- [x] 튜닝 메타데이터 model_name 매핑 수정
- [x] 전체 코드베이스 다정한 한국어 주석 추가
- [x] 루트 디렉토리 정리 (docs/, compat/ 분리)

---

## 알려진 이슈

| 이슈 | 심각도 | 상태 | 해결 방법 |
|------|:------:|:----:|---------|
| tune이 bert_vader에서 중단됨 | 중 | 미해결 | `./run.sh tune --force` 재실행 |
| SHAP MPS 비호환 | 낮 | 코드에 반영됨 | CPU fallback 구현 완료 |
| 3-seed 반복은 통계 검정력 낮음 | 낮 | 인지됨 | 한계점으로 서술, 향후 5~10 seed |
| `cat` → `bat` alias 문제 | 낮 | 우회 완료 | HEREDOC 대신 직접 -m "..." |

---

## 세이브 포인트

| 시점 | git commit | 설명 |
|------|-----------|------|
| 최신 | `28e82f2` | 문서 최종 정리, 깨진 유니코드 수정 |
| 정리 완료 | `adb9f7b` | compat 삭제, 호환 alias 제거 |
| 참고자료 | `783060b` | 논문 23편 매핑 가이드 |
| 문서 고도화 | `d41cea6` | EDA/ablation/통계/서술 문서 반영 |
| 버그 수정 | `2a7bed9` | Overlap@5 + XAI차트 + 튜닝 메타데이터 |
| 한국어 주석 | `f86e9d5` | 전체 코드베이스 주석 |
| 초기 커밋 | `4bed3a9` | 파이프라인 첫 커밋 |
