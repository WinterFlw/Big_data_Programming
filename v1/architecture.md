# architecture.md — 시스템 구조 및 데이터 흐름

> AI 에이전트가 코드 수정 전 전체 구조를 파악하기 위한 지도입니다.
> 이 파일은 하네스 시스템에서 **Planner 역할의 참조 문서**입니다.
> 마지막 업데이트: 2026-04-11 (1차 파이프라인) → 2026-04-30 (v2.1 안내 추가)

> ⚠ **본 문서는 1차 파이프라인 (v1 baseline) 기준입니다.**
> v2.1로 갱신된 내용 (8조건 ablation, XAI 4축, 모델 입력 단일 소스 원칙)은 다음을 참조하세요:
> - **단일 출처 명세서**: [`docs/파이프라인_명세서_v2.md`](docs/파이프라인_명세서_v2.md)
> - **참고문헌**: [`docs/참고문헌_v2.md`](docs/참고문헌_v2.md)
>
> 본 architecture.md는 1차 파이프라인 구현 구조의 baseline 기록으로 유지됩니다. v2.1 코드 갱신 후 동기화 예정.

---

## v2.1 변경사항 요약 (2026-04-30)

| 영역 | v1 (본 문서) | v2.1 (명세서 v2) |
|------|------|------|
| Ablation | 4조건 (CE/원본 vs αAttn/aug) | **8조건** (BERT × 4 + RoBERTa × 4) |
| 학습 augmentation | Slur-Masking (p=0.3) | **제거** ([MASK] shortcut 위험) |
| XAI 축 | 3축 (SHAP+LIME+Masking) | **4축** (Attribution / Faithfulness / **Context Learning(CI/IS/MSS)** / Plausibility) |
| 모델 입력 | 텍스트 + (Slur-Masked) | **텍스트만** (메타는 학습 supervision으로만) |
| 가설 | H1/H2/H3 | **H1~H4** (H4 multi-task 부가) |
| 학습 헤드 | 단일 | **듀얼 헤드** (D_B 부가만, Main + Aux for L_target) |
| 통계 검정 | McNemar / Wilcoxon | **Two-way ANOVA + 3-way ANOVA + paired t-test** |

→ v2.1 코드 작업 시 본 문서의 v1 흐름은 baseline reference로만 활용. 실제 구조는 명세서 v2.1을 따른다.

---

---

## 0. 하네스 제어 구조

이 프로젝트의 AI 작업은 4개의 역할로 분리됩니다.

```
┌─────────────────────────────────────────────────────┐
│  Orchestrator (오케스트레이터)                         │
│  run.sh → run_experiments.py                         │
│  전체 파이프라인 흐름을 총괄                            │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Planner  │→│Generator │→│Evaluator │──┐         │
│  │          │  │          │  │          │  │ 실패    │
│  │progress  │  │코드 작성  │  │CLAUDE.md │  │         │
│  │.md 읽기  │  │문서 수정  │  │규칙 검증  │  │         │
│  │작업 계획  │  │실험 실행  │  │일관성 확인│  │         │
│  └──────────┘  └──────────┘  └─────┬────┘  │         │
│                                    │       │         │
│                                 통과?      │         │
│                              ┌────┴────┐   │         │
│                              │Yes      │No │         │
│                              ▼         └───┘         │
│                     progress.md 업데이트              │
│                     → 커밋                            │
└─────────────────────────────────────────────────────┘
```

**건축 비유:**
- **Orchestrator** = 현장 소장 (run.sh가 전체 공정 지휘)
- **Planner** = 건축가 (progress.md + architecture.md로 작업 설계)
- **Generator** = 시공 팀 (실제 코드/문서 작업 수행)
- **Evaluator** = 감리사 (CLAUDE.md 규칙으로 품질 검수)

**핵심:** 감리사(Evaluator)가 문제를 발견하면, 시공 팀(Generator)에게 돌려보내 재작업합니다. 이 **생성 → 검증 → 재생성 루프**가 하네스의 품질 보장 메커니즘입니다.

---

## 1. 실행 흐름 (Entry Point → Module → Function)

```
사용자
  │
  ▼
run.sh  ─────────────────── CLI 진입점 (zsh)
  │
  ▼
run_experiments.py  ──────── Python 엔트리포인트 (argparse)
  │
  ├── "data"      → experiment_core.prepare_data()
  ├── "vader"     → experiment_core.extract_vader_features()
  ├── "eda"       → experiment_eda.run_eda()
  ├── "tune"      → experiment_core.run_hyperparameter_tuning()
  ├── "benchmark" → experiment_core.run_benchmark()
  ├── "freeze-study" → experiment_core.run_freeze_study()
  ├── "xai"       → experiment_xai.run_xai()
  ├── "dashboard" → experiment_dashboard.run_dashboard()
  ├── "all"       → 위 전부 순서대로 (tune은 --with-tuning일 때만)
  ├── "status"    → experiment_core.describe_status()
  └── "clean"     → outputs/ + checkpoints/ 삭제
```

### 대시보드 (별도 서버)

```
사용자
  │
  ▼
python3 dashboard_app.py  ── FastAPI 서버 (포트 8501)
  │
  ├── GET /                → 18탭 HTML 대시보드 (Chart.js)
  ├── GET /api/predict/status → Playground 모델 상태
  ├── POST /api/predict    → 실시간 모델 추론 (Attention + LIME)
  ├── GET /api/explorer    → 데이터셋 검색 API
  └── /static/*            → outputs/ 이미지 서빙
```

---

## 2. 모듈 의존성

```
utils.py (720줄) ◄──────── 모든 모듈이 의존
  │
  ├── 경로 상수 (OUTPUT_DIR, CHECKPOINT_DIR, ...)
  ├── seed_everything(), get_device()
  ├── compute_metrics(), compute_pairwise_significance()
  └── save_json(), save_text(), plot helpers

experiment_core.py (1,662줄) ◄── 핵심 엔진
  │
  ├── 데이터: prepare_data(), load_splits()
  ├── VADER: extract_vader_features()
  ├── 모델: TransformerCLSClassifier
  │         TransformerMLPClassifier  (ablation)
  │         HybridSentimentClassifier (VADER hybrid)
  ├── 학습: train_neural_model() [class weighting + label smoothing]
  ├── 벤치마크: run_tfidf_benchmark(), run_transformer_benchmark()
  ├── 튜닝: run_hyperparameter_tuning()
  ├── Freeze: run_freeze_study()
  └── 상태: describe_status()

experiment_eda.py (632줄)
  │
  ├── run_eda()
  ├── 의존: prepare_data() + extract_vader_features()
  └── 출력: outputs/reports/eda/

experiment_xai.py (1,109줄)
  │
  ├── run_xai()
  ├── 의존: data_splits.pkl + vader_features.pkl + best_models.json
  ├── 출력: outputs/xai/
  └── 신규: Human Rationale 비교 (Step 6-1)
  │         ├── _load_human_rationales()     dataset.json → majority vote
  │         ├── _compute_rationale_overlap() Model Top-5 vs Human tokens
  │         └── _plot_rationale_comparison() 박스플롯 시각화

experiment_dashboard.py (1,083줄)
  │
  ├── run_dashboard()
  ├── 의존: 모든 이전 산출물 (optional — 없으면 빈 값)
  └── 출력: outputs/dashboard/index.html (정적 HTML)

dashboard_app.py (4,021줄) ◄── FastAPI 대시보드 서버
  │
  ├── 18탭 인터랙티브 대시보드 (Chart.js)
  ├── Playground: v2.1 best_models 기반 실시간 추론 + Attention Heatmap + LIME
  ├── 의존: outputs/, data/, checkpoints/ (best_models.json 생성 이후)
  └── 독립 실행: python3 dashboard_app.py (포트 8501)

run_experiments.py (397줄) ◄── 오케스트레이터
  │
  └── 위 모듈들을 argparse 명령어에 매핑
```

---

## 3. 데이터 흐름

```
HateXplain JSON (GitHub)
    │
    ▼ prepare_data()
    │
    ├── 다수결 라벨링 (3인 annotator 중 2인 이상 동의)
    ├── Undecided 제외 → ~13,433건
    ├── Stratified Split: Train 70% / Val 10% / Test 20%
    │
    ▼ data_splits.pkl
    │
    ├───────────────────────────────┐
    │                               │
    ▼ extract_vader_features()      │
    │                               │
    ▼ vader_features.pkl            │
    │                               │
    ├── pos, neg, neu, compound     │
    │   (각 샘플당 4차원 벡터)       │
    │                               │
    ▼ run_eda()                     │
    │                               │
    ├── 텍스트 길이 분포             │
    ├── 클래스별 VADER 분포          │
    ├── 타겟 커뮤니티 분석           │
    └── 어휘 중복 (Jaccard)         │
                                    │
    ┌───────────────────────────────┘
    │
    ▼ run_hyperparameter_tuning() [선택]
    │
    ├── lr → batch → dropout → epochs 순차 탐색
    │
    ▼ transformer_tuning_best.json
    │
    ▼ run_benchmark()
    │
    ├── TF-IDF + LR        ─┐
    ├── TF-IDF + SVM        │  각 모델 x 3 seeds
    ├── BERT-base           │  (42, 52, 62)
    ├── BERT+MLP (ablation) │  [class weighting + label smoothing]
    ├── BERT+VADER          │
    └── RoBERTa+VADER      ─┘
    │
    ├── benchmark_summary.csv (mean +- std)
    ├── significance_tests.csv (paired t-test + Cohen's d)
    ├── best_models.json (최고 체크포인트 경로)
    └── checkpoints/*.pt
    │
    ▼ run_freeze_study()
    │
    ├── BERT+VADER (frozen encoder)   x 3 seeds
    └── BERT+VADER (fine-tuned)       x 3 seeds
    │
    ▼ run_xai()
    │
    ├── Baseline (BERT-base) 체크포인트 로드
    ├── Improved (RoBERTa+VADER) 체크포인트 로드
    ├── SHAP 분석 (CPU only)
    ├── LIME 분석
    ├── Overlap@5 교차 검증 (설명 안정성)
    ├── Human Rationale 비교 (설명 타당성)  ← 신규
    └── Before/After 케이스 비교
    │
    ▼ dashboard_app.py (FastAPI)
    │
    └── 18탭 인터랙티브 대시보드 + Playground
```

---

## 4. 모델 아키텍처

### 4.1 TransformerCLSClassifier (Baseline)

```
Input Text → Tokenizer → BERT-base-uncased
                              │
                          [CLS] (768d)
                              │
                          Dropout(p)
                              │
                        Linear(768, 3)
                              │
                        3-class output
                        (분류기 파라미터: ~2.3K)
```

### 4.2 TransformerMLPClassifier (Ablation)

```
Input Text → Tokenizer → BERT-base-uncased
                              │
                          [CLS] (768d)
                              │
                          Dropout(p)
                              │
                      Linear(768, 256) → ReLU
                              │
                      Linear(256, 3)
                              │
                        3-class output
                        (분류기 파라미터: ~197K)
```

### 4.3 HybridSentimentClassifier (Improved)

```
Input Text → Tokenizer → BERT/RoBERTa
                              │
                          [CLS] (768d)
                              │
Input Text → VADER → [pos, neg, neu, compound] (4d)
                              │
                    Concatenation → (772d)
                              │
                          Dropout(p)
                              │
                      Linear(772, 256) → ReLU
                              │
                      Linear(256, 3)
                              │
                        3-class output
                        (분류기 파라미터: ~198K)
```

---

## 5. XAI 분석 파이프라인 (이중 검증 구조)

```
run_xai()
  │
  ├── Step 1-3: 테스트셋 예측 + 성능 지표 + 혼동 행렬
  │
  ├── Step 4: 분석 대상 샘플 선정 (50개)
  │            ├── fixed_error (오분류→정분류) 우선
  │            ├── consistently_correct
  │            ├── model_disagreement
  │            └── fallback
  │
  ├── Step 5: SHAP + LIME 설명 생성
  │            ├── Baseline SHAP / LIME
  │            └── Improved SHAP / LIME
  │
  ├── Step 6: Overlap@5 (설명 안정성)
  │            └── SHAP Top-5 vs LIME Top-5 Jaccard
  │               "두 XAI 기법이 서로 동의하는가?"
  │
  ├── Step 6-1: Human Rationale 비교 (설명 타당성)  ← 신규
  │            ├── dataset.json에서 annotator rationale 로드
  │            ├── majority vote → binary mask
  │            ├── Model Top-5 vs Human rationale overlap
  │            │   "모델이 인간과 같은 근거를 보는가?"
  │            ├── rationale_overlap.csv 저장
  │            └── 박스플롯 시각화 (SHAP/LIME 각각)
  │
  ├── Step 7: 케이스별 SHAP 비교 차트 (최대 8개)
  │
  └── Step 8: 요약 저장
              ├── xai_summary.json (Overlap + Rationale 메트릭)
              ├── xai_summary.md (마크다운 테이블)
              └── xai_details.json (SHAP/LIME/Rationale 상세)
```

### 이중 검증 프레임워크

| 검증 차원 | 지표 | 비교 대상 | 질문 |
|-----------|------|-----------|------|
| **설명 안정성** | Overlap@5 (Jaccard) | SHAP Top-5 vs LIME Top-5 | 두 XAI 기법이 서로 동의하는가? |
| **설명 타당성** | Rationale Overlap | Model Top-5 vs Human Rationale | 모델이 인간과 같은 근거를 보는가? |

---

## 6. 대시보드 아키텍처

### 6.1 정적 대시보드 (experiment_dashboard.py)

`./run.sh dashboard` → `outputs/dashboard/index.html` 생성

### 6.2 FastAPI 대시보드 (dashboard_app.py)

`python3 dashboard_app.py` → http://localhost:8501

```
dashboard_app.py (4,021줄, 단일 파일)
  │
  ├── 데이터 로딩 (outputs/ JSON/CSV 읽기)
  │
  ├── API 엔드포인트
  │   ├── GET /              → 18탭 HTML 응답 (Chart.js 내장)
  │   ├── GET /api/predict/status → 모델 로드 상태
  │   ├── POST /api/predict  → 실시간 추론
  │   │   ├── v2 조건 지원 (A_B/B_B/C_B/D_B/A_R/B_R/C_R/D_R + D_B+Target)
  │   │   ├── Attention Heatmap (return_dict=True + output_attentions)
  │   │   ├── LIME 설명 (선택)
  │   │   └── VADER 감성 점수
  │   └── GET /api/explorer  → 데이터셋 검색 (dataset.json fallback)
  │
  ├── 모델 관리
  │   ├── BEST_MODELS_PATH = outputs/reports/best_models.json
  │   ├── checkpoint_path = best_models.json의 각 모델별 경로
  │   └── Lazy loading + 캐싱 (_playground_models dict)
  │
  └── 18탭 구성
      ├── 1. Overview          KPI 카드, 파이프라인 흐름도
      ├── 2. Pipeline Deep-Dive 8단계 연구 방법론
      ├── 3. E2E Pipeline      데이터 볼륨, 차원 변환, 시간
      ├── 4. Benchmark         F1 차트, 레이더, 테이블
      ├── 5. Statistical Tests P-value 히트맵
      ├── 6. Tuning            LR/Dropout 탐색 궤적
      ├── 7. Learning Curves   에폭별 학습곡선
      ├── 8. Freeze Study      Frozen vs Fine-tuned
      ├── 9. EDA               VADER, 타겟, 어휘 중첩
      ├── 10. XAI Analysis     Overlap@5 + Human Rationale
      ├── 11. XAI Cases        개별 SHAP/LIME 갤러리
      ├── 12. Error Analysis   오분류 패턴, 한계점
      ├── 13. Architecture     모델 구조 시각화
      ├── 14. Data Explorer    데이터셋 검색/필터
      ├── 15. Comparison       2모델 레이더 + Delta
      ├── 16. Report           자동 보고서 생성
      ├── 17. References       문헌 + 재현성 가이드
      └── 18. Playground       4모델 추론 + Attention + LIME
```

---

## 7. 디렉토리 구조

```
HateSpeachStudy/
│
├── CLAUDE.md              ← 하네스: AI 에이전트 규칙
├── progress.md            ← 하네스: 진행 상태 + Todo
├── architecture.md        ← 하네스: 이 파일 (시스템 구조도)
│
├── run.sh                 ← CLI 실행기
├── run_experiments.py     ← Python 엔트리포인트 (397줄)
├── experiment_core.py     ← 핵심 엔진 (1,662줄)
├── experiment_eda.py      ← EDA (632줄)
├── experiment_xai.py      ← XAI + Human Rationale (1,109줄)
├── experiment_dashboard.py ← 정적 대시보드 (1,083줄)
├── dashboard_app.py       ← FastAPI 대시보드 (4,021줄)
├── utils.py               ← 유틸리티 (720줄)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/                  ← HateXplain 원본 (git 포함, 12MB)
│   ├── dataset.json       ← 본문 + annotator rationale
│   └── post_id_divisions.json
│
├── checkpoints/           ← v2.1 재학습 체크포인트 (git 제외)
│
├── outputs/               ← 실험 결과 (git 포함, 38MB)
│   ├── data_splits.pkl
│   ├── vader_features.pkl
│   ├── experiment_config.json
│   ├── reports/
│   │   ├── eda/           ← EDA 시각화 + 통계
│   │   ├── benchmark_summary.csv
│   │   ├── significance_tests.csv
│   │   └── freeze_study.csv
│   ├── tuning/            ← 하이퍼파라미터 탐색
│   ├── runs/              ← seed별 개별 결과
│   ├── xai/               ← SHAP/LIME + Rationale 분석
│   │   ├── overlap_at_5.csv/png       ← 설명 안정성
│   │   ├── rationale_overlap.csv      ← 설명 타당성 (신규)
│   │   ├── rationale_overlap_shap.png
│   │   ├── rationale_overlap_lime.png
│   │   ├── xai_summary.json/md
│   │   ├── xai_details.json
│   │   ├── analysis_samples.csv
│   │   ├── case_summary.csv
│   │   └── cases/                     ← 케이스별 비교 차트
│   ├── dashboard/         ← 정적 HTML
│   └── logs/
│
└── docs/                  ← 프로젝트 문서 (한국어)
    ├── 파이프라인_상세분석.md
    ├── 파이프라인_검증_및_팀분업_가이드.md
    ├── 참고자료_종합_가이드.md
    ├── 프로젝트개요_재작성_2버전.md
    ├── 표절검증_최종보고서_v2.md
    ├── 표절위험도_평가보고서.md
    └── 하네스_엔지니어링_적용사례.md
```

---

## 8. 학습 파이프라인 상세

### 8.1 클래스 불균형 대응

```
compute_class_weight_tensor(labels, imbalance_threshold=0.40)
  │
  ├── 클래스 분포: hate 29.5% / offensive 27.2% / normal 38.8%
  ├── 소수 클래스 비율 27.2% < threshold 40% → balanced 가중치 적용
  │
  ├── DL: CrossEntropyLoss(weight=balanced_tensor, label_smoothing=0.1)
  └── ML: class_weight="balanced"
```

| 기법 | 적용 대상 | 메커니즘 |
|------|----------|---------|
| Class Weighting | ML + DL 전부 | sklearn `compute_class_weight("balanced")` → 소수 클래스에 높은 가중치 |
| Label Smoothing 0.1 | DL만 | hard [1,0,0] → soft [0.933,0.033,0.033]. hate/offensive 경계 모호함 대응 |

### 8.2 ML 하이퍼파라미터 튜닝 (TF-IDF + LR / SVM)

```
run_tfidf_baselines()
  │
  ├── TF-IDF 벡터화
  │   ├── ngram_range=(1, 3)     1~3gram 피처
  │   ├── max_features=50,000    빈도 상위 5만개
  │   └── sublinear_tf=True      1 + log(tf) 적용
  │
  ├── C 파라미터 Grid Search (시드별 독립)
  │   ├── 후보: [0.5, 1.0, 2.0]
  │   ├── 선택 기준: val macro F1
  │   └── SVM: CalibratedClassifierCV(cv=3)으로 확률 출력 추가
  │
  └── 3시드 x 2모델 = 6회 학습
```

### 8.3 DL 하이퍼파라미터 튜닝 (순차 탐색)

```
_tune_single_model()
  │
  ├── Step 1: learning_rate [1e-5, 2e-5, 3e-5] → 최적값 확정
  │                                                    │
  ├── Step 2: batch_size [64]                    → 고정 (MPS 한계)
  │                          ↑ lr 최적값 적용         │
  ├── Step 3: dropout [0.1, 0.2, 0.3]           → 최적값 확정
  │                          ↑ lr + batch 적용        │
  └── Step 4: epochs [5]                         → 고정 (early stopping)
                             ↑ lr + batch + dropout 적용

  대상: BERT+MLP/BERT+VADER/RoBERTa+VADER 순차 탐색 + B_B alpha grid + D_B+Target beta grid
  튜닝 시드: 42 고정 | 선택 기준: val macro F1
```

**Grid Search가 아닌 순차 탐색을 택한 이유:**
- Grid Search: 전체 조합 x 여러 조건을 모두 도는 방식은 v2.1 8조건에서 비용이 큼
- Sequential: 공통 기본 하이퍼파라미터를 줄인 뒤 B_B alpha와 D_B+Target beta만 별도 grid로 검증
- 파라미터 간 상호작용이 적다는 가정 하에 효율적

### 8.4 학습 루프 (train_neural_model)

```
train_neural_model()
  │
  ├── 옵티마이저: AdamW (weight_decay=0.01)
  ├── 스케줄러: Linear warmup (10%) → linear decay
  ├── 손실함수: CrossEntropyLoss(weight=balanced, label_smoothing=0.1)
  │
  ├── epoch 루프 (최대 5회)
  │   ├── train: forward → loss → backward → optimizer.step()
  │   ├── val: val macro F1 계산
  │   └── early stopping: patience=2 (2에폭 연속 개선 없으면 중단)
  │
  ├── 체크포인트 저장: best val F1 기준 model_state + optimizer_state
  └── 테스트 평가: 최적 체크포인트 로드 → test metrics 계산
```

### 8.5 핵심 설정값 요약 (experiment_config.json)

| 카테고리 | 파라미터 | 기본값 | 비고 |
|---------|---------|--------|------|
| **데이터** | split ratio | 70 / 10 / 20 | stratified |
| **토크나이저** | max_len | 128 | EDA 99%+ 커버 |
| **학습** | batch_size | 64 | MPS 메모리 한계 |
| | epochs | 5 | early stopping이 실질 종료 |
| | learning_rate | 2e-5 | 튜닝 확정값 |
| | warmup_ratio | 0.10 | 전체 스텝의 10% |
| | weight_decay | 0.01 | AdamW L2 |
| | dropout | 0.10 | 튜닝 확정값 |
| | early_stopping_patience | 2 | |
| | class_weighting | True (auto) | threshold=0.40 |
| | label_smoothing | 0.1 | hate/offensive 경계 |
| **MLP** | mlp_hidden | 256 | Hybrid/MLP 공통 |
| **벤치마크** | seeds | [42, 52, 62] | paired t-test 요건 |
| **ML 튜닝** | C candidates | [0.5, 1.0, 2.0] | val F1 기준 |
| **DL 튜닝** | lr candidates | [1e-5, 2e-5, 3e-5] | 순차 탐색 |
| | dropout candidates | [0.1, 0.2, 0.3] | 순차 탐색 |
| **XAI** | xai_sample_size | 50 | |
| | lime_num_features | 5 | |
| | shap_max_evals | 300 | |

---

## 9. 위험 지점 (수정 시 주의)

| 파일 | 위치 | 위험 | 이유 |
|------|------|------|------|
| `experiment_core.py` | `_TUNING_KEY_TO_MODEL_NAME` | 모델명 매핑 | 매핑 오류 시 잘못된 HF 모델 로드 |
| `experiment_core.py` | `run_transformer_benchmark()` model_specs | 모델 목록 | 여기서 빠지면 벤치마크에서 누락 |
| `experiment_core.py` | `load_tuned_hyperparams()` | 기본값 dict | 새 모델 추가 시 반드시 업데이트 |
| `experiment_xai.py` | `_compute_overlap_at_5()` | fuzzy 매칭 | 이전 버그 수정 완료 |
| `experiment_xai.py` | `_compute_rationale_overlap()` | human 토큰 기준 | 1.0 초과 버그 수정 완료 |
| `experiment_xai.py` | SHAP device | CPU 강제 | MPS에서 SHAP 비호환 |
| `dashboard_app.py` | `_predict_single()` | model.encoder 호출 | return_dict=True 필수 |
| `dashboard_app.py` | `BEST_MODELS_PATH` / checkpoint_path | 체크포인트 경로 | best_models.json 생성 전 Playground 비활성 |
| `utils.py` | `compute_pairwise_significance()` | 통계 검정 | scipy.stats.ttest_rel 사용 |
