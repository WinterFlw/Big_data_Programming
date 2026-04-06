# architecture.md — 시스템 구조 및 데이터 흐름

> AI 에이전트가 코드 수정 전 전체 구조를 파악하기 위한 지도입니다.

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

---

## 2. 모듈 의존성

```
utils.py  ◄──────────────── 모든 모듈이 의존
  │
  ├── 경로 상수 (OUTPUT_DIR, CHECKPOINT_DIR, ...)
  ├── seed_everything(), get_device()
  ├── compute_metrics(), compute_pairwise_significance()
  └── save_json(), save_text(), plot helpers

experiment_core.py  ◄─────── 핵심 엔진
  │
  ├── 데이터: prepare_data(), load_splits()
  ├── VADER: extract_vader_features()
  ├── 모델: TransformerCLSClassifier
  │         TransformerMLPClassifier  (ablation)
  │         HybridSentimentClassifier (VADER hybrid)
  ├── 학습: train_neural_model()
  ├── 벤치마크: run_tfidf_benchmark(), run_transformer_benchmark()
  ├── 튜닝: run_hyperparameter_tuning()
  ├── Freeze: run_freeze_study()
  └── 상태: describe_status()

experiment_eda.py
  │
  ├── run_eda()
  ├── 의존: prepare_data() + extract_vader_features()
  └── 출력: outputs/reports/eda/

experiment_xai.py
  │
  ├── run_xai()
  ├── 의존: data_splits.pkl + vader_features.pkl + best_models.json (from benchmark)
  └── 출력: outputs/xai/

experiment_dashboard.py
  │
  ├── run_dashboard()
  ├── 의존: 모든 이전 산출물 (optional — 없으면 빈 값)
  └── 출력: outputs/dashboard/index.html

run_experiments.py  ◄──────── 오케스트레이터
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
    ├── Undecided 919건 제외 → ~13,433건
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
    ├── TF-IDF + SVM        │  각 모델 × 3 seeds
    ├── BERT-base           │  (42, 52, 62)
    ├── BERT+MLP (ablation) │
    ├── BERT+VADER          │
    └── RoBERTa+VADER      ─┘
    │
    ├── benchmark_summary.csv (mean ± std)
    ├── significance_tests.csv (paired t-test + Cohen's d)
    ├── best_models.json (최고 체크포인트 경로)
    └── checkpoints/*.pt
    │
    ▼ run_freeze_study()
    │
    ├── BERT+VADER (frozen encoder)   × 3 seeds
    └── BERT+VADER (fine-tuned)       × 3 seeds
    │
    ▼ run_xai()
    │
    ├── Baseline (BERT-base) 체크포인트 로드
    ├── Improved (BERT+VADER) 체크포인트 로드
    ├── SHAP 분석 (CPU only)
    ├── LIME 분석
    ├── Overlap@5 교차 검증
    └── Before/After 케이스 비교
    │
    ▼ run_dashboard()
    │
    └── index.html (인터랙티브 대시보드)
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
                        (파라미터: ~2.3K)
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
                        (파라미터: ~197K)
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
                        (파라미터: ~198K)
```

---

## 5. 디렉토리 구조

```
HateSpeachStudy/
│
├── CLAUDE.md              ← 하네스: AI 에이전트 규칙
├── progress.md            ← 하네스: 진행 상태 + Todo
├── architecture.md        ← 하네스: 이 파일 (시스템 구조도)
│
├── run.sh                 ← CLI 실행기
├── run_fresh_full.sh      ← 클린 재실행
├── run_experiments.py     ← Python 엔트리포인트
├── experiment_core.py     ← 핵심 엔진 (~1,650줄)
├── experiment_eda.py      ← EDA (~630줄)
├── experiment_xai.py      ← XAI (~870줄)
├── experiment_dashboard.py ← 대시보드 (~940줄)
├── utils.py               ← 유틸리티 (~730줄)
├── requirements.txt
├── README.md
├── .gitignore
│
├── docs/                  ← 프로젝트 문서 (9개)
│   ├── 파이프라인_상세분석.md
│   ├── 파이프라인_검증_및_팀분업_가이드.md
│   ├── 참고자료_종합_가이드.md
│   ├── 프로젝트개요_재작성_2버전.md
│   ├── 표절검증_최종보고서_v2.md
│   ├── 표절위험도_평가보고서.md
│   ├── 추가_논문 요약 및 리뷰.docx
│   └── 수행계획서.docx / .pdf
│
├── data/                  ← [.gitignore] HateXplain 원본
├── checkpoints/           ← [.gitignore] 모델 .pt 파일
└── outputs/               ← [.gitignore] 모든 산출물
    ├── data_splits.pkl
    ├── vader_features.pkl
    ├── experiment_config.json
    ├── reports/
    │   ├── eda/           ← EDA 시각화 + 통계
    │   ├── benchmark_summary.csv
    │   ├── significance_tests.csv
    │   └── freeze_study.csv
    ├── tuning/            ← 하이퍼파라미터 탐색
    ├── runs/              ← seed별 개별 결과
    ├── xai/               ← SHAP/LIME 분석
    ├── dashboard/         ← index.html
    └── logs/
```

---

## 6. 핵심 설정값 (experiment_config.json)

| 카테고리 | 파라미터 | 기본값 |
|---------|---------|--------|
| **데이터** | split ratio | 70 / 10 / 20 |
| **토크나이저** | max_len | 128 |
| **학습** | batch_size | 64 |
| | epochs | 5 (early stopping) |
| | learning_rate | 2e-5 |
| | warmup_ratio | 0.10 |
| | weight_decay | 0.01 |
| | dropout | 0.10 |
| | early_stopping_patience | 2 |
| **MLP** | mlp_hidden | 256 |
| **벤치마크** | seeds | [42, 52, 62] |
| **XAI** | xai_sample_size | 24 |
| | lime_num_features | 5 |
| | shap_max_evals | 300 |

---

## 7. 위험 지점 (수정 시 주의)

| 파일 | 위치 | 위험 | 이유 |
|------|------|------|------|
| `experiment_core.py` | `_TUNING_KEY_TO_MODEL_NAME` | 모델명 매핑 | 매핑이 틀리면 잘못된 HF 모델 로드 |
| `experiment_core.py` | `run_transformer_benchmark()` model_specs | 모델 목록 | 여기서 빠지면 벤치마크에서 누락 |
| `experiment_core.py` | `load_tuned_hyperparams()` | 기본값 dict | 새 모델 추가 시 여기도 업데이트 필수 |
| `experiment_xai.py` | `_compute_overlap_at_5()` | fuzzy 매칭 | 이전에 버그 있었음 — 수정 완료 |
| `experiment_xai.py` | SHAP device | CPU 강제 | MPS에서 SHAP 작동 안 함 |
| `utils.py` | `compute_pairwise_significance()` | 통계 검정 | scipy.stats.ttest_rel 사용 |
