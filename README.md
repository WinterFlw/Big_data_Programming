# Hate Speech Detection with XAI-Driven Improvement

> HateXplain 데이터셋 기반 혐오표현 탐지 파이프라인
> **XAI 진단 → 피처 보강 → XAI 재진단**의 순환적 프레임워크

---

## 핵심 목표

1. BERT 계열 모델의 **hate / offensive 오분류 원인**을 SHAP + LIME으로 진단
2. **VADER 감성 피처** 결합으로 감성 맥락 보강 (pos, neg, neu, compound 4차원)
3. 개선 전후 XAI 비교로 **"왜 나아졌는지"** 정량적으로 설명 (Overlap@5 지표)

---

## 실험 환경

| 항목 | 스펙 |
|------|------|
| **Hardware** | Apple M3 Max, 64GB Unified Memory |
| **Backend** | PyTorch MPS (Metal Performance Shaders) |
| **Python** | 3.13 + PyTorch 2.10 + Transformers 5.3 |
| **XAI** | SHAP (CPU only, DeepExplainer MPS 비호환) + LIME |

---

## 데이터셋

[HateXplain](https://github.com/hate-alert/HateXplain) (Mathew et al., AAAI 2021)

| 항목 | 내용 |
|------|------|
| 출처 | Twitter + Gab, 20,148건 영어 텍스트 |
| 분류 | 3-class: `hatespeech` / `offensive` / `normal` |
| 라벨링 | 3인 다수결 투표, undecided 제외 → **~13,433건** 사용 |
| 분할 | **70 / 10 / 20** stratified split (train / val / test) |

---

## 모델 구성

### Baselines (전통 ML + Transformer)

| 모델 | 설명 | 클래스 |
|------|------|--------|
| TF-IDF + LR | Logistic Regression (1~3gram, C 튜닝) | `sklearn` |
| TF-IDF + SVM | LinearSVC + CalibratedClassifierCV | `sklearn` |
| BERT-base | `bert-base-uncased` [CLS] → Linear → 3-class | `TransformerCLSClassifier` |

### Improved (VADER Hybrid)

| 모델 | 아키텍처 | 클래스 |
|------|---------|--------|
| BERT + VADER | `[CLS](768d) + VADER(4d)` → MLP(256) → ReLU → 3-class | `HybridSentimentClassifier` |
| RoBERTa + VADER | `[CLS](768d) + VADER(4d)` → MLP(256) → ReLU → 3-class | `HybridSentimentClassifier` |

### VADER 감성 피처 (4차원)

| 피처 | 설명 | 범위 |
|------|------|------|
| `pos` | 긍정 감성 비율 | 0 ~ 1 |
| `neg` | 부정 감성 비율 | 0 ~ 1 |
| `neu` | 중립 감성 비율 | 0 ~ 1 |
| `compound` | 종합 감성 점수 | -1 ~ +1 |

> **핵심 아이디어**: BERT가 놓치는 감성적 뉘앙스를 VADER의 명시적 감성 점수로 보완.
> hate speech는 높은 neg + 낮은 compound, offensive는 중간 수준의 부정성을 보임.

### 추가 실험

| 실험 | 설명 |
|------|------|
| **Freeze Study** | BERT+VADER encoder 동결 vs fine-tuning 비교 |
| **Hyperparameter Tuning** | lr → batch_size → dropout → epochs 순차 탐색 |
| **XAI** | SHAP + LIME → Overlap@5 분석, Before/After 케이스 비교 |

---

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 개별 단계 실행
./run.sh data          # 데이터 전처리 + 70/10/20 split
./run.sh vader         # VADER 감성 피처 추출
./run.sh tune          # 하이퍼파라미터 튜닝 (lr, batch, dropout, epochs)
./run.sh benchmark     # 반복 벤치마크 (seed 3회: 42, 52, 62)
./run.sh freeze-study  # encoder freeze vs fine-tuning 비교
./run.sh xai           # SHAP + LIME + Overlap@5 분석
./run.sh dashboard     # HTML 대시보드 생성

# 전체 실행
./run.sh all                # 전체 파이프라인 (튜닝 제외)
./run.sh all --with-tuning  # 튜닝 포함 전체 실행

# 유틸리티
./run.sh status        # 파이프라인 진행 상태 확인
./run.sh clean         # 산출물 초기화
```

---

## 파이프라인 흐름

```
data → vader → tune(선택) → benchmark → freeze-study → xai → dashboard
 │       │        │             │            │           │        │
 │       │        │             │            │           │        └─ HTML 대시보드 생성
 │       │        │             │            │           └─ SHAP/LIME 비교 분석
 │       │        │             │            └─ encoder 동결 실험
 │       │        │             └─ seed 3회 반복 벤치마크
 │       │        └─ 하이퍼파라미터 순차 탐색
 │       └─ VADER 감성 피처 추출 (4차원)
 └─ HateXplain 다운로드 + 전처리 + 분할
```

---

## 파일 구조

```
HateSpeachStudy/
├── run.sh                     # CLI 실행기 (메인 진입점)
├── run_fresh_full.sh          # 클린 재실행 스크립트
├── run_experiments.py         # 파이프라인 관제탑 (argparse 엔트리포인트)
├── experiment_core.py         # 핵심 실험 로직 (모델, 학습, 벤치마크, 튜닝)
├── experiment_xai.py          # XAI 분석 (SHAP, LIME, Overlap@5)
├── experiment_dashboard.py    # 인터랙티브 HTML 대시보드 생성기
├── utils.py                   # 공통 유틸리티 (시드, 디바이스, 시각화, I/O)
├── requirements.txt           # Python 의존성
├── README.md                  # 이 문서
├── .gitignore
│
├── docs/                      # 프로젝트 문서
│   ├── 파이프라인_상세분석.md
│   ├── 프로젝트개요_재작성_2버전.md
│   ├── 표절검증_최종보고서_v2.md
│   ├── 표절위험도_평가보고서.md
│   ├── 추가_논문 요약 및 리뷰.docx
│   └── 수행계획서.docx / .pdf
│
├── compat/                    # 레거시 호환용 래퍼 스크립트
│   ├── s00_data.py ~ s09_xai_phase2.py
│   └── stage01~04_*.py
│
├── data/                      # HateXplain 원본 (자동 다운로드)
├── checkpoints/               # 모델 체크포인트 (.pt)
└── outputs/
    ├── experiment_config.json # 실험 설정 (재현용)
    ├── data_splits.pkl        # 전처리된 분할 데이터
    ├── vader_features.pkl     # VADER 피처 캐시
    ├── reports/               # 벤치마크 요약, 데이터 프로파일
    ├── tuning/                # 하이퍼파라미터 탐색 결과
    ├── runs/                  # seed별 반복 실험 산출물
    ├── xai/                   # SHAP/LIME 분석 결과
    ├── dashboard/             # HTML 대시보드
    └── logs/                  # 실행 로그
```

---

## 주요 산출물

| 용도 | 파일 | 설명 |
|------|------|------|
| 재현 | `experiment_config.json` | 실험 설정 전체 기록 |
| 재현 | `data_splits.pkl`, `vader_features.pkl` | 전처리 캐시 |
| 벤치마크 | `reports/benchmark_summary.csv` | 모델별 성능 요약 (mean ± std) |
| 벤치마크 | `reports/model_comparison.png` | 성능 비교 막대 그래프 |
| 벤치마크 | `reports/per_class_f1_heatmap.png` | 클래스별 F1 히트맵 |
| Freeze | `reports/freeze_study.csv` | 동결 vs 미세조정 비교 |
| 튜닝 | `tuning/transformer_tuning_log.csv` | 탐색 이력 |
| XAI | `xai/xai_summary.md` | SHAP/LIME 분석 요약 |
| XAI | `xai/overlap_at_5.png` | Overlap@5 분포 시각화 |
| XAI | `xai/cases/` | 케이스별 SHAP attribution 비교 |
| 시각화 | `dashboard/index.html` | 인터랙티브 대시보드 (브라우저에서 바로 열기) |

---

## 핵심 클래스 & 함수

### 모델 클래스 (`experiment_core.py`)

| 클래스 | 역할 |
|--------|------|
| `TransformerCLSClassifier` | [CLS] → Dropout → Linear(768, 3) — baseline |
| `HybridSentimentClassifier` | [CLS](768d) + VADER(4d) → Linear(772, 256) → ReLU → Linear(256, 3) — improved |

### 주요 함수

| 함수 | 위치 | 역할 |
|------|------|------|
| `prepare_data()` | `experiment_core.py` | HateXplain 다운로드 + 분할 |
| `extract_vader_features()` | `experiment_core.py` | VADER 4차원 감성 피처 추출 |
| `train_neural_model()` | `experiment_core.py` | 단일 시드 학습 루프 (AdamW + warmup + early stopping) |
| `run_benchmark()` | `experiment_core.py` | TF-IDF + Transformer 전체 벤치마크 |
| `run_freeze_study()` | `experiment_core.py` | encoder 동결 비교 실험 |
| `run_hyperparameter_tuning()` | `experiment_core.py` | lr → batch → dropout → epochs 순차 탐색 |
| `run_xai()` | `experiment_xai.py` | SHAP/LIME 비교 분석 전체 파이프라인 |
| `run_dashboard()` | `experiment_dashboard.py` | HTML 대시보드 생성 |

---

## 학습 설정 (기본값)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `max_len` | 128 | 토큰 최대 길이 |
| `batch_size` | 64 | MPS에서 ~6GB 메모리 |
| `epochs` | 5 | 최대 에포크 (early stopping) |
| `learning_rate` | 2e-5 | BERT 표준 fine-tuning lr |
| `warmup_ratio` | 0.10 | 전체 스텝의 10% warmup |
| `weight_decay` | 0.01 | AdamW L2 정규화 |
| `dropout` | 0.10 | 분류 헤드 드롭아웃 |
| `mlp_hidden` | 256 | 하이브리드 MLP 은닉층 |
| `early_stopping_patience` | 2 | 개선 없이 허용 에포크 |
| `seeds` | [42, 52, 62] | 3회 반복 실험 시드 |

---

## 참고문헌

| 논문 | 역할 |
|------|------|
| Mathew et al. (2021) — HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection, AAAI | 데이터셋 |
| Davidson et al. (2017) — Automated Hate Speech Detection and the Problem of Offensive Language | 3-class 분류 체계 |
| Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers | Baseline 모델 |
| Liu et al. (2019) — RoBERTa: A Robustly Optimized BERT Pretraining Approach | Improved 모델 |
| Caselli et al. (2021) — HateBERT: Retraining BERT for Abusive Language Detection | 도메인 특화 모델 |
| Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions | SHAP |
| Ribeiro et al. (2016) — "Why Should I Trust You?": Explaining the Predictions of Any Classifier | LIME |
| Hutto & Gilbert (2014) — VADER: A Parsimonious Rule-based Model for Sentiment Analysis | 감성 피처 |

---

## License

This project is for academic purposes (빅데이터프로그래밍 수업 프로젝트).
