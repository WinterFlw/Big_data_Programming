# Hate Speech Detection with XAI-Driven Improvement

HateXplain 데이터셋 기반 혐오표현 탐지 파이프라인.
**XAI 진단 → 피처 보강 → XAI 재진단**의 순환적 프레임워크를 구현합니다.

## 핵심 목표

1. BERT 계열 모델의 hate/offensive 오분류 원인을 SHAP + LIME으로 진단
2. VADER 감성 피처 결합으로 감성 맥락 보강
3. 개선 전후 XAI 비교로 **왜 나아졌는지** 정량적으로 설명

## 실험 환경

- **Hardware**: Apple M3 Max, 64GB Unified Memory
- **Backend**: PyTorch MPS (Metal Performance Shaders)
- **Python**: 3.13 + PyTorch 2.10 + Transformers 5.3

## 데이터셋

[HateXplain](https://github.com/hate-alert/HateXplain) (Mathew et al., AAAI 2021)
- Twitter + Gab, 20,148건 영어 텍스트
- 3-class: hatespeech / offensive / normal
- 다수결 라벨링, undecided 제외 → ~13,433건 사용
- 70 / 10 / 20 stratified split

## 모델 구성

### Baselines
| 모델 | 설명 |
|------|------|
| TF-IDF + LR/SVM | 전통 ML 베이스라인 |
| BERT-base | `bert-base-uncased` fine-tuning |

### Improved (VADER Hybrid)
| 모델 | 아키텍처 |
|------|---------|
| BERT + VADER | `[CLS](768d) + VADER(4d)` → MLP(256) → 3-class |
| RoBERTa + VADER | `[CLS](768d) + VADER(4d)` → MLP(256) → 3-class |

### VADER 감성 피처 (4차원)
- `pos`, `neg`, `neu`: 감성 비율 (0~1)
- `compound`: 종합 감성 점수 (-1~+1)

### 추가 실험
- **Freeze Study**: BERT+VADER encoder freeze vs fine-tuning 비교
- **Hyperparameter Tuning**: lr, batch_size, dropout, epochs 순차 탐색
- **XAI**: SHAP + LIME + Overlap@5 분석

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 개별 단계 실행
./run.sh data          # 데이터 전처리 + 70/10/20 split
./run.sh vader         # VADER 감성 피처 추출
./run.sh tune          # 하이퍼파라미터 튜닝
./run.sh benchmark     # 반복 벤치마크 (seed 3회)
./run.sh freeze-study  # freeze vs fine-tuning
./run.sh xai           # SHAP + LIME + Overlap@5
./run.sh dashboard     # HTML 대시보드 생성

# 전체 실행
./run.sh all
./run.sh all --with-tuning  # 튜닝 포함

# 상태 확인
./run.sh status
```

## 파일 구조

```
HateSpeachStudy/
├── run.sh                    # CLI 실행기
├── run_fresh_full.sh         # 클린 재실행
├── run_experiments.py        # 엔트리포인트
├── experiment_core.py        # 핵심 실험 로직
├── experiment_xai.py         # XAI 분석 (SHAP, LIME)
├── experiment_dashboard.py   # HTML 대시보드 생성
├── utils.py                  # 공통 유틸리티
├── requirements.txt
├── s00~s09_*.py              # 호환용 래퍼
├── stage01~04_*.py           # 호환용 래퍼
│
├── data/                     # HateXplain 원본 (자동 생성)
├── checkpoints/              # 모델 체크포인트 (.pt)
└── outputs/
    ├── reports/              # 데이터 프로파일, 벤치마크 요약
    ├── tuning/               # 하이퍼파라미터 탐색 결과
    ├── runs/                 # seed별 반복 실험 결과
    ├── xai/                  # SHAP/LIME 분석 결과
    ├── dashboard/            # HTML 대시보드
    └── logs/                 # 실행 로그
```

## 주요 산출물

| 용도 | 파일 |
|------|------|
| 재현 | `outputs/experiment_config.json`, `data_splits.pkl`, `vader_features.pkl` |
| 보고서 | `reports/benchmark_summary.md`, `reports/model_comparison.png` |
| XAI | `xai/xai_summary.md`, `xai/overlap_at_5.png`, `xai/cases/` |
| 시각화 | `dashboard/index.html` (브라우저에서 바로 열기) |

## 참고문헌

- Mathew et al. (2021) — HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection, AAAI
- Davidson et al. (2017) — Automated Hate Speech Detection and the Problem of Offensive Language
- Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers
- Caselli et al. (2021) — HateBERT: Retraining BERT for Abusive Language Detection
- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP)
- Ribeiro et al. (2016) — "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)
- Hutto & Gilbert (2014) — VADER: A Parsimonious Rule-based Model for Sentiment Analysis
