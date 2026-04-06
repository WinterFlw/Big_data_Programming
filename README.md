# Hate Speech Detection with XAI-Based Verification

> HateXplain 데이터셋 기반 혐오표현 탐지 파이프라인
> **"가설 수립 → 실험적 검증 → XAI 사후 검증"** 의 과학적 검증 프레임워크

한성대학교 빅데이터프로그래밍 수업 프로젝트

---

## 빠른 시작

### 1단계: 대시보드 바로 보기

```bash
git clone https://github.com/WinterFlw/Big_data_Programming.git
cd Big_data_Programming
pip install fastapi uvicorn
python3 dashboard_app.py
```

브라우저에서 **http://localhost:8501** 접속하면 18개 탭 대시보드가 즉시 작동합니다.
`outputs/`(38MB)와 `data/`(12MB)가 git에 포함되어 있어 별도 데이터 준비가 필요 없습니다.

<details>
<summary><strong>대시보드 실행 옵션 상세</strong></summary>

**기본 실행:**
```bash
python3 dashboard_app.py          # http://localhost:8501 (기본)
```

**포트/호스트 변경 (uvicorn 직접 호출):**
```bash
# 다른 포트에서 실행
uvicorn dashboard_app:app --host 0.0.0.0 --port 9000

# 자동 리로드 (개발 모드)
uvicorn dashboard_app:app --host 0.0.0.0 --port 8501 --reload
```

**백그라운드 실행:**
```bash
nohup python3 dashboard_app.py > dashboard.log 2>&1 &
echo $!  # PID 확인
# 종료: kill $(lsof -t -i:8501)
```

**필수 의존성:**
```bash
pip install fastapi uvicorn           # 최소 (대시보드만)
pip install -r requirements.txt       # 전체 (Playground LIME/SHAP 포함)
```

| 의존성 | 용도 | 필수 여부 |
|--------|------|:---------:|
| `fastapi`, `uvicorn` | 웹 서버 | 필수 |
| `torch`, `transformers` | Playground 모델 추론 | Playground 사용 시 |
| `lime` | LIME 해석 | Playground LIME 버튼 사용 시 |
| `shap` | SHAP 해석 | XAI 탭 활용 시 |
| `vaderSentiment` | VADER 감성 피처 | VADER 모델 Playground 시 |

> Playground 관련 패키지가 없어도 대시보드 자체는 정상 실행됩니다.
> 해당 탭 접근 시에만 동적으로 import하며, 없으면 안내 메시지를 표시합니다.

</details>

### 2단계: Playground 활성화 (선택)

Playground 탭(실시간 모델 추론 + Attention Heatmap + LIME)을 사용하려면
학습된 모델 체크포인트가 필요합니다.

**다운로드:** [Google Drive에서 models.zip 다운로드 (1.7GB)](https://drive.google.com/file/d/1wS6Qcc7NLyajC-fXQeK6zpZXSffnoQQB/view?usp=sharing)

```bash
# 다운로드한 models.zip을 프로젝트 루트로 이동 후:
unzip models.zip -d models/
```

압축 해제하면 아래 4개 파일이 생성됩니다:

```
models/
├── bert_base_seed_42.pt      (418MB)  BERT-base
├── bert_mlp_seed_42.pt       (418MB)  BERT+MLP (Ablation 통제)
├── bert_vader_seed_42.pt     (418MB)  BERT+VADER
└── roberta_vader_seed_42.pt  (476MB)  RoBERTa+VADER (최고 성능)
```

> `models/` 폴더가 없으면 `checkpoints/` 폴더를 자동 탐색합니다.
> Playground가 필요 없다면 이 단계를 건너뛰세요 — 나머지 17개 탭은 체크포인트 없이 정상 작동합니다.

### 3단계: 전체 파이프라인 재실행 (선택, 4-5시간)

처음부터 모든 실험을 재현하려면:

```bash
pip install -r requirements.txt

# 전체 실행 (튜닝 포함)
./run.sh all --with-tuning

# 또는 단계별 실행
./run.sh data          # 데이터 전처리 + stratified split
./run.sh vader         # VADER 감성 피처 추출
./run.sh eda           # 탐색적 데이터 분석
./run.sh tune          # 하이퍼파라미터 튜닝 (~2시간)
./run.sh benchmark     # 6모델 x 3시드 벤치마크 (~2.5시간)
./run.sh freeze-study  # 인코더 동결 비교
./run.sh xai           # SHAP + LIME 분석 (~30분)
./run.sh dashboard     # 정적 HTML 대시보드 생성
```

---

## 실험 결과 요약

### 모델 성능 (Macro F1, 3시드 평균)

| 순위 | 모델 | Macro F1 | Hate F1 | Offensive F1 | Normal F1 |
|:----:|------|:--------:|:-------:|:------------:|:---------:|
| **1** | **RoBERTa+VADER** | **0.6863 ± 0.0023** | **0.775** | **0.547** | **0.737** |
| 2 | BERT+MLP | 0.6810 ± 0.0021 | 0.772 | 0.538 | 0.733 |
| 3 | BERT+VADER | 0.6794 ± 0.0034 | 0.772 | 0.534 | 0.732 |
| 4 | BERT-base | 0.6744 ± 0.0019 | 0.766 | 0.530 | 0.727 |
| 5 | TF-IDF+SVM | 0.6393 | 0.737 | 0.487 | 0.694 |
| 6 | TF-IDF+LR | 0.6370 | 0.735 | 0.481 | 0.695 |

### 통계적 유의성 (Paired t-test)

| 비교 | p-value | Cohen's d | 유의성 |
|------|:-------:|:---------:|:------:|
| RoBERTa+VADER vs BERT-base | **0.0366** | -2.93 | **유의** |
| BERT+VADER vs BERT-base | 0.1384 | - | 비유의 |
| BERT+MLP vs BERT+VADER | 0.5666 | - | 비유의 |

### 핵심 발견

1. **RoBERTa+VADER가 최고 성능** — BERT-base 대비 +1.19%p, 통계적으로 유의 (p=0.037)
2. **VADER 단독 효과는 제한적** — 같은 인코더(BERT) 기반에서 VADER 추가 시 p=0.138 (비유의)
3. **인코더 품질이 핵심** — Freeze Study에서 동결 시 F1=0.324 (random 수준), 미세조정 시 F1=0.679 (+109%)
4. **Ablation 확인** — BERT+MLP vs BERT+VADER (p=0.567): MLP 구조가 아닌 인코더 사전학습이 결정 요인
5. **hate/offensive 경계 모호** — 어휘 Jaccard 유사도 0.71, offensive F1 전 모델 0.53~0.55

### 결론

> 가설 "VADER 감성 점수가 혐오표현 탐지를 향상시킨다"는 **부분 채택**.
> VADER는 강력한 인코더(RoBERTa)와 결합할 때만 유의미한 시너지를 만들며,
> 같은 인코더(BERT)에서는 효과가 미미하다. **인코더 사전학습 품질이 가장 중요한 변수**이다.

---

## 핵심 목표

1. BERT 계열 모델의 **hate / offensive 오분류 패턴**을 EDA + XAI로 분석
2. **VADER 감성 피처** 결합 + **Ablation Study**(BERT+MLP)로 개선 요인 분리
3. 개선 전후 XAI 비교로 **Before/After 차이를 정량적으로 검증** (Overlap@5 + 통계 검정)

---

## 실험 환경

| 항목 | 스펙 |
|------|------|
| **Hardware** | Apple M3 Max, 64GB Unified Memory |
| **Backend** | PyTorch MPS (Metal Performance Shaders) |
| **Python** | 3.13 + PyTorch 2.10 + Transformers 5.3 |
| **XAI** | SHAP (CPU only, MPS 비호환) + LIME |
| **대시보드** | FastAPI + Chart.js (dashboard_app.py) |
| **총 실행 시간** | ~4-5시간 (튜닝 2h + 벤치마크 2.5h + XAI 30min) |

---

## 데이터셋

[HateXplain](https://github.com/hate-alert/HateXplain) (Mathew et al., AAAI 2021)

| 항목 | 내용 |
|------|------|
| 출처 | Twitter + Gab, 20,148건 영어 텍스트 |
| 분류 | 3-class: `hatespeech` / `offensive` / `normal` |
| 라벨링 | 3인 다수결 투표, undecided 제외 → **~13,433건** 사용 |
| 분할 | **60 / 20 / 20** stratified split (train / val / test) |
| 시드 | 42, 52, 62 (3회 반복) |

---

## 모델 구성 (6개 + Freeze Study)

### Baselines

| 모델 | 설명 | 클래스 |
|------|------|--------|
| TF-IDF + LR | Logistic Regression (1~3gram) | `sklearn` |
| TF-IDF + SVM | LinearSVC + CalibratedClassifierCV | `sklearn` |
| BERT-base | `bert-base-uncased` [CLS] → Linear → 3-class | `TransformerCLSClassifier` |

### Ablation (통제 모델)

| 모델 | 아키텍처 | 목적 |
|------|---------|------|
| BERT+MLP | [CLS](768d) → MLP(256) → ReLU → 3-class | VADER 없이 동일 MLP로 성능 향상 원인 분리 |

### Improved (핵심 가설)

| 모델 | 아키텍처 | 클래스 |
|------|---------|--------|
| BERT+VADER | [CLS](768d) + VADER(4d) → MLP(256) → 3-class | `HybridSentimentClassifier` |
| **RoBERTa+VADER** | [CLS](768d) + VADER(4d) → MLP(256) → 3-class | `HybridSentimentClassifier` |

### VADER 감성 피처 (4차원)

| 피처 | 설명 | 범위 |
|------|------|------|
| `compound` | 종합 감성 점수 | -1 ~ +1 |
| `pos` / `neg` / `neu` | 긍정/부정/중립 감성 비율 | 0 ~ 1 |

> **핵심 아이디어**: 768차원 [CLS] 벡터에 단 4차원(0.5%)의 감성 신호를 추가.
> 강력한 인코더(RoBERTa)와 결합할 때 통계적으로 유의미한 시너지를 만든다.

---

## 대시보드 (18탭)

`python3 dashboard_app.py` → http://localhost:8501

| # | 탭 | 내용 | 체크포인트 |
|---|-----|------|:---:|
| 1 | Overview | KPI 카드, 파이프라인 흐름도, 주요 발견 | |
| 2 | Pipeline Deep-Dive | 8단계 연구 방법론 스토리 (가설→XAI 사후검증) | |
| 3 | E2E Pipeline | 기술적 데이터 흐름, 차원 변환, 시간 프로파일링 | |
| 4 | Benchmark | F1 바 차트, 레이더, 클래스별 성능 테이블 | |
| 5 | Statistical Tests | P-value 히트맵, 유의성 검정 테이블 | |
| 6 | Tuning | LR/Dropout 탐색 궤적 차트 | |
| 7 | Learning Curves | 에폭별 학습곡선 (모델/시드 선택) | |
| 8 | Freeze Study | Frozen vs Fine-tuned 인코더 비교 | |
| 9 | EDA | VADER 분포, 타겟 커뮤니티, 어휘 중첩도 | |
| 10 | XAI Analysis | Overlap@5, 혼동행렬 비교 | |
| 11 | XAI Cases | 개별 샘플 SHAP/LIME 갤러리 | |
| 12 | Error Analysis | 오분류 패턴, VADER 맹점, Ablation 다이어그램, 한계점 | |
| 13 | Architecture | 3개 모델 구조 시각화 | |
| 14 | Data Explorer | 데이터셋 검색/필터 브라우저 | |
| 15 | Comparison | 2모델 레이더 차트 + Delta 분석 | |
| 16 | Report | 자동 보고서 생성 + 클립보드 복사 | |
| 17 | References | 7개 핵심 문헌 + 재현성 가이드 + 학습 시간 | |
| 18 | **Playground** | **Attention Heatmap + LIME + 4모델 실시간 추론** | **필요** |

> 한국어/영어 전환 | 다크/라이트 테마 | 이미지 클릭 확대 | PDF Export

---

## 파이프라인 흐름

```
data → vader → eda → tune(선택) → benchmark → freeze-study → xai → dashboard
 │       │      │        │            │            │           │        │
 │       │      │        │            │            │           │        └─ FastAPI 대시보드
 │       │      │        │            │            │           └─ SHAP/LIME Before/After
 │       │      │        │            │            └─ encoder 동결 vs 미세조정
 │       │      │        │            └─ 6모델 x 3시드 벤치마크 + paired t-test
 │       │      │        └─ lr / dropout / batch / epochs 순차 탐색
 │       │      └─ 텍스트 길이 / VADER 분포 / 타겟 / 어휘 중첩 분석
 │       └─ VADER 감성 피처 추출 (4차원)
 └─ HateXplain 다운로드 + majority vote + stratified split
```

---

## 파일 구조

```
Big_data_Programming/
│
├── README.md                  # 이 문서
├── CLAUDE.md                  # AI 에이전트 규칙 (하네스 엔지니어링)
├── requirements.txt           # Python 의존성
├── .gitignore
│
├── run.sh                     # CLI 실행기 (메인 진입점)
├── run_experiments.py         # 파이프라인 관제탑 (argparse)
├── experiment_core.py         # 핵심 실험 (모델, 학습, 벤치마크, 튜닝)
├── experiment_eda.py          # 탐색적 데이터 분석
├── experiment_xai.py          # XAI 분석 (SHAP, LIME, Overlap@5)
├── experiment_dashboard.py    # 정적 HTML 대시보드 생성기
├── dashboard_app.py           # FastAPI 대시보드 서버 (18탭 + Playground)
├── utils.py                   # 공통 유틸리티
│
├── data/                      # HateXplain 원본 데이터 (git 포함, 12MB)
│   ├── dataset.json
│   └── post_id_divisions.json
│
├── models/                    # Playground용 최적 체크포인트 (git 제외, 1.7GB)
│   ├── bert_base_seed_42.pt           Google Drive에서 다운로드
│   ├── bert_mlp_seed_42.pt            ↓
│   ├── bert_vader_seed_42.pt          ↓
│   └── roberta_vader_seed_42.pt       ↓
│
├── checkpoints/               # 전체 체크포인트 (git 제외, 17GB, ./run.sh full로 생성)
│
├── outputs/                   # 실험 결과 (git 포함, 38MB)
│   ├── reports/               #   벤치마크 요약, 통계 검정, EDA, freeze study
│   ├── tuning/                #   하이퍼파라미터 탐색 이력
│   ├── runs/                  #   시드별 학습 로그, confusion matrix
│   ├── xai/                   #   SHAP/LIME 분석, 케이스 이미지
│   ├── dashboard/             #   정적 HTML 대시보드
│   └── logs/                  #   실행 로그
│
└── docs/                      # 프로젝트 문서 (한국어)
    ├── 파이프라인_상세분석.md
    ├── 표절검증_최종보고서_v2.md
    └── ...
```

### git에 포함되는 것 / 제외되는 것

| 항목 | 크기 | git | 비고 |
|------|------|:---:|------|
| 소스 코드 (*.py, *.sh) | ~200KB | O | |
| outputs/ (CSV, JSON, PNG) | 38MB | O | 대시보드 즉시 작동 |
| data/ (dataset.json) | 12MB | O | Data Explorer 작동 |
| models/ (최적 .pt 4개) | 1.7GB | **X** | [Google Drive](https://drive.google.com/file/d/1wS6Qcc7NLyajC-fXQeK6zpZXSffnoQQB/view?usp=sharing) |
| checkpoints/ (전체 .pt 41개) | 17GB | **X** | `./run.sh full`로 재생성 |

---

## 학습 설정

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `max_len` | 128 | 토큰 최대 길이 (EDA 결과 99%+ 미잘림) |
| `batch_size` | 64 | MPS 메모리 30-48GB 소비 |
| `epochs` | 5 | 최대 에폭 (early stopping patience=2) |
| `learning_rate` | 2e-5 | 튜닝으로 확인된 최적값 |
| `dropout` | 0.10 | 전 모델 최적 (짧은 학습에 강한 정규화 불필요) |
| `warmup_ratio` | 0.10 | 전체 스텝의 10% warmup |
| `weight_decay` | 0.01 | AdamW L2 정규화 |
| `mlp_hidden` | 256 | 하이브리드/MLP 은닉층 |
| `seeds` | [42, 52, 62] | 3회 반복 (paired t-test 최소 요건) |

---

## 핵심 클래스 & 함수

### 모델 클래스 (`experiment_core.py`)

| 클래스 | 아키텍처 | 역할 |
|--------|---------|------|
| `TransformerCLSClassifier` | [CLS] → Dropout → Linear(768→3) | Baseline |
| `TransformerMLPClassifier` | [CLS] → MLP(768→256) → ReLU → 3 | Ablation 통제 |
| `HybridSentimentClassifier` | [CLS]+VADER → MLP(772→256) → 3 | 핵심 가설 |

### 주요 함수

| 함수 | 위치 | 역할 |
|------|------|------|
| `prepare_data()` | `experiment_core.py` | HateXplain 다운로드 + 분할 |
| `extract_vader_features()` | `experiment_core.py` | VADER 4차원 감성 피처 추출 |
| `train_neural_model()` | `experiment_core.py` | 단일 시드 학습 (AdamW + warmup + early stopping) |
| `run_benchmark()` | `experiment_core.py` | 6모델 전체 벤치마크 |
| `run_freeze_study()` | `experiment_core.py` | 인코더 동결 비교 실험 |
| `run_hyperparameter_tuning()` | `experiment_core.py` | lr → batch → dropout → epochs 순차 탐색 |
| `run_eda()` | `experiment_eda.py` | 탐색적 데이터 분석 |
| `run_xai()` | `experiment_xai.py` | SHAP/LIME 비교 분석 |
| `compute_pairwise_significance()` | `utils.py` | paired t-test + Cohen's d |

---

## 참고문헌

| 논문 | 역할 |
|------|------|
| Mathew et al. (2021) — HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection, AAAI | 데이터셋 + XAI 평가 프레임워크 |
| **Cheng (2022)** — Towards Explainable and Adaptive Sentiment-enhanced Hate Speech Detection, Virginia Tech | 감성 기반 혐오표현 탐지 선행 연구. **차별점: Cheng은 XAI 진단→개선 순환, 본 연구는 사전 가설→사후 검증** |
| Davidson et al. (2017) — Automated Hate Speech Detection and the Problem of Offensive Language | 3-class 분류 체계 |
| Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers, NAACL | Baseline 인코더 |
| Liu et al. (2019) — RoBERTa: A Robustly Optimized BERT Pretraining Approach | 강화된 사전학습 |
| Hutto & Gilbert (2014) — VADER: A Parsimonious Rule-based Model for Sentiment Analysis, ICWSM | 감성 피처 |
| Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions, NeurIPS | SHAP |
| Ribeiro et al. (2016) — "Why Should I Trust You?", KDD | LIME |

---

## 한계점 및 Future Work

본 프로젝트는 **사전 가설 → 통제된 실험 → XAI 사후 검증** 구조를 따릅니다.
아래는 투명하게 밝히는 한계점과 향후 개선 방향입니다.

| 항목 | 현재 접근 | 한계 |
|------|----------|------|
| **XAI 역할** | 사후 검증 도구 | 피드백 루프가 아닌 가설 검증 도구로만 활용 |
| **VADER 선택** | 선행 연구(Cheng 2022) 기반 사전 가설 | hate/offensive 구분의 본질은 대상 집단 지향성 — VADER로 직접 포착 불가 |
| **통계 검정** | paired t-test (n=3) | 3개 시드는 검정력이 낮음 |
| **SHAP 구조** | 텍스트 입력만 perturbation | VADER 4차원의 기여도를 직접 산출하지 못함 |
| **offensive F1** | 전 모델 0.53~0.55 | 어휘 Jaccard 0.71의 구조적 한계 |

### Future Work

- 시드 5~10회로 확장하여 통계적 검정력 강화
- HateBERT / TweetBERT 등 도메인 특화 모델 비교
- HateXplain target 정보를 모델 입력으로 활용
- feature-level SHAP으로 VADER 기여도 직접 측정
- Overlap@K 민감도 분석 (K=3, 5, 10)
- 한국어/다국어 혐오표현 데이터셋 확장

---

## License

This project is for academic purposes (한성대학교 빅데이터프로그래밍 수업 프로젝트).
HateXplain dataset: CC-BY 4.0 | BERT/RoBERTa: Apache 2.0 | VADER: MIT
