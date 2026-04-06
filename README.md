# Hate Speech Detection with XAI-Based Verification

> HateXplain 데이터셋 기반 혐오표현 탐지 파이프라인
> **가설 수립 → 실험적 검증 → XAI Before/After 해석**의 과학적 프레임워크

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

### Ablation (파라미터 수 통제)

| 모델 | 아키텍처 | 클래스 |
|------|---------|--------|
| BERT + MLP | `[CLS](768d)` → MLP(256) → ReLU → 3-class (**VADER 없이 동일 MLP**) | `TransformerMLPClassifier` |

> **왜 필요한가?** BERT+VADER의 MLP 헤드(~198K params)가 BERT-base의 Linear 헤드(~2.3K params)보다 86배 크다.
> BERT+MLP은 동일한 MLP를 사용하되 VADER를 제거하여, 성능 향상이 MLP 크기 때문인지 VADER 때문인지 분리한다.

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

## 빠른 시작 (Quick Start)

### 1. 대시보드만 바로 보기 (실험 결과 포함됨)

```bash
git clone https://github.com/WinterFlw/Big_data_Programming.git
cd Big_data_Programming
pip install fastapi uvicorn
python3 dashboard_app.py
# 브라우저에서 http://localhost:8501 접속
```

> `outputs/`와 `data/`가 git에 포함되어 있어 **clone 직후 18개 탭 대시보드가 바로 작동**합니다.
> Playground(실시간 모델 추론)만 체크포인트가 필요합니다 (아래 참조).

---

### 2. 모델 체크포인트 다운로드 (Playground용)

체크포인트는 용량 문제로 git에 포함되지 않습니다.
Playground 탭(실시간 추론, Attention Heatmap, LIME 분석)을 사용하려면 아래 절차를 따르세요.

> **Playground가 필요 없다면 이 단계는 건너뛰세요.**
> 나머지 17개 탭은 체크포인트 없이 정상 작동합니다.

**Google Drive 다운로드:**

| 파일 | 내용 | 크기 |
|------|------|------|
| `models.zip` | 최적 모델 4개 (BERT-base, BERT+MLP, BERT+VADER, RoBERTa+VADER) | ~1.7GB |

> 다운로드 링크: **[Google Drive에서 다운로드](https://drive.google.com/file/d/1wS6Qcc7NLyajC-fXQeK6zpZXSffnoQQB/view?usp=sharing)**

**압축 해제 방법:**

```bash
cd Big_data_Programming

# 1. 다운로드한 models.zip을 프로젝트 루트에 복사
# 2. 압축 해제
unzip models.zip -d models/

# 3. 확인 — 아래 4개 파일이 있으면 OK
ls -lh models/
# bert_base_seed_42.pt      (418MB)  BERT-base
# bert_mlp_seed_42.pt       (418MB)  BERT+MLP (Ablation)
# bert_vader_seed_42.pt     (418MB)  BERT+VADER
# roberta_vader_seed_42.pt  (476MB)  RoBERTa+VADER (Best)

# 4. 대시보드 실행 — Playground 포함 전체 작동
python3 dashboard_app.py
```

**폴더 구조 (압축 해제 후):**

```
Big_data_Programming/
├── models/                          ← 여기에 압축 해제
│   ├── bert_base_seed_42.pt
│   ├── bert_mlp_seed_42.pt
│   ├── bert_vader_seed_42.pt
│   └── roberta_vader_seed_42.pt
├── outputs/                         ← git에 포함됨
├── data/                            ← git에 포함됨
├── dashboard_app.py
└── ...
```

> `models/` 폴더가 없으면 `checkpoints/` 폴더를 자동으로 탐색합니다.
> 전체 파이프라인을 처음부터 돌린 경우(`./run.sh full`) `checkpoints/`에 자동 생성되므로
> 별도 다운로드가 필요 없습니다.

<details>
<summary><b>체크포인트 직접 생성하는 방법 (4-5시간 소요)</b></summary>

```bash
pip install -r requirements.txt
./run.sh full    # data → vader → eda → tune → benchmark → freeze → xai → dashboard
# checkpoints/ 디렉토리에 41개 .pt 파일이 자동 생성됩니다
# dashboard_app.py가 checkpoints/를 자동으로 찾습니다
```

</details>

---

### 3. 전체 파이프라인 재실행 (처음부터)

```bash
# 의존성 설치
pip install -r requirements.txt

# 개별 단계 실행
./run.sh data          # 데이터 전처리 + 70/10/20 split
./run.sh vader         # VADER 감성 피처 추출
./run.sh eda           # 탐색적 데이터 분석 (텍스트 길이, VADER 분포, 타겟, 어휘 겹침)
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

> 전체 파이프라인 소요 시간: **약 4-5시간** (M3 Max 64GB, MPS 기준)
> 튜닝(~2h)과 벤치마크(~2.5h)가 전체의 90%를 차지합니다.

---

## 파이프라인 흐름

```
data → vader → eda → tune(선택) → benchmark → freeze-study → xai → dashboard
 │       │      │        │            │            │           │        │
 │       │      │        │            │            │           │        └─ HTML 대시보드
 │       │      │        │            │            │           └─ SHAP/LIME Before/After
 │       │      │        │            │            └─ encoder 동결 실험
 │       │      │        │            └─ seed 3회 벤치마크 + 통계 검정
 │       │      │        └─ 하이퍼파라미터 순차 탐색
 │       │      └─ 텍스트 길이/VADER 분포/타겟/어휘 분석
 │       └─ VADER 감성 피처 추출 (4차원)
 └─ HateXplain 다운로드 + 전처리 + 분할
```

---

## 파일 구조

```
HateSpeachStudy/
├── CLAUDE.md                  # 하네스: AI 에이전트 규칙 (코드/서술/git 규칙)
├── progress.md                # 하네스: 파이프라인 진행 상태 + Todo
├── architecture.md            # 하네스: 시스템 구조도 + 데이터 흐름
│
├── run.sh                     # CLI 실행기 (메인 진입점)
├── run_fresh_full.sh          # 클린 재실행 스크립트
├── run_experiments.py         # 파이프라인 관제탑 (argparse 엔트리포인트)
├── experiment_core.py         # 핵심 실험 로직 (모델, 학습, 벤치마크, 튜닝)
├── experiment_eda.py          # 탐색적 데이터 분석 (텍스트 길이, VADER 분포, 타겟)
├── experiment_xai.py          # XAI 분석 (SHAP, LIME, Overlap@5)
├── experiment_dashboard.py    # 정적 HTML 대시보드 생성기
├── dashboard_app.py           # FastAPI 대시보드 서버 (18탭, 실시간 Playground)
├── utils.py                   # 공통 유틸리티 (시드, 디바이스, 시각화, I/O)
├── requirements.txt           # Python 의존성
├── README.md                  # 이 문서
├── .gitignore
│
├── docs/                      # 프로젝트 문서
│   ├── 파이프라인_상세분석.md
│   ├── 파이프라인_검증_및_팀분업_가이드.md
│   ├── 참고자료_종합_가이드.md
│   ├── 하네스_엔지니어링_적용사례.md
│   ├── 프로젝트개요_재작성_2버전.md
│   ├── 표절검증_최종보고서_v2.md
│   ├── 표절위험도_평가보고서.md
│   ├── 추가_논문 요약 및 리뷰.docx
│   └── 수행계획서.docx / .pdf
│
│
├── data/                      # HateXplain 원본 (git 포함, 12MB)
├── models/                    # Playground용 최적 체크포인트 4개 (git 제외, 1.7GB -- Google Drive)
├── checkpoints/               # 전체 체크포인트 41개 (git 제외, 17GB -- ./run.sh full로 생성)
└── outputs/                   # 실험 결과 (git 포함, 38MB)
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

## 대시보드 (18탭)

`python3 dashboard_app.py`로 실행하면 http://localhost:8501 에서 아래 탭을 볼 수 있습니다.

| # | 탭 | 내용 | 체크포인트 필요 |
|---|-----|------|:---:|
| 1 | Overview | KPI 카드, 파이프라인 흐름도, 주요 발견 | |
| 2 | Pipeline Deep-Dive | 8단계 연구 방법론 스토리 (가설→XAI) | |
| 3 | E2E Pipeline | 기술적 데이터 흐름, 차원 변환, 시간 프로파일링 | |
| 4 | Benchmark | F1 바 차트, 레이더, 클래스별 성능 | |
| 5 | Statistical Tests | P-value 히트맵, 유의성 테이블 | |
| 6 | Tuning | LR/Dropout 탐색 궤적 차트 | |
| 7 | Learning Curves | 에폭별 학습곡선 (모델/시드 선택) | |
| 8 | Freeze Study | Frozen vs Fine-tuned 비교 | |
| 9 | EDA | VADER 분포, 타겟 커뮤니티, 어휘 중첩 | |
| 10 | XAI Analysis | Overlap@5, 혼동행렬 비교 | |
| 11 | XAI Cases | 개별 샘플 SHAP/LIME 갤러리 | |
| 12 | Error Analysis | 오분류 패턴, VADER 맹점, Ablation 다이어그램 | |
| 13 | Architecture | 3개 모델 구조 시각화 | |
| 14 | Data Explorer | 데이터셋 검색/필터 브라우저 | |
| 15 | Comparison | 2모델 레이더 차트 + Delta 분석 | |
| 16 | Report | 자동 보고서 생성 + 클립보드 복사 | |
| 17 | References | 7개 핵심 문헌 + 재현성 가이드 | |
| 18 | **Playground** | **Attention Heatmap + LIME + 4모델 실시간 추론** | **필요** |

> 한국어/영어 전환, 다크/라이트 테마, 이미지 클릭 확대, PDF Export 지원

---

## 핵심 클래스 & 함수

### 모델 클래스 (`experiment_core.py`)

| 클래스 | 역할 |
|--------|------|
| `TransformerCLSClassifier` | [CLS] → Dropout → Linear(768, 3) — baseline |
| `TransformerMLPClassifier` | [CLS] → Linear(768, 256) → ReLU → Linear(256, 3) — ablation (MLP 크기 통제) |
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
| `run_eda()` | `experiment_eda.py` | 탐색적 데이터 분석 (텍스트 길이, VADER, 타겟, 어휘) |
| `run_xai()` | `experiment_xai.py` | SHAP/LIME 비교 분석 전체 파이프라인 |
| `run_dashboard()` | `experiment_dashboard.py` | HTML 대시보드 생성 |
| `compute_pairwise_significance()` | `utils.py` | paired t-test + Cohen's d 통계 검정 |

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

## 연구 투명성: 한계점 및 Future Work

본 프로젝트는 **가설 기반 실험 → 통제된 비교 → XAI 사후 검증** 구조를 따릅니다.
아래는 연구 설계의 투명한 한계점과 향후 개선 방향입니다.

### 설계 결정에 대한 투명한 논의

| 항목 | 현재 접근 | 한계/비고 |
|------|----------|----------|
| **XAI 역할** | 사전 가설의 사후 검증 (Before/After) | 진정한 피드백 루프가 아닌, 가설 검증 도구로 활용 |
| **VADER 선택 근거** | 감성 강도 차이 가설 (EDA로 실증) | hate/offensive 구분의 본질은 대상 집단 지향성 (Davidson et al., 2017) — VADER는 이를 직접 포착하지 못함 |
| **파라미터 수 통제** | BERT+MLP ablation으로 분리 | Ablation 추가로 교란 변수 대응 완료 |
| **SHAP의 VADER 설명** | 텍스트 입력만 perturbation | SHAP이 VADER 4차원의 기여도를 직접 산출하지 못하는 구조적 한계 |
| **통계적 검정** | paired t-test + Cohen's d | 3-seed 반복은 검정력이 낮음 — 유의하지 않아도 "차이 없음"이 아닐 수 있음 |
| **Overlap@5** | K=5, threshold=0.6 | 추가적인 K 민감도 분석이 필요 |

### Future Work

- **Target community features**: HateXplain의 target 정보를 모델 입력으로 활용
- **HateBERT / TweetBERT**: 도메인 특화 사전학습 모델과의 비교
- **SHAP에 VADER 포함**: feature-level SHAP으로 VADER 기여도 직접 측정
- **더 많은 seed 반복** (5~10회): 통계적 검정력 강화
- **Overlap@K 민감도 분석**: K=3, 5, 10에서의 결과 비교

---

## License

This project is for academic purposes (빅데이터프로그래밍 수업 프로젝트).
