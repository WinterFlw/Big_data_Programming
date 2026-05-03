# Hate Speech Detection with XAI-Based Verification

> HateXplain 데이터셋 기반 혐오표현 탐지 파이프라인
> **"선행연구 기반 가설 → 통제된 ablation → XAI 사후 검증"** 의 과학적 검증 프레임워크

한성대학교 빅데이터프로그래밍 수업 프로젝트

> ⚠ **현재 상태 (2026-04-30)**:
> - **1차 파이프라인 (baseline, 2026-04-11 완료)**: 6모델 + XAI 3축, Macro F1 0.6822 — 본 README의 "실험 결과 요약" 참조
> - **v2.1 (진행 중)**: 8조건 풀 ablation (BERT × 4 + RoBERTa × 4) + XAI 4축 (CI/IS/MSS 자동 메트릭) + 모델 입력 단일 소스 원칙
> - **단일 출처 명세서**: [`docs/파이프라인_명세서_v2.md`](docs/파이프라인_명세서_v2.md)
> - **참고문헌 (12편)**: [`docs/참고문헌_v2.md`](docs/참고문헌_v2.md)
> - **발표 26p 와꾸**: [`docs/발표_와꾸_v2.md`](docs/발표_와꾸_v2.md)
>
> **척추 메시지**: *"단어 단서뿐 아니라 맥락 단서까지 함께 학습한 모델"*이 베이스 대비 분류 성능 + 판단 투명성 모두 향상됨을 정량 입증한다.

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
v2.1 재학습 후 생성되는 `outputs/reports/best_models.json`과 `checkpoints/*.pt`가 필요합니다.

구형 `models.zip` / `models/` 체크포인트 번들은 v1 구조라 사용하지 않습니다.

```bash
# v2.1 checkpoint 생성
./run.sh tune --force
./run.sh benchmark
```

이후 Playground는 `best_models.json`에 기록된 checkpoint 경로를 자동으로 읽습니다.
주요 후보는 다음 v2 조건명입니다:

```
A_B, B_B, C_B, D_B, A_R, B_R, C_R, D_R, D_B+Target
```

> Playground가 필요 없다면 이 단계를 건너뛰세요. 데이터/EDA/정적 리포트는 체크포인트 없이도 확인할 수 있습니다.

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

### XAI 이중 검증 결과

| 검증 차원 | 지표 | BERT-base | RoBERTa+VADER | 해석 |
|-----------|------|:---------:|:-------------:|------|
| **설명 안정성** | Overlap@5 (SHAP↔LIME) | 0.628 | **0.724** | 두 XAI 기법 간 Top-5 토큰 일치도 |
| **설명 타당성** | SHAP Top-5 vs Human Rationale | 0.579 | **0.688** | 모델이 인간과 같은 근거를 보는 정도 |
| **설명 타당성** | LIME Top-5 vs Human Rationale | **0.725** | 0.692 | LIME 기준 인간 판단 정렬도 |
| | Overlap >= 60% 샘플 수 | 36/50 | **42/50** | |
| | Rationale 보유 분석 샘플 | 33/50 | 33/50 | hatespeech/offensive만 rationale 보유 |

> **설명 안정성**: SHAP과 LIME이 같은 토큰을 중요하다고 보는가? (기법 간 일치도)
> **설명 타당성**: 모델이 인간 annotator와 같은 근거를 보는가? (인간 판단 정렬도)

### 결론

> 가설 "VADER 감성 점수가 혐오표현 탐지를 향상시킨다"는 **부분 채택**.
> VADER는 강력한 인코더(RoBERTa)와 결합할 때만 유의미한 시너지를 만들며,
> 같은 인코더(BERT)에서는 효과가 미미하다. **인코더 사전학습 품질이 가장 중요한 변수**이다.
>
> XAI 이중 검증에서 RoBERTa+VADER는 설명 안정성(Overlap@5 0.724)과
> SHAP 기반 설명 타당성(Human Rationale 0.688) 모두에서 BERT-base를 상회하여,
> 성능 향상이 설명 가능성 개선과 동반됨을 확인했다.

---

## 핵심 목표 (v2.1)

1. 베이스 BERT의 **단어 과의존(H1)을 진단** (SHAP top-k + CI 분석)
2. **두 개선 기법 결합** — Rationale-aware Attention Loss(학습 손실층) + VADER concat(모델 입력층)
3. **8조건 풀 ablation** — BERT × 4 (A/B/C/D) + RoBERTa × 4, 두 기법의 주효과·상호작용·사전학습 모델 강건성 동시 검증
4. **자동 XAI 4축** (Attribution / Faithfulness / Context Learning(CI/IS/MSS) / Plausibility) — 인간 라벨 의존 최소화한 맥락 학습 정량 입증

### 1차 baseline 핵심 목표 (참고용, 완료됨)

1. BERT 계열 모델의 **hate / offensive 오분류 패턴**을 EDA + XAI로 분석
2. **VADER 감성 피처** 결합 + **Ablation Study**(BERT+MLP)로 개선 요인 분리
3. 개선 전후 XAI 비교로 **Before/After 차이를 정량적으로 검증** (Overlap@5 + Human Rationale)

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
| 라벨링 | 3인 다수결 투표, undecided 제외 → **19,192건** 사용 |
| 분할 | **70 / 10 / 20** stratified split (train / val / test) |
| 시드 | 42, 52, 62 (3회 반복) |

---

## 모델 구성 — v2.1 (8조건 풀 ablation)

### 8조건 매트릭스 (BERT × 4 + RoBERTa × 4)

**BERT 패밀리**
| | VADER X | VADER O |
|--|--------|--------|
| Attn X | A_B: BERT + MLP (베이스) | C_B: BERT + VADER + MLP |
| Attn O | B_B: A_B + Attn Loss | **D_B: 둘 다 결합** |

**RoBERTa 패밀리** (동일 구조)
| | VADER X | VADER O |
|--|--------|--------|
| Attn X | A_R | C_R |
| Attn O | B_R | **D_R: 사전학습 모델 강건성 검증** |

- 분류 헤드 모든 transformer 조건에서 MLP 통일 (768→256→3 또는 772→256→3)
- VADER 유무는 입력 차원만 차이 (768d ↔ 772d)
- 학습 손실: `L_total = L_cls + α·L_attn (+ β·L_target)`
- α 그리드 {0.0, 0.1, 0.3, 0.5, 0.7, 1.0} → B_B 조건에서 최적값 결정

### 모델 입력 단일 소스 원칙 (v2.1 신규)

모델은 텍스트(post_tokens)만 입력. rationale·target·source·agreement는 학습 supervision 또는 분석으로만 사용 (입력 X). VADER 4d는 텍스트로부터 자동 추출되는 파생 피처이므로 본 원칙 부합.

추론 시 텍스트만 주어져도 동작 → 범용성 확보.

### 1차 baseline 모델 구성 (참고용)

다음은 1차 파이프라인(2026-04-11 완료) 모델 라인업이다. v2.1에서는 위 8조건 매트릭스로 확장된다.

기존: 6개 + Freeze Study

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
├── experiment_xai.py          # XAI 분석 (SHAP, LIME, Overlap@5, Human Rationale)
├── experiment_dashboard.py    # 정적 HTML 대시보드 생성기
├── dashboard_app.py           # FastAPI 대시보드 서버 (18탭 + Playground)
├── utils.py                   # 공통 유틸리티
│
├── data/                      # HateXplain 원본 데이터 (git 포함, 12MB)
│   ├── dataset.json
│   └── post_id_divisions.json
│
├── checkpoints/               # v2.1 재학습 체크포인트 (git 제외, ./run.sh benchmark로 생성)
│
├── outputs/                   # 실험 결과/캐시
│   ├── reports/               #   벤치마크 요약, 통계 검정, EDA, freeze study
│   ├── tuning/                #   하이퍼파라미터 탐색 이력
│   ├── runs/                  #   시드별 학습 로그, confusion matrix
│   ├── xai/                   #   SHAP/LIME 분석, Human Rationale 비교, 케이스 이미지
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
| outputs/ (CSV, JSON, PNG) | 변동 | O/X 혼합 | v2 split/VADER/EDA는 보존, 학습 결과는 재생성 |
| data/ (dataset.json) | 12MB | O | Data Explorer 작동 |
| models/ | - | **X** | v1 구형 번들 제거, 사용 안 함 |
| checkpoints/ | 변동 | **X** | `./run.sh benchmark`로 v2.1 checkpoint 재생성 |

---

## 학습 설정 및 하이퍼파라미터 튜닝

### 클래스 불균형 대응

HateXplain의 클래스 분포는 hate 29.5% / offensive 27.2% / normal 38.8%로 중간 수준의 불균형이다.
이에 대해 두 가지 기법을 동시에 적용한다.

| 기법 | 적용 대상 | 설명 |
|------|----------|------|
| **Class Weighting** | 전 모델 (ML + DL) | `imbalance_threshold=0.40` 설정. 소수 클래스 비율(27.2%)이 40% 미만이므로 `sklearn.compute_class_weight("balanced")`로 자동 가중치 계산 → `CrossEntropyLoss(weight=...)` 또는 sklearn `class_weight="balanced"` 적용 |
| **Label Smoothing** | DL 모델만 | `label_smoothing=0.1`. hard label [1, 0, 0] → soft label [0.933, 0.033, 0.033]. hate/offensive 어휘 Jaccard 유사도 0.71로 경계가 모호한 샘플에서 과적합 방지 |

### 머신러닝 모델 (TF-IDF + LR / SVM)

**TF-IDF 벡터화:**

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `ngram_range` | (1, 3) | 1~3gram 피처. "fuck"(1) + "fuck you"(2) + "go fuck yourself"(3) |
| `max_features` | 50,000 | 빈도 상위 5만개만 사용. 희귀 단어 노이즈 제거 |
| `sublinear_tf` | True | `1 + log(tf)` 적용. 동일 단어 반복 출현의 영향 완화 |

**하이퍼파라미터 튜닝 (C 파라미터):**

| 모델 | 파라미터 | 후보 | 선택 기준 | 설명 |
|------|---------|------|----------|------|
| Logistic Regression | C | [0.5, 1.0, 2.0] | val macro F1 | 정규화 강도의 역수. C가 클수록 결정 경계 복잡 |
| LinearSVC | C | [0.5, 1.0, 2.0] | val macro F1 | 동일. SVM은 `CalibratedClassifierCV(cv=3)`로 확률 출력 추가 |

- 시드별(42, 52, 62) 독립 실행 → C 선택도 시드마다 독립
- LR: `max_iter=2000`, SVM: `max_iter=10000`
- `class_weight="balanced"` 자동 적용

### 딥러닝 모델 (BERT / RoBERTa 계열)

**하이퍼파라미터 튜닝 — 순차 탐색 (Sequential Search):**

Grid Search(모든 조합)가 아닌, 한 파라미터씩 최적화 후 고정하는 순차 전략이다.
앞 단계의 최적값을 다음 단계에 반영하므로, 탐색 공간을 대폭 줄일 수 있다.

```
Step 1: learning_rate 탐색 [1e-5, 2e-5, 3e-5]  → 2e-5 확정
Step 2: batch_size 탐색    [64]                  → 64 고정 (MPS 메모리 한계)
Step 3: dropout 탐색       [0.1, 0.2, 0.3]      → 0.1 확정
Step 4: epochs 탐색        [5]                   → 5 고정 (early stopping이 실질 종료)
```

대상 모델: BERT+MLP, BERT+VADER, RoBERTa+VADER 순차 탐색 + B_B alpha grid + D_B+Target beta grid
튜닝 시드: 42 고정 (재현성), 선택 기준: val macro F1

**튜닝 대상 파라미터:**

| 순서 | 파라미터 | 후보 | 최적값 | 설명 |
|:---:|---------|------|:------:|------|
| 1 | `learning_rate` | [1e-5, 2e-5, 3e-5] | 2e-5 | 가중치 업데이트 보폭. BERT 논문 권장 범위 2e-5~5e-5 |
| 2 | `batch_size` | [64] | 64 | M3 Max MPS 메모리(30~48GB) 기준 최대치. 고정 |
| 3 | `dropout` | [0.1, 0.2, 0.3] | 0.1 | 뉴런 비활성 비율. 5에폭 짧은 학습에서는 낮을수록 유리 |
| 4 | `epochs` | [5] | 5 | early stopping patience=2가 실질적 종료 조건 |

**고정 파라미터 (탐색 대상 아님):**

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `weight_decay` | 0.01 | AdamW의 L2 정규화. 가중치 폭발 방지 |
| `warmup_ratio` | 0.10 | 전체 스텝의 10%는 LR을 0에서 서서히 증가. 학습 초반 불안정 방지 |
| `max_len` | 128 | 토큰 최대 길이. EDA에서 99%+ 텍스트가 128 이내 확인 |
| `mlp_hidden` | 256 | [CLS](768d) → 256d 차원 축소. Hybrid/MLP 모델 공통 |
| `early_stopping_patience` | 2 | val F1이 2에폭 연속 개선 없으면 학습 중단 |
| `label_smoothing` | 0.1 | hard → soft label. hate/offensive 경계 모호함 대응 |
| `class_weight` | balanced (자동) | `compute_class_weight_tensor(threshold=0.40)`이 데이터 분포 기반 자동 판단 |

**옵티마이저:** AdamW (Adam + 올바른 L2 정규화)
**스케줄러:** Linear warmup → linear decay

### 벤치마크 실행 구성

| 항목 | 값 | 설명 |
|------|-----|------|
| `seeds` | [42, 52, 62] | 3회 반복. paired t-test 최소 요건 |
| 모델 수 | 메인 10개 + 부가 1개 | TF-IDF 2개 + v2.1 DL 8조건 + D_B target aux |
| 총 학습 횟수 | 24회 + 부가 실험 | 8조건 x 3시드 + D_B+Target |
| 통계 검정 | paired t-test + 2/3-way ANOVA | 3시드 F1 배열과 factor 효과 검정 |

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
| `run_xai()` | `experiment_xai.py` | SHAP/LIME + Human Rationale 이중 검증 |
| `_load_human_rationales()` | `experiment_xai.py` | dataset.json에서 majority vote rationale 로드 |
| `_compute_rationale_overlap()` | `experiment_xai.py` | Model Top-5 vs Human rationale 토큰 overlap |
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
| **Human Rationale** | majority vote 기반 | annotator 간 합의도에 따라 rationale 품질 상이, normal 클래스는 rationale 없음 |

### Future Work

- 시드 5~10회로 확장하여 통계적 검정력 강화
- HateBERT / TweetBERT 등 도메인 특화 모델 비교
- HateXplain target 정보를 모델 입력으로 활용
- feature-level SHAP으로 VADER 기여도 직접 측정
- Overlap@K 민감도 분석 (K=3, 5, 10)
- Rationale overlap의 annotator 합의도별 세분화 분석
- 한국어/다국어 혐오표현 데이터셋 확장

---

## License

This project is for academic purposes (한성대학교 빅데이터프로그래밍 수업 프로젝트).
HateXplain dataset: CC-BY 4.0 | BERT/RoBERTa: Apache 2.0 | VADER: MIT
