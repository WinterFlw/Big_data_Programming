# 15 Seed 반복 실험 및 XAI 보강 계획

> 작성일: 2026-05-13
> 프로젝트: HateSpeachStudy v2.1
> 목적: 8조건 ablation의 통계적 검정력을 높이고, XAI 해석을 seed 안정성까지 포함한 검증 체계로 보강한다.

---

## 1. 핵심 결론

본 프로젝트의 다음 실험 단위는 **8조건 × 15 seed**를 기본으로 설정한다.

15 seed 반복은 단일 실행 결과가 아니라, 모델 학습 과정의 stochastic variance를 추정하기 위한 반복 설계다. 모든 조건은 동일한 데이터 분할, 동일한 평가 지표, 동일한 family별 공통 hyperparameter를 사용하고, 동일 seed 안에서 조건 간 성능을 paired comparison으로 비교한다.

최종 목표는 다음 세 가지다.

1. `D_B`가 `A_B`보다 안정적으로 높은 성능을 보이는지 검정한다.
2. Attention Loss와 VADER의 주효과 및 상호작용을 평가한다.
3. XAI 결과가 특정 checkpoint 하나에만 의존하지 않고 seed가 바뀌어도 유지되는지 확인한다.

---

## 2. 실험 조건

v2.1의 메인 ablation 조건은 다음 8개다.

| 조건 | Family | Attention Loss | VADER | 의미 |
|---|---|---:|---:|---|
| A_B | BERT | X | X | BERT baseline |
| B_B | BERT | O | X | BERT + Attention Loss |
| C_B | BERT | X | O | BERT + VADER |
| D_B | BERT | O | O | BERT + Attention Loss + VADER |
| A_R | RoBERTa | X | X | RoBERTa baseline |
| B_R | RoBERTa | O | X | RoBERTa + Attention Loss |
| C_R | RoBERTa | X | O | RoBERTa + VADER |
| D_R | RoBERTa | O | O | RoBERTa + Attention Loss + VADER |

15 seed 기준 학습 횟수는 다음과 같다.

```text
8 conditions × 15 seeds = 120 training runs
```

권장 seed 목록은 다음과 같다.

```python
[42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182]
```

기존 seed 흐름을 유지하므로 로그 확인과 추가 실행 관리가 쉽다.

---

## 3. Hyperparameter 통제 원칙

Ablation의 핵심은 **조건 외 변수를 통제하는 것**이다. 따라서 조건별 튜닝값을 그대로 쓰면 안 된다. `A_B`, `B_B`, `C_B`, `D_B`는 BERT family 공통 hyperparameter를 사용하고, `A_R`, `B_R`, `C_R`, `D_R`는 RoBERTa family 공통 hyperparameter를 사용한다.

예시:

```python
BERT_COMMON = {
    "learning_rate": 2e-5,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 5,
}

ROBERTA_COMMON = {
    "learning_rate": 2e-5,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 5,
}
```

조건별로 달라져야 하는 값은 아래 두 축뿐이다.

```text
use_attention_loss: true / false
use_vader: true / false
```

이 원칙이 깨지면 성능 차이가 VADER 또는 Attention Loss 때문인지, learning rate/dropout 때문인지 분리할 수 없다.

---

## 4. 예상 실행 시간

기존 실행 로그 기준 평균 시간은 다음과 같다.

| 실행 단위 | 평균 시간 |
|---|---:|
| BERT 조건 1회 | 약 5.3분 |
| RoBERTa 조건 1회 | 약 6.7분 |
| 8조건 × 1 seed | 약 48분 |
| 8조건 × 15 seed | 약 12시간 |

Freeze Study까지 15 seed로 실행하면 추가로 약 2시간 내외가 필요하다.

```text
메인 benchmark: 약 12시간
freeze-study 포함: 약 14시간
XAI: 별도 수 시간 이상
```

XAI는 SHAP/LIME 계산이 병목이므로 학습 batch와 분리하여 실행한다.

---

## 5. 통계 분석 계획

### 5.1 Primary Metric

주요 성능 지표는 Macro F1로 둔다.

```text
Primary metric: Macro F1
Secondary metrics: class-wise F1, accuracy, AUROC
Repeated unit: random seed
Design: paired repeated evaluation across identical seeds
```

Macro F1을 primary metric으로 두는 이유는 HateXplain 3-class 문제가 class imbalance와 hate/offensive 경계 모호성을 포함하기 때문이다.

### 5.2 Paired Comparison

동일 seed 안에서 조건 간 차이를 비교한다.

주요 비교:

```text
A_B vs D_B
A_B vs B_B
A_B vs C_B
B_B vs D_B
C_B vs D_B
A_R vs D_R
```

보고 항목:

```text
mean difference
95% confidence interval
paired t-test p-value
Cohen's dz
Holm-Bonferroni corrected p-value
```

### 5.3 ANOVA

BERT family 내부에서는 2-way 분석을 수행한다.

```text
Macro F1 ~ Attention Loss × VADER
block: seed
```

전체 8조건에서는 family까지 포함한 3-way 분석을 수행한다.

```text
Macro F1 ~ Family × Attention Loss × VADER
block: seed
```

여기서 seed는 반복 측정 단위로 취급한다. 단순 독립 표본처럼 다루지 않고, 동일 seed 내 paired 구조를 보존하는 것이 중요하다.

---

## 6. 15 Seed의 통계적 근거

기존 3-seed 파일럿 기준으로 `D_B - A_B`의 차이는 다음과 같았다.

```text
평균 차이: 약 +0.0095 Macro F1
paired difference 표준편차: 약 0.0050
효과크기 dz: 약 1.91
```

이 효과가 유지된다면 `D_B > A_B`는 5 seed만으로도 유의해질 가능성이 높다. 그러나 `D_B > B_B`, `D_R > A_R` 같은 중간 크기 효과는 10 seed에서도 애매할 수 있다. 15 seed는 이런 세부 ablation 비교의 검정력을 높이는 현실적인 선택이다.

요약:

| 비교 | 5 seed | 10 seed | 15 seed |
|---|---|---|---|
| D_B > A_B | 가능성 높음 | 안정적 | 매우 안정적 |
| D_B > C_B | 가능성 높음 | 안정적 | 매우 안정적 |
| D_B > B_B | 부족할 수 있음 | 애매 | 비교적 안정 |
| D_R > A_R | 부족할 수 있음 | 애매 | 비교적 안정 |
| B_B > A_B | 대체로 부족 | 부족 가능 | 여전히 애매할 수 있음 |
| C_B > A_B | 효과 거의 없음 | 어려움 | 어려움 |

따라서 15 seed는 “핵심 효과”뿐 아니라 “세부 ablation 차이”까지 방어하기 위한 선택이다.

---

## 7. XAI 보강 설계

성능 검증을 15 seed로 확장하면 XAI도 단일 checkpoint 해석에서 벗어나야 한다. XAI의 핵심 질문은 다음과 같다.

```text
D_B가 어떤 근거로 맞추는가?
그 근거가 인간 rationale과 맞는가?
그 근거가 seed가 바뀌어도 유지되는가?
```

따라서 XAI는 다음 4단계로 나눈다.

### 7.1 Primary XAI

핵심 비교 조건만 깊게 분석한다.

```text
A_B vs D_B
15 seed 전체
200 stratified samples
SHAP + LIME + rationale + masking
```

목적은 seed별 XAI 지표의 평균과 분산을 추정하는 것이다.

### 7.2 Deep XAI

정성 분석과 세부 사례 시각화를 위한 확장 분석이다.

```text
A_B vs D_B
median-performing seed
500 stratified samples
case visualization
```

대표 checkpoint는 최고 성능 seed가 아니라 median-performing seed를 사용한다. 이는 cherry-picking 위험을 줄이기 위함이다.

### 7.3 Ablation XAI Matrix

8조건 전체의 자동 XAI 메트릭을 축약 분석한다.

```text
8 conditions
median-performing seed
50 samples
CI / MSS / LOO / attention rollout entropy
```

8조건 전체에 15 seed × 500 sample XAI를 수행하는 것은 비용이 과도하다. 따라서 전체 matrix는 reduced sample로 운영하고, 이 한계를 보고서에 명시한다.

### 7.4 Improvement Case XAI

성능 개선 사례를 별도 분석한다.

```text
A_B wrong, D_B correct cases
fixed_error 50 samples
qualitative comparison
```

이 분석은 대표 샘플 설명용이며, 전체 XAI 통계의 근거로 사용하지 않는다.

---

## 8. XAI 샘플링 원칙

XAI 메인 샘플은 fixed_error 위주로만 뽑으면 편향된다. 따라서 representative XAI는 stratified sample로 구성한다.

권장 구성:

```text
총 500 samples

label stratified:
- hatespeech 약 170
- offensive 약 170
- normal 약 160

case type:
- both correct
- A_B wrong, D_B correct
- both wrong
- model disagreement
```

source와 target도 가능하면 분포를 기록한다.

```text
source: gab / twitter
target: Islam, Women, African 등 주요 target
```

---

## 9. XAI 지표

XAI 보강에서 사용할 지표는 다음과 같다.

### 9.1 SHAP-LIME Agreement

```text
Overlap@5
Overlap@10
```

두 XAI 방법이 같은 토큰을 중요하다고 보는지 확인한다.

### 9.2 Human Rationale Alignment

```text
Precision@5
Recall@5
F1@5
token-level coverage
```

모델 attribution이 인간 annotator rationale과 얼마나 정렬되는지 평가한다.

### 9.3 Faithfulness

```text
Comprehensiveness
Sufficiency
Leave-one-out probability drop
```

중요 토큰을 제거하거나 유지했을 때 예측 확률이 얼마나 바뀌는지 측정한다.

### 9.4 Seed Stability

```text
Top-k Jaccard across seeds
rank correlation across seeds
rationale overlap variance across seeds
```

같은 샘플에서 seed가 바뀌어도 설명이 유지되는지 확인한다.

### 9.5 Subgroup XAI

```text
label별: hatespeech / offensive / normal
source별: gab / twitter
target별: 주요 target category
```

성능 개선과 설명 패턴이 특정 subgroup에만 치우치는지 확인한다.

---

## 10. Batch Runner 요구사항

15 seed 운영을 위해 batch runner는 필수다. 단순 반복 스크립트가 아니라 resume 가능한 실행 관리자여야 한다.

필수 기능:

```text
1. seed 목록 관리
2. condition 목록 관리
3. 완료 run skip
4. 실패 run 재시도
5. run별 로그 저장
6. 최종 summary 재집계
7. XAI와 benchmark 분리 실행
```

권장 CLI:

```bash
./run.sh batch benchmark --seeds 42,52,62,72,82,92,102,112,122,132,142,152,162,172,182 --resume
./run.sh batch xai-primary --conditions A_B,D_B --sample-size 200 --resume
./run.sh batch xai-deep --conditions A_B,D_B --sample-size 500 --median-seed
./run.sh batch xai-ablation --sample-size 50 --median-seed
```

완료 판정 기준:

```text
outputs/runs/{condition}/seed_{seed}/metrics.json
outputs/runs/{condition}/seed_{seed}/history.csv
outputs/runs/{condition}/seed_{seed}/predictions.pkl
checkpoints/{condition}_seed_{seed}.pt
```

위 산출물이 모두 있으면 해당 run은 완료로 본다.

---

## 11. 보고서 문장 예시

### 11.1 성능 검증

```text
본 연구는 8개 ablation 조건을 15개 random seed에서 반복 실행하여,
학습 초기화와 mini-batch 순서에 따른 stochastic variance를 추정했다.
모든 조건은 동일한 데이터 분할과 family별 공통 hyperparameter를 사용했으며,
동일 seed 내 조건 간 성능 차이를 paired comparison으로 검정했다.
주요 비교에는 paired t-test, Holm-Bonferroni 보정, Cohen's dz,
95% confidence interval을 함께 보고했다.
```

### 11.2 XAI 검증

```text
XAI 평가는 핵심 비교 조건 A_B와 D_B에 대해 동일한 stratified sample set을 사용하고,
15개 seed checkpoint에서 설명 지표를 반복 측정하여 explanation stability를 검증했다.
정성적 case visualization은 cherry-picking을 피하기 위해 median-performing seed를 사용했다.
```

### 11.3 한계

```text
15 seed 반복은 학습 stochasticity를 추정하지만,
다른 train/test split에서의 데이터 샘플링 변동성을 완전히 반영하지는 않는다.
이를 보완하기 위해 test set bootstrap confidence interval과 subgroup 분석을 함께 보고한다.
```

---

## 12. 최종 권장안

최종 실행 설계는 다음과 같다.

```text
Benchmark:
8조건 × 15 seed = 120 training runs

Freeze Study:
2조건 × 15 seed = 30 training runs, 선택

Primary XAI:
A_B vs D_B × 15 seed × 200 samples

Deep XAI:
A_B vs D_B × median seed × 500 samples

Ablation XAI:
8조건 × median seed × 50 samples

Case Study:
fixed_error 50 samples
```

이 설계는 성능 검정과 XAI 해석을 모두 seed 반복 기반으로 끌어올린다. 핵심은 단순히 많이 돌리는 것이 아니라, **동일 seed paired design + 공통 hyperparameter 통제 + XAI seed stability**를 함께 확보하는 것이다.
