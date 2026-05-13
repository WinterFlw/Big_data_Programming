# 03. Validation and Statistical Plan

---

## 1. 검증 목표

15 seed 실험의 목적은 단순히 평균 성능을 높게 만드는 것이 아니다.

목표는 다음이다.

```text
1. 조건 간 차이가 seed stochasticity를 넘어서는지 확인
2. Attention Loss와 VADER의 주효과/상호작용 평가
3. BERT와 RoBERTa family에서 효과가 유지되는지 확인
4. XAI 지표도 seed 반복 기준으로 안정적인지 확인
```

---

## 2. Seed 설계

권장 seed:

```python
[42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182]
```

같은 seed를 모든 조건에 공통 적용한다. 이렇게 해야 paired comparison이 가능하다.

```text
seed 42: A_B, B_B, C_B, D_B, A_R, B_R, C_R, D_R
seed 52: A_B, B_B, C_B, D_B, A_R, B_R, C_R, D_R
...
```

---

## 3. 주요 지표

Primary metric:

```text
Macro F1
```

Secondary metrics:

```text
Accuracy
AUROC
Macro Precision
Macro Recall
Class-wise F1
```

Macro F1을 primary로 두는 이유는 class imbalance와 hate/offensive 경계 모호성을 동시에 고려하기 위해서다.

---

## 4. Paired tests

주요 비교는 동일 seed 내 paired difference로 계산한다.

핵심 비교:

```text
D_B - A_B
B_B - A_B
C_B - A_B
D_B - B_B
D_B - C_B
D_R - A_R
D_B - D_R
```

각 비교에서 보고할 값:

```text
mean difference
standard deviation of paired difference
95% confidence interval
paired t-test p-value
Holm-Bonferroni adjusted p-value
Cohen's dz
```

---

## 5. Multiple comparison correction

여러 pairwise test를 수행하므로 p-value 보정이 필요하다.

권장 방식:

```text
Holm-Bonferroni correction
```

Holm 방식은 Bonferroni보다 덜 보수적이면서 family-wise error rate를 통제한다.

---

## 6. ANOVA

### 6.1 BERT family 2-way analysis

BERT family에서는 다음 구조를 검정한다.

```text
Macro F1 ~ Attention Loss x VADER
block = seed
```

질문:

```text
Attention Loss 주효과가 있는가?
VADER 주효과가 있는가?
Attention Loss와 VADER 상호작용이 있는가?
```

### 6.2 3-way analysis

전체 8조건에서는 family까지 포함한다.

```text
Macro F1 ~ Family x Attention Loss x VADER
block = seed
```

질문:

```text
BERT와 RoBERTa family 차이가 있는가?
개선 기법의 효과가 family에 따라 달라지는가?
```

---

## 7. Confidence interval

각 모델 성능은 다음을 보고한다.

```text
mean ± standard deviation
95% CI of mean
```

조건 간 차이는 paired difference의 95% CI를 보고한다.

```text
CI = mean_diff ± t_(0.975, n-1) * sd_diff / sqrt(n)
```

15 seed라면 자유도는 14다.

---

## 8. Bootstrap 보강

Seed 반복은 학습 stochasticity를 반영하지만, test sample 자체의 불확실성은 완전히 반영하지 못한다.

따라서 최종 주요 비교에는 test set bootstrap CI를 추가할 수 있다.

```text
Seed CI: training stochasticity
Bootstrap CI: test sample uncertainty
```

권장:

```text
bootstrap iterations = 1000
metric = Macro F1 difference
paired at sample level
```

---

## 9. 성공 기준

강한 성공:

```text
D_B - A_B mean difference > 0
Holm-adjusted p < 0.05
95% CI lower bound > 0
Cohen's dz medium or larger
```

중간 성공:

```text
D_B - A_B mean difference > 0
uncorrected p < 0.05
effect size meaningful
CI mostly positive
```

보수적 해석:

```text
mean difference > 0 but p >= 0.05
```

이 경우 “개선 경향”으로만 표현한다.

---

## 10. 보고 시 금지할 표현

아래 표현은 통계 결과가 매우 강하지 않으면 피한다.

```text
완전히 입증했다
항상 우월하다
VADER 단독 효과가 명확하다
Attention Loss 단독 효과가 명확하다
XAI가 개선 원인을 증명했다
```

권장 표현:

```text
15 seed 반복에서 안정적인 개선 경향을 보였다.
paired comparison 기준 유의한 차이를 보였다.
효과크기와 신뢰구간을 함께 고려할 때 실질적 개선으로 해석된다.
```

