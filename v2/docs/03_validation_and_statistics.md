# 03. Validation and Statistical Plan

---

## 1. 검증 목표

15 seed 실험의 목적은 단순히 평균 성능을 높게 만드는 것이 아니다.

목표는 다음이다.

```text
1. 15 seed 평균과 표준편차로 단일 실행 의존을 줄인다.
2. 핵심 비교인 baseline 대비 제안 모델 차이를 같은 seed 기준 paired t-test로 확인한다.
3. p-value 하나가 아니라 평균 차이, 95% CI, effect size를 함께 본다.
4. XAI는 대표 사례와 rationale alignment 중심으로 성능 해석을 보조한다.
```

이 문서의 역할은 "성능과 XAI 지표가 통계적으로 반복되는가"를 검증하는 것이다.

학부 프로젝트 기준의 주 분석은 아래 수준으로 고정한다.

```text
메인 분석:
15 seed mean ± std
핵심 비교 A_B vs D_B paired t-test
mean difference + 95% CI + effect size
XAI 대표 사례와 evidence bundle 설명

보조/부록 분석:
Holm-adjusted p-value
ANOVA
bootstrap CI
여러 조건 간 pairwise comparison
```

```text
통계 문서가 답하는 질문:
- 개선이 seed 변동을 넘어서는가?
- 설명 지표가 반복 가능한가?

통계 문서가 직접 답하지 않는 질문:
- report/dashboard가 어떤 XAI artifact를 읽어야 하는가?
- TF-IDF 대비 강점을 어떤 산출물 묶음으로 방어할 것인가?
```

후자는 `04_xai_protocol.md`, `07_output_and_report_contract.md`, `08_xai_report_template.md`의 `xai-bundle` 계약이 맡는다.

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

## 4. 우리 수준의 핵심 분석

주요 비교는 모든 조합이 아니라, 연구에서 실제로 주장하고 싶은 비교만 전면에 둔다.

발표/보고서의 핵심 비교:

```text
A_B: 기본 BERT baseline
D_B: Rationale-aware Attention + VADER 적용 BERT

핵심 질문:
D_B가 A_B보다 Macro F1에서 안정적으로 나아졌는가?
```

검정 방식:

```text
같은 seed에서 나온 A_B와 D_B의 Macro F1 차이를 paired t-test로 비교한다.
```

보고할 값:

```text
condition별 Macro F1 mean ± std
paired mean difference
95% confidence interval
paired t-test p-value
paired effect size 또는 Cohen's dz
```

아래 비교는 표/부록에 둘 수 있지만 발표의 주연으로 세우지 않는다.

```text
B_B - A_B
C_B - A_B
D_B - B_B
D_B - C_B
D_R - A_R
D_B - D_R
```

---

## 5. Holm 보정의 위치

Holm-Bonferroni 보정은 메인 분석이 아니라 보조 안전장치다.

여러 pairwise test를 동시에 많이 수행하면 우연히 p-value가 작게 나오는 비교가 생길 수 있다. Holm 보정은 이런 과대해석을 줄이기 위한 다중비교 보정이다.

우리 보고서에서의 위치:

```text
본문:
핵심 비교 A_B vs D_B의 paired t-test, 평균 차이, CI, effect size를 중심으로 설명한다.

부록/보조:
여러 비교를 함께 보여줄 때 paired_tests_holm.csv의 adjusted p-value를 같이 제시한다.
```

발표에서 권장하는 표현:

```text
주요 비교는 동일 seed 기반 paired t-test로 검정했다.
여러 조건을 동시에 비교하는 보조 분석에서는 과대해석을 피하기 위해 adjusted p-value도 함께 확인했다.
```

---

## 6. ANOVA의 위치

ANOVA는 구현되어 있지만, 학부 발표의 핵심 검정으로 세우기에는 설명 비용이 크다. 따라서 `Attention Loss 주효과`, `VADER 주효과`, `상호작용`을 정교하게 주장하기보다, 부록 또는 보조 분석으로 둔다.

본문에서는 아래처럼 말한다.

```text
8조건 ablation 결과는 평균 성능표와 핵심 paired comparison을 중심으로 해석했다.
ANOVA는 보조적으로 확인했으며, 본 연구의 주요 결론은 A_B와 D_B의 같은 seed 기반 비교에 둔다.
```

### 6.1 BERT family 2-way analysis

BERT family에서는 필요 시 다음 구조를 보조적으로 확인한다.

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

## 8. Bootstrap 보강 (작업 #7 — 구현 완료)

Seed 반복은 학습 stochasticity를 반영하지만, 작은 표본(15 seed)에서 t-분포 CI는 정규성 가정이 약하다. 다만 학부 프로젝트 본문에서는 bootstrap을 길게 설명하지 않고, 95% CI를 보강하는 선택지로만 둔다.

작업 #7에서 **percentile bootstrap CI**를 도입. `pipeline/statistics.py`의 `_bootstrap_ci()`가 numpy로 N회 resample → 평균 분포 → 양 끝 분위수.

```python
# 구현 (pipeline/statistics.py)
def _mean_ci(values, bootstrap_iterations=None):
    if bootstrap_iterations and bootstrap_iterations > 0:
        return _bootstrap_ci(values, iterations=bootstrap_iterations)
    return _mean_ci_t(values)  # numpy 없을 때 fallback
```

manifest 설정:
```json
"statistics": { "bootstrap_iterations": 1000 }
```
0이거나 키가 없으면 자동으로 t-분포 fallback. `summarize_benchmark`와 `compute_paired_tests` 둘 다 적용.

```text
Seed CI: training stochasticity (15 seed mean ± std)
Bootstrap CI: training + sample noise (percentile bootstrap on seed means)
```

향후 보강 가능 (현재 미구현): sample-level paired bootstrap (test sample 단위 resample).

---

## 8.1 ANOVA Effect Size (작업 #14 — 구현 완료, 보조)

F·p 값만으로는 "얼마나 큰 효과인가"를 판단할 수 없으므로 ANOVA 출력에 효과 크기 두 개를 추가했다. 이 값은 발표 본문보다는 부록/검증 문서에서 확인한다.

```text
eta_squared          = SS_factor / SS_total
partial_eta_squared  = SS_factor / (SS_factor + SS_residual)
```

`pipeline/statistics.py`의 `_anova_table_to_rows()`가 자동 계산. `anova_2way_bert.csv` / `anova_2way_roberta.csv` / `anova_3way.csv` 모두 컬럼 9개로 확장 (sum_sq, df, F, p_value, eta_squared, partial_eta_squared).

Cohen(1988) 해석 기준:

| η² 범위 | 효과 크기 |
|---|---|
| < 0.01 | negligible |
| 0.01 ~ 0.06 | small |
| 0.06 ~ 0.14 | medium |
| ≥ 0.14 | large |

Smoke 결과 예시 (가짜 데이터, BERT 2-way): attention_loss η² = 0.6263, partial η² = 0.9102 — large.

---

## 9. 성공 기준

강한 성공:

```text
D_B - A_B mean difference > 0
paired t-test p < 0.05
95% CI lower bound > 0
effect size가 작지 않음
XAI 대표 사례가 연구 가설과 크게 충돌하지 않음
```

중간 성공:

```text
D_B - A_B mean difference > 0
paired t-test p < 0.05 또는 CI가 대부분 positive
effect size가 해석 가능한 수준
XAI에서 일부 설명 근거 확인
```

보수적 해석:

```text
mean difference > 0 but p >= 0.05
```

이 경우 “개선 경향”으로만 표현한다.

Holm-adjusted p-value가 같이 유의하면 더 보수적으로도 방어 가능하다고 덧붙일 수 있다. 그러나 Holm을 통과하지 못했다고 해서 핵심 분석 전체가 무의미해지는 것은 아니며, 평균 차이와 effect size, CI를 함께 보고 신중하게 해석한다.

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

---

## 11. 통계 결과와 evidence bundle의 관계

TF-IDF와 최종 정확도 차이가 크지 않을 가능성은 충분히 있다. 이 경우에도 문서 역할은 분리해서 해석한다.

```text
03_validation_and_statistics.md:
  성능 차이가 통계적으로 유의한가?
  XAI 지표가 seed 반복 기준으로 안정적인가?

04_xai_protocol.md / 07_output_and_report_contract.md / 08_xai_report_template.md:
  어떤 판단 근거와 취약성을 evidence bundle로 남길 것인가?
  report/dashboard는 어떤 JSON/CSV를 우선 소비할 것인가?
```

즉 성능 검정은 v2의 "얼마나 나아졌는가"를 답하고, evidence bundle은 v2의 "무엇을 더 남기는가"를 답한다.

권장 표현:

```text
15 seed 반복에서 안정적인 개선 경향을 보였다.
paired comparison 기준 유의한 차이를 보였다.
효과크기와 신뢰구간을 함께 고려할 때 실질적 개선으로 해석된다.
```
