# 08. XAI Report Template

> 목적: XAI 결과를 최종 보고서에 넣을 때 어떤 표, 그림, 문장을 사용할지 미리 정한다. XAI는 모델을 설계한 근거가 아니라, 학습된 모델의 판단 패턴을 사후 검증하는 근거다.

---

## 1. XAI section 기본 구조

최종 보고서의 XAI 장은 아래 순서로 쓴다.

```text
1. XAI 분석 목적
2. XAI evidence bundle 요약
3. XAI sample selection
4. 사용한 XAI 방법
5. Primary XAI 결과
6. Seed stability 결과
7. Deep case analysis
8. Ablation XAI 결과
9. 해석과 한계
```

보고서 서술용 상위 claim은 가능하면 아래 두 파일을 우선 소스로 삼는다.

```text
xai/evidence_bundle/xai_claims.json
xai/evidence_bundle/xai_dashboard_bundle.json
```

---

## 2. XAI 분석 목적 문장

권장 문장:

```text
본 연구에서 XAI는 모델 설계 단계의 입력이 아니라, 15 seed 반복 실험으로 학습된 모델이 어떤 토큰과 맥락 단서에 의존하는지 사후적으로 점검하기 위한 분석 절차로 사용하였다.
```

```text
특히 baseline과 최종 v2 조건의 attribution pattern, 인간 rationale과의 정렬성, 중요 토큰 제거에 따른 예측 변화, seed 변화에 따른 설명 안정성을 비교하였다.
```

피해야 할 문장:

```text
XAI가 모델 개선의 원인을 증명했다.
XAI를 통해 VADER feature를 설계했다.
XAI 결과만으로 모델이 혐오표현 맥락을 완전히 이해한다고 볼 수 있다.
```

추가 권장 문장:

```text
TF-IDF baseline도 강한 분류 성능을 보일 수 있으나, 본 v2 파이프라인은 XAI evidence bundle을 통해 판단 근거와 취약성을 함께 보고한다.
```

---

## 3. Sample selection 서술

권장 구조:

```text
Primary XAI는 test split에서 label과 case type을 층화하여 200개 샘플을 고정한 뒤, A_B와 D_B 조건의 15개 seed checkpoint에 동일하게 적용하였다.
```

```text
Deep XAI는 각 조건에서 median-performing seed를 선택하고, 500개 층화 샘플에 대해 더 자세한 token attribution과 case-level plot을 생성하였다.
```

```text
Ablation XAI는 8개 조건 전체에 대해 각 조건의 median-performing seed와 50개 공통 샘플을 사용하여 조건 간 설명 지표의 방향성을 비교하였다.
```

### 3.1 Evidence bundle 요약 문장

권장 문장:

```text
Primary, Deep, Ablation XAI 결과는 최종적으로 evidence bundle로 통합하여, 토큰 attribution raw log, faithfulness/context/plausibility 지표, 해석 claim, dashboard 요약 JSON을 함께 저장하였다.
```

---

## 4. Primary XAI 결과표

표 제목:

```text
Table X. Primary XAI comparison between baseline and v2 final condition across 15 seeds
```

권장 컬럼:

```text
Metric
A_B mean
A_B 95% CI
D_B mean
D_B 95% CI
Mean difference
paired p-value
adjusted p-value (supplementary, if multiple XAI metrics are compared)
Effect size
Interpretation
```

포함할 metric:

```text
SHAP-LIME Overlap@5
Rationale Precision@5
Rationale Recall@5
Rationale F1@5
Comprehensiveness
Sufficiency
Leave-one-out Drop
Top-k Jaccard across seeds
Rank Correlation across seeds
```

---

## 5. Seed stability 결과표

표 제목:

```text
Table X. Seed-level explanation stability
```

권장 컬럼:

```text
Condition
Top-k Jaccard mean
Top-k Jaccard std
Rank correlation mean
Rank correlation std
Rationale F1 variance
Faithfulness variance
Interpretation
```

해석 문장:

```text
D_B 조건에서 top-k attribution overlap과 rank correlation이 baseline보다 높다면, 해당 조건의 설명 패턴이 단일 checkpoint 선택에 덜 민감하다는 보조 근거로 해석할 수 있다.
```

---

## 6. Deep case analysis 구성

사례는 아래 유형을 균형 있게 고른다.

```text
both_correct
baseline_wrong_v2_correct
both_wrong
model_disagreement
```

사례별로 포함할 내용:

```text
sample id
true label
baseline prediction
v2 prediction
top attribution tokens
human rationale tokens
masking result
brief interpretation
```

보고서 문장 예시:

```text
baseline이 오분류하고 D_B가 정분류한 사례에서는 D_B의 상위 attribution token이 인간 rationale과 더 많이 겹치는 경향이 관찰되었다.
```

주의:

```text
개별 사례는 설명을 돕는 보조 자료다.
개별 사례만으로 전체 성능 개선을 주장하지 않는다.
```

---

## 7. Ablation XAI 결과표

표 제목:

```text
Table X. XAI metrics across ablation conditions
```

권장 컬럼:

```text
Condition
Backbone
Attention loss
Sentiment feature
Rationale F1@5
Comprehensiveness
Sufficiency
Attention entropy
MSS
Interpretation
```

목적:

```text
8조건 전체에서 attention loss와 sentiment feature가 설명 지표에 어떤 방향의 변화를 만드는지 확인한다.
```

---

## 8. 그림 구성

권장 그림:

```text
1. Baseline vs D_B SHAP token attribution example
2. Human rationale vs model attribution overlap heatmap
3. Seed stability distribution plot
4. Ablation XAI metric heatmap
5. Masking faithfulness curve
```

그림 caption에는 반드시 아래를 포함한다.

```text
condition
seed selection rule
sample selection rule
XAI method
metric definition
```

---

## 9. 해석과 한계

권장 해석:

```text
XAI 결과가 성능 결과와 같은 방향을 보이면, v2 조건이 단순히 metric만 개선한 것이 아니라 인간 rationale과 더 정렬된 판단 패턴을 보일 가능성을 제시한다.
```

권장 한계:

```text
SHAP과 LIME은 근사적 설명 방법이므로 실제 모델 내부 인과 구조를 완전히 복원하지 않는다.
```

```text
인간 rationale 자체도 annotation bias와 annotator disagreement의 영향을 받을 수 있다.
```

```text
XAI 결과는 통계적 성능 검정과 함께 해석해야 하며, 단독 결론으로 사용하지 않는다.
```

---

## 10. 최종 XAI 결론 템플릿

최종 보고서에서는 아래 형식으로 결론을 쓴다.

```text
종합하면, D_B 조건은 A_B baseline 대비 macro-F1에서 [방향/크기]의 차이를 보였고, XAI 분석에서도 [rationale alignment/faithfulness/seed stability] 지표가 [방향]으로 나타났다. 이는 v2 조건이 성능 지표뿐 아니라 설명 가능성 측면에서도 연구 가설과 일관된 경향을 보였음을 시사한다. 다만 XAI는 사후 분석 도구이며, 설명 지표만으로 모델 개선의 인과적 원인을 확정할 수는 없다.
```

TF-IDF 대비 강점 방어용 문장:

```text
TF-IDF와 최종 정확도 차이가 크지 않더라도, v2는 예측값만 남기는 것이 아니라 token attribution, faithfulness, plausibility, subgroup 취약성까지 포함한 full XAI evidence bundle을 남긴다. 이는 결과 해석과 오류 분석 측면에서 본 연구 파이프라인의 실질적 차별점이다.
```
