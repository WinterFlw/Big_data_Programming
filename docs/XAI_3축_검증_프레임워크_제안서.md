# XAI 3축 검증 프레임워크 제안서 (DEPRECATED v1)

> ⚠ **DEPRECATED — v1 기준 문서**
>
> 본 문서는 1차 파이프라인(2026-04-11 기준) XAI 3축(SHAP+LIME+Masking) 설계 자료입니다.
> **현재 단일 출처는 [`파이프라인_명세서_v2.md`](파이프라인_명세서_v2.md)** 이며, v2.1에서는 **XAI 4축**(Attribution / Faithfulness / **Context Learning(CI/IS/MSS)** / Plausibility)으로 확장되었습니다.
> 본 문서는 baseline 기록으로 유지됩니다.

---

> HateSpeachStudy — 혐오표현 탐지 모델의 맥락 이해 능력을 검증하기 위한 XAI 프레임워크 설계 (v1)
>
> 한성대학교 빅데이터프로그래밍

---

## 1. 문제 인식

현재 혐오표현 탐지 모델은 Macro F1 등 분류 성능 지표에서 일정 수준의 성과를 보이지만, 모델이 "왜" 그런 판단을 내렸는지에 대한 설명의 신뢰성은 충분히 검증되지 않고 있다. 특히 다음과 같은 문제가 관찰된다.

**표면 토큰 편중 문제:** 모델이 "nigger", "bitch" 같은 명시적 비하어가 포함된 문장은 높은 정확도로 탐지하지만, 비하어 없이 맥락상 혐오를 전달하는 문장(예: "those people should go back to their countries")에서는 탐지 성능과 설명 품질이 모두 떨어진다.

**XAI 도구의 구조적 한계:** SHAP과 LIME은 각각 개별 토큰의 기여도를 측정하는 additive feature attribution 방식이므로, 단어들의 조합이 만들어내는 맥락적 혐오를 포착하는 데 근본적인 한계가 있다. 예를 들어, "dirty immigrants should leave"라는 문장에서 어떤 단어 하나를 개별적으로 제거해도 예측이 크게 변하지 않을 수 있으나, 이들의 조합이 혐오를 구성하고 있다.

이 문제를 해결하기 위해, SHAP과 LIME에 **마스킹 기반 인과적 검증**을 추가한 3축 검증 프레임워크를 제안한다.

---

## 2. 핵심 아이디어: XAI 3축 검증

기존의 SHAP + LIME 교차 검증은 "두 기법이 같은 토큰을 중요하다고 보는가"(설명 안정성)만 확인할 수 있었다. 여기에 마스킹 검증을 추가하면, 모델이 해당 토큰을 실제로 "이해하고" 보는 것인지까지 검증할 수 있다.

```
┌────────────────────────────────────────────────────────────────┐
│                    XAI 3축 검증 프레임워크                       │
│                                                                │
│   [축 1] SHAP          어떤 토큰이 예측에 기여했는가?            │
│          (기여도 분해)   Shapley value 기반 토큰별 기여도 정량화  │
│                                                                │
│   [축 2] LIME          독립적으로 봤을 때 어떤 단어가 중요한가?   │
│          (로컬 근사)    단어 perturbation 기반 로컬 선형 모델     │
│                                                                │
│   [축 3] Masking       그 토큰들이 실제로 예측을 좌우하는가?      │
│          (인과적 검증)  토큰 제거/유지 시 예측 변화 직접 측정     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

세 축은 각각 다른 질문에 답한다.

**SHAP** — "어떤 토큰이 예측에 기여했는가?" 게임 이론의 Shapley value를 통해 각 토큰의 예측 기여도를 정량화한다. 토크나이저 단위로 동작하므로 서브워드 수준의 세밀한 분석이 가능하지만, 결과가 서브워드 조각으로 파편화될 수 있다.

**LIME** — "독립적으로 봤을 때 어떤 단어가 중요한가?" 단어를 하나씩 제거하면서 예측 변화를 관찰하고, 이를 로컬 선형 모델로 근사한다. 단어 단위로 동작하여 해석이 직관적이지만, 단어 간 상호작용(interaction) 효과를 포착하지 못한다. 맥락적 혐오에서 LIME의 기여도가 전반적으로 낮게 나오는 이유이다.

**마스킹** — "그 토큰들이 실제로 예측을 좌우하는가?" SHAP과 LIME이 "이 토큰이 중요하다"고 말할 때, 실제로 그 토큰을 가렸을 때 예측이 변하는지를 직접 측정한다. 이는 SHAP/LIME의 설명에 대한 인과적 검증(causal verification)이다.

이 세 축의 교차를 통해 다음과 같은 판단이 가능하다.

| SHAP | LIME | 마스킹 | 해석 |
|:---:|:---:|:---:|---|
| 중요 | 중요 | 예측 변화 큼 | 모델이 해당 토큰을 실제로 이해하고 사용함 (신뢰 가능) |
| 중요 | 중요 | 예측 변화 작음 | XAI 도구가 잘못 짚음 — 모델은 다른 곳에 의존 (허위 설명) |
| 중요하지 않음 | 중요하지 않음 | 예측 변화 큼 | 분산된 맥락 의존 — 개별 토큰이 아닌 조합이 중요 (맥락적 혐오 패턴) |
| 불일치 | 불일치 | 검증 필요 | 설명 불안정 — 추가 분석 필요 |

---

## 3. 마스킹 검증 상세 설계

### 3.1 Comprehensiveness (포괄성)

XAI 도구가 중요하다고 판단한 Top-k 토큰을 제거했을 때 예측 확률이 얼마나 감소하는지를 측정한다. 이 값이 높으면 XAI 설명이 모델의 실제 판단 근거를 충실하게 반영한 것이고, 낮으면 XAI가 잘못 짚은 것이다.

```
Comprehensiveness = P(y | x) - P(y | x \ Top-k)

예시:
  원본:     P(hate | "those dirty immigrants should leave")  = 0.85
  Top-5 제거: P(hate | "those [MASK] [MASK] should [MASK]")   = 0.40
  결과:     0.85 - 0.40 = 0.45 (높음 → XAI 설명이 충실)
```

(DeYoung et al., 2020)

### 3.2 Sufficiency (충분성)

Top-k 토큰만 남기고 나머지를 모두 마스킹했을 때 예측이 유지되는지를 측정한다. 이 값이 낮으면 Top-k만으로 예측이 충분하다는 의미이고, 높으면 맥락에 의존하는 부분이 크다는 의미이다.

```
Sufficiency = P(y | x) - P(y | Top-k only)

예시:
  원본:        P(hate | "those dirty immigrants should leave")  = 0.85
  Top-5만 유지: P(hate | "[MASK] dirty immigrants [MASK] leave") = 0.72
  결과:        0.85 - 0.72 = 0.13 (낮음 → Top-5만으로 충분)
```

### 3.3 Slur-Free Prediction (맥락 의존도 측정)

문장 내 모든 명시적 비하어를 마스킹한 뒤 모델의 예측 변화를 관찰한다. 이를 통해 모델이 표면 토큰에 의존하는 정도와 맥락을 이해하는 정도를 직접적으로 분리할 수 있다.

```
Slur-Free Score = P(y | x \ slurs) / P(y | x)

예시:
  원본:         P(hate | "niggers are ruining this country")  = 0.92
  비하어 마스킹: P(hate | "[MASK] are ruining this country")   = 0.55
  결과:         0.55 / 0.92 = 0.60 (60% 유지 → 맥락도 일부 활용)
```

이 지표를 Baseline(BERT-base)과 Improved(RoBERTa+VADER) 간에 비교하면, VADER 감성 피처의 추가가 맥락 의존도를 실제로 높였는지를 검증할 수 있다.

---

## 4. SHAP · LIME · 마스킹의 역할 분담

각 XAI 기법이 "맥락"에 대해 가지는 강점과 한계를 명확히 정리하면 다음과 같다.

### SHAP의 맥락 포착 능력

SHAP(Partition Explainer)는 계층적으로 토큰을 묶어가며 Shapley value를 계산하므로, 어느 정도의 토큰 간 상호작용을 반영할 수 있다. 그러나 최종 결과는 개별 토큰에 귀속되므로, "이 두 단어가 함께 있을 때만 중요하다"는 정보는 소실된다. 또한 토크나이저 단위(서브워드)로 동작하여 결과가 파편화되는 문제가 있다.

### LIME의 맥락 포착 한계

LIME은 단어를 독립적으로 제거하며 perturbation을 수행하므로, 단어 간 상호작용 효과를 원리적으로 포착할 수 없다. 맥락적 혐오 문장에서 개별 단어를 하나씩 빼도 예측이 크게 변하지 않는 경우, LIME은 모든 단어의 기여도를 낮게 산출하게 된다. 이는 LIME의 구조적 한계이지 모델의 한계가 아닐 수 있으며, 마스킹 검증을 통해 이를 구분할 수 있다.

### 마스킹이 보완하는 지점

마스킹은 SHAP/LIME이 "중요하다고 본 토큰"을 실제로 제거/유지함으로써, 해당 설명의 인과적 타당성을 직접 검증한다. 특히:

- SHAP/LIME이 높은 기여도를 부여했지만 마스킹 시 예측 변화가 작은 경우 → 허위 양성 설명 식별
- SHAP/LIME이 낮은 기여도를 부여했지만 마스킹 시 예측이 크게 변하는 경우 → 분산된 맥락 의존 패턴 식별
- Slur-Free 실험을 통해 모델의 "맥락 이해 수준"을 표면 토큰과 분리하여 정량화

---

## 5. 전체 파이프라인에서의 위치

```
Phase 1: 기존 XAI 분석
  ├── SHAP Top-5 추출 (서브워드 → 단어 aggregation 적용)
  ├── LIME Top-5 추출
  ├── Overlap@5 (설명 안정성: SHAP ∩ LIME)
  └── Human Rationale 비교 (설명 타당성: Model Top-5 vs Human)

Phase 2: 마스킹 검증 (신규 추가)
  ├── Comprehensiveness (Top-k 제거 → 예측 변화)
  ├── Sufficiency (Top-k만 유지 → 예측 유지 여부)
  └── Slur-Free Prediction (비하어 제거 → 맥락 의존도)

Phase 3: 종합 분석
  ├── 3축 교차 분석 (SHAP × LIME × 마스킹)
  ├── Baseline vs Improved 비교
  │   ├── 설명 안정성 변화 (Overlap@5)
  │   ├── 설명 타당성 변화 (Human Rationale)
  │   ├── 설명 충실성 변화 (Comprehensiveness)
  │   └── 맥락 의존도 변화 (Slur-Free Score)
  └── 한계점 분석 (LIME의 상호작용 미포착 등)
```

---

## 6. 설명 안정성과 설명 타당성 평가 기준

해당 기준은 절대적인 정답을 제공하는 것은 아니지만, 설명의 신뢰성과 해석 가능성을 평가하기 위한 기준으로 사용한다.

### 설명 안정성 (Stability)

서로 다른 XAI 기법(SHAP, LIME)에서 추출된 중요 토큰 간의 일치도를 Jaccard similarity를 통해 정량화한다. Top-5 기준에서 과반수 이상(3개 이상)의 토큰이 일치하는 경우를 설명 결과의 안정성이 확보된 것으로 간주한다. 이는 설명 결과의 신뢰성을 간접적으로 검증한다.

```
Overlap@k = |A ∩ B| / |A ∪ B|
(A: SHAP Top-k, B: LIME Top-k)
Jaccard ≥ 0.6 → 상대적으로 안정적인 설명으로 판단
```

### 설명 타당성 (Validity)

HateXplain 데이터셋의 인간 주석 기반 human rationale과의 일치도를 측정하여, 모델이 중요하게 판단한 토큰이 인간의 판단과 얼마나 일치하는지를 통해 평가한다. 이는 모델 설명이 인간 판단과 얼마나 정합적인지를 평가한다.

```
Model Top-5 vs Human rationale overlap 측정
overlap ≥ 0.5 → 인간 판단 근거와 일정 수준 정렬된 설명으로 해석
```

인간 rationale 대비 감성·맥락 토큰 정렬이 낮은 경우, 모델이 의미적 맥락보다 표면 단서(욕설 등)에 편중되어 있을 가능성을 시사한다.

### 설명 충실성 (Faithfulness) — 마스킹 기반

XAI 도구의 설명이 모델의 실제 추론 과정을 얼마나 충실하게 반영하는지를 Comprehensiveness와 Sufficiency를 통해 직접 검증한다. 이는 설명 안정성·타당성과 독립적인 제3의 평가 축이다.

---

## 7. 개선 모델과 검증 대상

### Baseline 모델
| 모델 | 역할 |
|---|---|
| BERT-base | 순수 Transformer 기반 baseline |

### 개선 모델
| 모델 | 의미 |
|---|---|
| BERT+VADER | 감성 맥락 정보 보완을 통한 설명 타당성 개선 검증 |
| RoBERTa+VADER | 모델 아키텍처 차이에 따른 성능 및 설명 변화 비교 |

### Ablation 통제
| 모델 | 의미 |
|---|---|
| BERT+MLP | VADER 없이 동일 MLP 구조 → VADER의 실질적 기여 분리 |

Before/After 비교를 통해:
- 욕설 토큰 중요도 감소 여부
- 감성/맥락 토큰 중요도 증가 여부
- 인간 rationale과의 정렬 개선 여부
- 맥락 의존도(Slur-Free Score) 향상 여부

를 정량적으로 확인한다.

---

## 8. 학술적 기여

이 프레임워크의 학술적 기여는 다음과 같다.

**첫째,** SHAP + LIME의 2축 교차 검증에 마스킹 기반 인과적 검증을 추가하여 3축 체계로 확장하였다. 기존 연구에서 Comprehensiveness/Sufficiency는 단독 평가 지표로 사용되었으나(DeYoung et al., 2020), 본 연구에서는 SHAP/LIME의 설명에 대한 직접적 검증 수단으로 결합한다.

**둘째,** LIME의 구조적 한계(단어 간 상호작용 미포착)를 마스킹 실험을 통해 정량적으로 증명하고, 이를 맥락적 혐오 탐지의 근본적 난제와 연결하여 분석한다.

**셋째,** Slur-Free Prediction이라는 실험을 통해 모델의 "표면 의존도 vs 맥락 이해도"를 직접 분리하여 측정하는 방법론을 제시한다. 이는 혐오표현 탐지 모델의 실질적 이해 능력을 평가하는 새로운 관점이다.

---

## 9. 참고 문헌

- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.
- DeYoung, J., et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. ACL.
- Mathew, B., et al. (2021). HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection. AAAI.
- Cheng, L. (2022). Explainable Detection of Online Sexism. Virginia Tech.
