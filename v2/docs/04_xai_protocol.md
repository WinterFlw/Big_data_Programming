# 04. XAI Protocol

---

## 1. XAI의 역할

XAI는 모델 설계 입력이 아니라 **사후 검증 도구**다.

금지되는 흐름:

```text
XAI 진단 -> 모델 설계 변경 -> 다시 XAI
```

본 프로젝트의 흐름:

```text
선행연구 기반 가설 -> 통제된 ablation -> XAI 사후 검증
```

XAI는 다음 질문에 답한다.

```text
모델이 어떤 토큰에 주목했는가?
그 토큰은 인간 rationale과 정렬되는가?
중요 토큰을 제거하면 예측이 흔들리는가?
seed가 바뀌어도 설명 패턴이 유지되는가?
이 모든 정보를 report/dashboard가 바로 읽을 수 있는 evidence bundle로 남길 수 있는가?
```

---

## 2. XAI stage 구성

### 2.1 Primary XAI

핵심 비교를 seed 반복으로 검증한다.

```text
Models: A_B vs D_B
Seeds: 15
Samples: 200 stratified samples
Methods: SHAP, LIME, rationale alignment, masking
```

목적:

```text
XAI metric의 평균과 seed variance 추정
```

### 2.2 Deep XAI

정성적 사례와 자세한 설명을 만든다.

```text
Models: A_B vs D_B
Seed: median-performing seed
Samples: 500 stratified samples
Methods: SHAP, LIME, case plots
```

최고 성능 seed를 쓰지 않고 median-performing seed를 사용한다. cherry-picking을 피하기 위해서다.

### 2.3 Ablation XAI Matrix

8조건 전체를 가볍게 비교한다.

```text
Models: A_B, B_B, C_B, D_B, A_R, B_R, C_R, D_R
Seed: median-performing seed per condition
Samples: 50
Methods: CI, MSS, LOO, attention rollout entropy
```

목적:

```text
전체 조건 간 자동 XAI metric 패턴 확인
```

### 2.4 Improvement Case XAI

성능 개선 사례를 별도 분석한다.

```text
Case type: A_B wrong, D_B correct
Samples: 50
Use: qualitative report only
```

이 결과는 대표 사례 설명용이다. 전체 통계 결론의 근거로 과도하게 쓰지 않는다.

### 2.5 XAI Evidence Bundle

Primary, Deep, Ablation XAI가 끝나면 이를 별도 `xai-bundle` stage에서 통합한다.

```text
Input:
- xai-primary outputs
- xai-deep outputs
- xai-ablation outputs

Output:
- xai/evidence_bundle/xai_claims.json
- xai/evidence_bundle/xai_dashboard_bundle.json
- xai/evidence_bundle/token_attributions.jsonl
- xai/evidence_bundle/faithfulness_metrics.csv
- xai/evidence_bundle/context_metrics.csv
- xai/evidence_bundle/plausibility_metrics.csv
```

이 stage는 SHAP/LIME을 다시 계산하는 단계가 아니라, **이미 생성된 XAI 산출물을 재현 가능한 근거 묶음으로 정리하는 단계**다.

---

## 3. 샘플링 원칙

Representative XAI sample은 fixed_error에 치우치면 안 된다.

권장 500 sample 구성:

```text
label stratified:
- hatespeech about 170
- offensive about 170
- normal about 160

case type coverage:
- both correct
- A_B wrong, D_B correct
- both wrong
- model disagreement
```

Primary XAI 200 sample도 같은 원칙으로 축소 샘플링한다.

---

## 4. XAI 지표

### 4.1 Attribution Agreement

```text
SHAP-LIME Overlap@5
SHAP-LIME Overlap@10
```

서로 다른 XAI 방법이 같은 토큰을 중요하게 보는지 평가한다.

### 4.2 Human Rationale Alignment

```text
Precision@5
Recall@5
F1@5
Token coverage
```

모델 attribution과 인간 rationale의 정렬도를 평가한다.

### 4.3 Faithfulness

```text
Comprehensiveness
Sufficiency
Leave-one-out drop
```

중요 토큰 제거/유지만으로 예측 확률이 얼마나 바뀌는지 측정한다.

### 4.4 Context Learning

```text
CI: attribution concentration
MSS: minimum sufficient subset
Interaction strength
Attention rollout entropy
```

단일 단어 과의존이 아니라 복수 단서와 맥락을 함께 쓰는지 본다.

### 4.5 Seed Stability

```text
Top-k Jaccard across seeds
Rank correlation across seeds
Rationale overlap variance
Faithfulness metric variance
```

같은 sample, 같은 condition에서 seed만 바뀌었을 때 설명이 유지되는지 확인한다.

---

## 5. SHAP 토큰 처리 기준

SHAP output은 tokenizer subword일 수도 있고, word-level token일 수도 있다. 따라서 토큰 집계 함수는 다음을 구분해야 한다.

```text
BERT subword: ## prefix
RoBERTa subword: Ġ prefix
SentencePiece: ▁ prefix
Word-level token: prefix 없음, 공백 단위
```

prefix가 없는 word-level token을 RoBERTa subword로 오판해 이어붙이면 안 된다.

검증 예시:

```text
["hello", "world"] -> ["hello", "world"]
["mu", "##zzi", "##e"] -> ["muzzie"]
```

---

## 6. XAI 산출물

권장 산출물:

```text
xai/primary/seed_level_metrics.csv
xai/primary/paired_xai_tests.csv
xai/primary/seed_stability.csv
xai/deep/xai_details.json
xai/deep/case_summary.csv
xai/deep/cases/*.png
xai/ablation/xai_ablation_metrics.csv
xai/samples/primary_samples.csv
xai/samples/deep_samples.csv
xai/xai_summary.json
xai/evidence_bundle/xai_run_metadata.json
xai/evidence_bundle/xai_claims.json
xai/evidence_bundle/xai_dashboard_bundle.json
xai/evidence_bundle/token_attributions.jsonl
xai/evidence_bundle/faithfulness_metrics.csv
xai/evidence_bundle/context_metrics.csv
xai/evidence_bundle/plausibility_metrics.csv
```

report/dashboard는 raw deep case보다 `xai_claims.json`과 `xai_dashboard_bundle.json`을 우선 소비한다.

---

## 7. 보고서 해석 원칙

XAI는 “성능 개선의 원인을 증명”하지 않는다. XAI는 모델 판단 패턴이 가설과 일관적인지 보여주는 사후 근거다.

권장 표현:

```text
XAI 분석 결과는 D_B가 baseline보다 human rationale과 더 정렬되는 경향을 보였다.
중요 토큰 제거 실험에서 D_B의 attribution이 예측 확률에 더 큰 영향을 보였다.
seed stability 분석에서 설명 패턴이 단일 checkpoint에만 의존하지 않음을 확인했다.
TF-IDF가 강한 baseline임에도, v2는 판단 근거와 취약성을 함께 제공하는 XAI evidence bundle을 남긴다.
```

금지 표현:

```text
XAI가 VADER 추가를 증명했다.
XAI 결과를 바탕으로 VADER를 설계했다.
모델이 완전히 맥락을 이해한다.
```
