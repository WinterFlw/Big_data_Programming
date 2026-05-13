# 09. End-to-End XAI Evidence Bundle Agent Brief

> 역할: v2 end-to-end 파이프라인에서 XAI 산출물을 "그림 몇 장"이 아니라 재현 가능한 evidence bundle로 묶는다.

---

## 1. 왜 이 문서가 필요한가

교수 피드백의 핵심은 다음 질문이다.

```text
TF-IDF baseline과 정확도 차이가 크지 않은데, 이 프로젝트의 강점은 무엇인가?
```

이 질문에 대한 v2의 답은 "정확도가 압도적으로 높다"가 아니다.

```text
TF-IDF는 강한 baseline이다.
따라서 본 프로젝트의 차별점은 단순 정확도 경쟁이 아니라,
모델 판단 근거를 추적하고 검증할 수 있는 XAI evidence bundle을 제공하는 것이다.
```

즉 v2 XAI의 목표는 예측 결과를 아래 질문에 답할 수 있는 증거 패키지로 바꾸는 것이다.

```text
모델이 어떤 토큰을 근거로 판단했는가?
그 근거는 SHAP/LIME 사이에서 안정적인가?
중요 토큰을 제거하면 예측이 실제로 흔들리는가?
인간 rationale과 모델 근거가 얼마나 정렬되는가?
모델이 소수 비하어 토큰에만 과의존하는가, 아니면 여러 맥락 단서를 쓰는가?
source/target subgroup에서 취약한 패턴이 있는가?
```

---

## 2. 에이전트에게 줄 첫 지시문

```text
당신은 HateSpeachStudy v2_15seed end-to-end 파이프라인의 XAI Evidence Bundle 구축 에이전트입니다.
목표는 xai-primary, xai-deep, xai-ablation 산출물을 통합하여 reports/dashboard가 바로 읽을 수 있는 full XAI evidence bundle을 생성하는 것입니다.
이 프로젝트는 실시간 UX 최적화가 아니라 연구 검증 파이프라인입니다.
따라서 basic/full on-off 모드를 만들지 말고, canonical run_id 아래 full bundle을 항상 저장하세요.
다만 dashboard/report에서는 bundle 중 핵심 요약만 노출하고, 상세 raw evidence는 drill-down 또는 artifact link로 제공하세요.
```

---

## 3. 반드시 읽을 문서

```text
docs/02_e2e_pipeline.md
docs/04_xai_protocol.md
docs/07_output_and_report_contract.md
docs/08_xai_report_template.md
docs/agent_tasks/00_common_agent_rules.md
docs/agent_tasks/03_xai_agent.md
docs/agent_tasks/06_integration_lead_agent.md
```

---

## 4. 설계 원칙

### 4.1 저장은 full, 노출은 요약

```text
저장: 가능한 XAI 증거를 full bundle로 남긴다.
노출: 보고서와 dashboard에는 핵심 주장과 대표 사례만 보여준다.
```

이유:

```text
프로젝트 평가는 실시간 응답 속도보다 검증 가능성이 중요하다.
나중에 교수 질문이 들어왔을 때 raw artifact로 되짚을 수 있어야 한다.
하지만 사용자 화면에 모든 수치를 동시에 뿌리면 설명 가능성이 아니라 정보 과부하가 된다.
```

### 4.2 XAI는 사후 검증이다

금지 표현:

```text
XAI가 모델 개선 원인을 증명했다.
XAI 결과를 보고 VADER를 설계했다.
Attention이 곧 모델의 진짜 이유다.
모델이 완전히 맥락을 이해한다.
```

권장 표현:

```text
XAI evidence bundle은 통제된 ablation 조건 간 설명 패턴 차이를 사후 검증한다.
여러 독립 지표가 같은 방향을 보이는지 확인한다.
성능 차이가 작더라도 판단 근거의 투명성과 취약성 분석을 제공한다.
```

### 4.3 정확도와 XAI의 역할을 분리한다

```text
benchmark: 성능 차이, seed variance, p-value, effect size를 담당한다.
XAI bundle: 판단 근거, 설명 안정성, faithfulness, rationale alignment, context dependence를 담당한다.
```

TF-IDF 대비 강점은 아래처럼 말한다.

```text
TF-IDF도 분류 성능은 강할 수 있다.
하지만 본 v2 파이프라인은 예측값뿐 아니라 근거 토큰, 설명 충실도, human rationale 정렬도,
단어 의존성, subgroup 취약성까지 함께 산출한다.
따라서 결과 해석과 오류 분석에 더 적합하다.
```

---

## 5. Canonical output root

모든 산출물은 run_id 내부에 저장한다.

```text
v2/outputs/experiments/v2_15seed/xai/
```

기존 top-level `outputs/xai/`는 canonical output으로 쓰지 않는다.

---

## 6. Evidence bundle 구조

기존 XAI stage 산출물은 유지한다.

```text
xai/
  samples/
    primary_samples.csv
    deep_samples.csv
    ablation_samples.csv
  primary/
    seed_level_metrics.csv
    paired_xai_tests.csv
    seed_stability.csv
  deep/
    xai_details.json
    case_summary.csv
    cases/
  ablation/
    xai_ablation_metrics.csv
  xai_summary.json
```

여기에 end-to-end 해석용 bundle을 추가한다.

```text
xai/evidence_bundle/
  xai_run_metadata.json
  xai_sample_manifest.csv
  xai_predictions.csv
  token_attributions.jsonl
  method_agreement.csv
  faithfulness_metrics.csv
  context_metrics.csv
  plausibility_metrics.csv
  subgroup_xai_metrics.csv
  xai_risk_flags.csv
  xai_claims.json
  xai_interpretation_cards.json
  xai_dashboard_bundle.json
  README.md
```

`xai/evidence_bundle/README.md`는 각 파일의 의미와 생성 시각을 설명하는 index 파일이다.

---

## 7. 파일별 계약

### 7.1 `xai_run_metadata.json`

목적: XAI 재현성.

필수 필드:

```text
run_id
created_at
commit_hash
config_path
manifest_path
data_split_hash
conditions
seeds
primary_sample_count
deep_sample_count
ablation_sample_count
xai_methods
shap_config
lime_config
masking_config
known_limitations
```

`known_limitations`에는 최소 아래를 포함한다.

```text
SHAP/LIME are perturbation-based approximations.
Human rationale is a plausibility proxy, not a ground truth of model reasoning.
Attention rollout is a support metric, not a direct causal explanation.
```

### 7.2 `xai_sample_manifest.csv`

목적: 어떤 샘플을 왜 분석했는지 방어.

필수 컬럼:

```text
sample_id
post_id
split
label
label_name
source
primary_target
sample_set
sample_policy
case_type
text_length
has_human_rationale
has_explicit_slur
```

값 규칙:

```text
sample_set: primary | deep | ablation | case
sample_policy: stratified_random | median_seed_deep | fixed_error_case | ablation_light
```

주의:

```text
정량 지표용 sample은 fixed_error에 치우치면 안 된다.
fixed_error sample은 qualitative case로만 사용한다.
```

### 7.3 `xai_predictions.csv`

목적: XAI 대상 샘플에서 모델이 실제로 무엇을 예측했는지 기록.

필수 컬럼:

```text
sample_id
condition
seed
checkpoint_path
true_label
pred_label
prob_hatespeech
prob_offensive
prob_normal
confidence
margin
is_correct
case_type
```

`confidence` 규칙:

```text
high: margin >= 0.30
medium: 0.10 <= margin < 0.30
low: margin < 0.10
```

### 7.4 `token_attributions.jsonl`

목적: SHAP/LIME 원자료 보존.

한 줄은 하나의 sample-condition-method 조합이다.

필수 필드:

```json
{
  "sample_id": "string",
  "condition": "D_B",
  "seed": 42,
  "method": "shap",
  "pred_label": "hatespeech",
  "tokens": [
    {"token": "example", "score": 0.123, "rank": 1, "is_human_rationale": true}
  ],
  "top_tokens": ["example"]
}
```

토큰 처리 기준:

```text
BERT ## subword, RoBERTa Ġ subword, SentencePiece ▁ prefix, word-level token을 구분한다.
word-level token을 RoBERTa subword로 오판해 이어붙이지 않는다.
```

### 7.5 `method_agreement.csv`

목적: 설명 안정성.

필수 컬럼:

```text
sample_id
condition
seed
overlap_at_5
overlap_at_10
jaccard_at_5
jaccard_at_10
agreement_level
```

`agreement_level` 규칙:

```text
high: overlap_at_5 >= 0.60
medium: 0.30 <= overlap_at_5 < 0.60
low: overlap_at_5 < 0.30
```

### 7.6 `faithfulness_metrics.csv`

목적: XAI가 찍은 토큰이 실제 예측에 영향을 주는지 검증.

필수 컬럼:

```text
sample_id
condition
seed
k_policy
comprehensiveness
sufficiency
loo_drop
mss
original_pred_prob
masked_pred_prob
kept_only_pred_prob
```

`k_policy` 권장값:

```text
top5
top10
top20pct
```

### 7.7 `context_metrics.csv`

목적: 단어 의존성 vs 맥락 단서 사용의 근거.

필수 컬럼:

```text
sample_id
condition
seed
ci
mss
interaction_strength
attention_rollout_entropy
context_pattern
```

`context_pattern` 예시:

```text
single_token_concentrated
mixed_evidence
context_distributed
inconclusive
```

해석 규칙:

```text
CI가 낮고 MSS/interaction/rollout entropy가 높으면 context_distributed 쪽으로 해석한다.
단, attention rollout 단독으로 맥락 이해를 주장하지 않는다.
```

### 7.8 `plausibility_metrics.csv`

목적: 모델 근거와 human rationale의 부합도.

필수 컬럼:

```text
sample_id
condition
seed
method
rationale_precision_at_5
rationale_recall_at_5
rationale_f1_at_5
rationale_iou_at_5
human_rationale_count
agreement
```

주의:

```text
normal class는 rationale이 없거나 부족할 수 있다.
Plausibility는 보조 지표로만 해석한다.
```

### 7.9 `subgroup_xai_metrics.csv`

목적: source/target별 XAI 패턴 차이.

필수 컬럼:

```text
group_type
group_value
condition
seed
sample_count
macro_f1
mean_ci
mean_comprehensiveness
mean_rationale_f1_at_5
mean_overlap_at_5
```

`group_type`:

```text
source
target
label
```

### 7.10 `xai_risk_flags.csv`

목적: 보고서/대시보드에서 조심해서 볼 샘플 목록.

필수 컬럼:

```text
sample_id
condition
seed
risk_flag
severity
reason
recommended_handling
```

권장 flag:

```text
low_confidence
shap_lime_disagreement
explicit_slur_dependence
low_rationale_alignment
masking_unfaithful
truncated_input
subgroup_low_performance
```

### 7.11 `xai_claims.json`

목적: 보고서에 쓸 주장을 수치 근거와 함께 관리.

필수 구조:

```json
{
  "claims": [
    {
      "claim_id": "context_shift_d_b_vs_a_b",
      "claim": "D_B shows less single-token concentration than A_B.",
      "status": "supported | weak | contradicted | inconclusive",
      "evidence": [
        {"metric": "ci", "direction": "down", "delta": -0.05},
        {"metric": "mss", "direction": "up", "delta": 1.2}
      ],
      "limitations": ["CI/MSS are post-hoc metrics, not causal proof."]
    }
  ]
}
```

주요 claim 후보:

```text
D_B improves or preserves performance over A_B.
D_B has lower attribution concentration than A_B.
D_B has higher faithfulness than A_B.
D_B aligns with human rationale at least as well as A_B.
VADER/Attention effects are consistent with ablation statistics.
```

### 7.12 `xai_interpretation_cards.json`

목적: dashboard와 보고서 사례 설명.

한 카드의 필수 필드:

```json
{
  "sample_id": "string",
  "condition": "D_B",
  "prediction_summary": "D_B predicted hatespeech with high confidence.",
  "evidence_summary": "Top attribution tokens include ...",
  "faithfulness_summary": "Masking top tokens reduced predicted probability by ...",
  "context_summary": "The attribution pattern is context-distributed / single-token-concentrated / inconclusive.",
  "risk_flags": ["explicit_slur_dependence"],
  "plain_explanation_ko": "보고서/대시보드용 2-3문장 설명"
}
```

### 7.13 `xai_dashboard_bundle.json`

목적: dashboard가 여러 CSV/JSON을 직접 뒤지지 않도록 하는 통합 view.

필수 필드:

```text
run_id
generated_at
summary
claims
primary_metric_table
ablation_metric_table
case_cards
risk_flag_counts
artifact_paths
limitations
```

---

## 8. 구현 순서

### Step 1. 기존 XAI stage 산출물을 보존한다

아래는 기존 계약 그대로 유지한다.

```text
xai-primary
xai-deep
xai-ablation
```

이 문서는 기존 stage를 갈아엎으라는 뜻이 아니다. 기존 stage 결과를 읽어 end-to-end evidence bundle을 합성하라는 뜻이다.

### Step 2. bundle synthesis stage를 추가한다

권장 CLI:

```bash
./run.sh e2e xai-bundle --run-id v2_15seed
```

`xai-bundle`은 학습이나 SHAP/LIME 재계산을 하지 않는다. 이미 생성된 XAI artifact를 읽어서 evidence bundle을 만든다.

### Step 3. `all` 실행에 포함한다

권장 순서:

```text
benchmark -> aggregate -> xai-primary -> xai-deep -> xai-ablation -> xai-bundle -> report -> dashboard
```

### Step 4. report/dashboard는 bundle을 우선 읽는다

```text
reports/final_report.md는 xai_claims.json과 xai_dashboard_bundle.json을 우선 사용한다.
dashboard/index.html은 xai_dashboard_bundle.json을 우선 사용한다.
세부 raw artifact는 artifact link로 연결한다.
```

---

## 9. Dashboard 노출 원칙

dashboard는 모든 raw 수치를 한 화면에 뿌리지 않는다.

첫 화면:

```text
XAI 핵심 결론 3-5개
claim status
A_B vs D_B 주요 metric delta
risk flag summary
대표 case cards
```

drill-down:

```text
token attribution table
faithfulness table
context metrics
plausibility metrics
subgroup metrics
raw artifact links
```

---

## 10. 최종 보고서 문장 템플릿

교수 질문 대응용 기본 문장:

```text
TF-IDF와 최종 정확도 차이가 크지 않은 것은 맞습니다.
다만 본 프로젝트의 강점은 단순 분류 성능이 아니라, 모델 판단 근거를 XAI evidence bundle로 검증할 수 있다는 점입니다.
SHAP/LIME attribution, masking faithfulness, human rationale alignment, context metrics(CI/MSS/interaction), subgroup 분석을 통해
모델이 어떤 단어와 맥락에 의존했는지 확인할 수 있습니다.
따라서 본 프로젝트는 정확도만 높은 분류기가 아니라, 판단 근거를 분석 가능한 혐오표현 탐지 파이프라인을 목표로 합니다.
```

주의 문장:

```text
XAI는 성능 개선의 인과적 증명이 아니라, 조건 간 판단 패턴 차이를 보여주는 사후 근거다.
여러 독립 지표가 같은 방향을 가리킬 때만 moderate/strong evidence로 표현한다.
```

---

## 11. 완료 기준

아래 파일이 생성되어야 한다.

```text
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_run_metadata.json
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_sample_manifest.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_predictions.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/token_attributions.jsonl
outputs/experiments/v2_15seed/xai/evidence_bundle/method_agreement.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/faithfulness_metrics.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/context_metrics.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/plausibility_metrics.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/subgroup_xai_metrics.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_risk_flags.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_interpretation_cards.json
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json
outputs/experiments/v2_15seed/xai/evidence_bundle/README.md
```

최소 검증 명령:

```bash
python3 -m compileall pipeline scripts/validate_commit_message.py
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_run_metadata.json >/tmp/xai_meta_check.json
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json >/tmp/xai_claims_check.json
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json >/tmp/xai_dashboard_check.json
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

---

## 12. 금지 사항

```text
basic/full 모드를 새로 만들지 않는다.
기존 top-level outputs/xai를 canonical output으로 쓰지 않는다.
fixed_error 사례를 전체 XAI 결론으로 과대 해석하지 않는다.
XAI 지표 하나만으로 strong claim을 만들지 않는다.
통계 artifact가 없는데 "유의하다"는 표현을 쓰지 않는다.
Attention heatmap을 모델의 실제 이유라고 단정하지 않는다.
```

---

## 13. 에이전트 완료 보고 양식

```text
[xai evidence bundle report]
Run id:
Files created:
Source artifacts read:
Bundle artifacts created:
Report/dashboard integration:
Commands run:
Pass:
Fail:
Remaining limitation:
Ready for final report: yes/no
```
