# 07. Output and Report Contract

> 목적: v2 파이프라인이 어떤 파일을 만들고, 각 파일이 어떤 컬럼과 의미를 가져야 하는지 정한다. 이 문서는 코드 구현과 최종 보고서 생성을 연결하는 계약이다.

---

## 1. Canonical output root

v2의 기준 산출물 위치:

```text
outputs/experiments/v2_15seed/
```

이 폴더 안의 결과를 canonical로 본다. 기존 top-level 산출물은 필요할 때 export하는 view로만 취급한다.

---

## 2. 최상위 구조

```text
outputs/experiments/v2_15seed/
  manifest.json
  execution_status.csv
  data/
  benchmark/
  freeze/
  xai/
  reports/
  dashboard/
```

필수 파일:

```text
manifest.json
benchmark/benchmark_runs.csv
benchmark/benchmark_summary.csv
benchmark/paired_tests_holm.csv
xai/xai_summary.json
reports/final_report.md
dashboard/index.html
```

---

## 3. `benchmark_runs.csv`

목적: condition x seed 단위의 원시 결과 테이블.

필수 컬럼:

```text
run_id
condition
backbone
use_attention_loss
use_sentiment
seed
status
train_seconds
best_epoch
macro_f1
weighted_f1
accuracy
precision_macro
recall_macro
loss
checkpoint_path
metrics_path
predictions_path
```

권장 추가 컬럼:

```text
gpu_name
cuda_version
python_version
torch_version
commit_hash
data_split_hash
```

---

## 4. `benchmark_summary.csv`

목적: condition별 15 seed 요약.

필수 컬럼:

```text
condition
backbone
n_seeds
macro_f1_mean
macro_f1_std
macro_f1_ci_low
macro_f1_ci_high
weighted_f1_mean
accuracy_mean
best_seed
median_seed
failed_seed_count
```

해석 원칙:

```text
best_seed는 보고서 대표 성능으로 쓰지 않는다.
median_seed는 XAI deep analysis checkpoint 선택에 사용한다.
최종 성능 주장은 mean과 CI를 기준으로 한다.
```

---

## 5. `paired_tests.csv`와 `paired_tests_holm.csv`

목적: 같은 seed 안에서 조건 차이를 검정한다.

필수 컬럼:

```text
comparison
metric
condition_a
condition_b
n_pairs
mean_diff
std_diff
ci_low
ci_high
test_name
p_value
p_value_holm
effect_size
significant_0_05
```

주요 비교:

```text
A_B vs B_B
A_B vs C_B
A_B vs D_B
A_R vs B_R
A_R vs C_R
A_R vs D_R
B_B vs D_B
C_B vs D_B
```

보고서에서는 p-value만 단독으로 쓰지 않고 `mean_diff`, `CI`, `effect_size`를 같이 쓴다.

---

## 6. XAI output 구조

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
  evidence_bundle/
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
  xai_summary.json
```

---

## 7. `xai/evidence_bundle/`

목적: TF-IDF 대비 차별점인 "판단 근거를 분석 가능한 파이프라인"을 report/dashboard가 바로 읽을 수 있게 묶는다.

상세 계약은 아래 문서를 따른다.

```text
docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md
```

해석 원칙:

```text
raw XAI artifact는 full bundle로 저장한다.
report/dashboard는 xai_claims.json과 xai_dashboard_bundle.json을 우선 읽는다.
XAI는 성능 개선의 인과적 증명이 아니라 조건 간 판단 패턴 차이의 사후 근거다.
```

---

## 8. `xai/primary/seed_level_metrics.csv`

목적: A_B와 D_B의 XAI 지표를 seed별로 비교한다.

필수 컬럼:

```text
run_id
condition
seed
sample_count
shap_lime_overlap_at_5
shap_lime_overlap_at_10
rationale_precision_at_5
rationale_recall_at_5
rationale_f1_at_5
comprehensiveness
sufficiency
loo_drop
topk_jaccard_mean
rank_corr_mean
```

해석:

```text
rationale_f1_at_5가 높으면 인간 rationale과 더 잘 맞는 경향이다.
comprehensiveness가 높으면 중요 토큰 제거가 예측에 더 크게 영향을 준다.
seed stability 지표가 높으면 설명이 단일 checkpoint에 덜 의존한다.
```

---

## 9. `xai/deep/case_summary.csv`

목적: 최종 보고서에 들어갈 정성 사례를 관리한다.

필수 컬럼:

```text
sample_id
true_label
baseline_prediction
v2_prediction
case_type
baseline_confidence
v2_confidence
top_tokens_baseline
top_tokens_v2
human_rationale_tokens
plot_path
comment
```

case_type 예시:

```text
both_correct
baseline_wrong_v2_correct
both_wrong
model_disagreement
```

---

## 10. Final report 구조

`reports/final_report.md`는 아래 구조를 따른다.

```text
1. 연구 목적
2. 모델과 실험 조건
3. 데이터와 split
4. 15 seed 반복 실험 설계
5. 성능 결과
6. 통계 검정
7. XAI 분석
8. 오류 및 사례 분석
9. 한계
10. 재현 방법
```

`reports/final_report.docx`는 같은 내용을 Word 제출용으로 변환한 버전이다.

---

## 11. Dashboard 구조

`dashboard/index.html`은 아래 정보를 제공한다.

```text
run_id
execution status
condition summary
seed-level distribution
paired test results
XAI summary
case examples
artifact links
```

dashboard는 해석의 보조 도구다. 최종 canonical text는 `final_report.md`와 `final_report.docx`다.

---

## 11. 재현성 metadata

가능하면 모든 주요 산출물에 아래 metadata를 포함한다.

```text
run_id
created_at
code_commit
data_split_hash
manifest_hash
python_version
torch_version
transformers_version
device
```

이 metadata는 이후 결과가 “어떤 코드와 설정에서 나온 것인지”를 추적하기 위해 필요하다.
