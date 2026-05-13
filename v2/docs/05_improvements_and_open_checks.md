# 05. Improvements and Open Checks

---

## 1. 이번 redesign의 핵심 개선사항

### 1.1 산출물 분리

기존 top-level `outputs/`를 바로 덮어쓰지 않고 run_id 기반 output root를 둔다.

```text
outputs/experiments/v2_15seed/
```

효과:

```text
기존 3-seed 결과와 새 15-seed 결과 혼입 방지
재현성 강화
보고서/대시보드 기준 명확화
```

### 1.2 15 seed 반복

8조건을 15 seed로 반복한다.

```text
8 x 15 = 120 training runs
```

효과:

```text
학습 stochasticity 추정
paired comparison 검정력 강화
세부 ablation 비교 방어력 증가
```

### 1.3 Hyperparameter 통제

조건별 튜닝값을 섞지 않고 family별 공통값을 사용한다.

효과:

```text
VADER/Attention Loss 효과 해석 가능
ablation 설계의 내부 타당도 개선
```

### 1.4 XAI seed stability

XAI를 단일 checkpoint 해석에서 seed 반복 검증으로 확장한다.

효과:

```text
설명 패턴의 반복 가능성 확인
case visualization cherry-picking 위험 감소
```

### 1.5 Report/dashboard 재설계

최종 보고와 대시보드는 run_id 내부 artifact를 기준으로 생성한다.

효과:

```text
최종 결과 기준이 명확해짐
중간 산출물과 최종 산출물 분리
```

### 1.6 XAI evidence bundle stage 명시

Primary, Deep, Ablation XAI 결과를 그대로 흩어 두지 않고 별도 `xai-bundle` stage에서 통합한다.

효과:

```text
TF-IDF 대비 v2 강점을 "설명 가능한 근거 묶음"으로 방어 가능
report/dashboard가 raw case가 아니라 계약된 JSON/CSV를 직접 읽을 수 있음
XAI 결과를 그림 몇 장이 아니라 재현 가능한 산출물 묶음으로 남길 수 있음
```

---

## 2. 바꿔야 할 코드

### 2.1 `run.sh`

추가:

```text
e2e command
```

예:

```bash
./run.sh e2e all --run-id v2_15seed --resume
```

### 2.2 `run_experiments.py`

추가:

```text
e2e subcommand parser
run-id option
resume/force/dry-run option
stage dispatch
```

### 2.3 `experiment_core.py`

추가/수정:

```text
run_e2e_plan()
run_e2e_benchmark()
is_e2e_run_complete()
load_e2e_run_record()
aggregate_e2e_benchmark()
run_e2e_freeze()
```

핵심:

```text
output_root 인자 지원
checkpoint_path를 run_id 내부로 저장
condition x seed 단위 resume
family별 공통 hyperparameter 강제
```

### 2.4 `experiment_xai.py`

추가/수정:

```text
run_e2e_xai_primary()
run_e2e_xai_deep()
run_e2e_xai_ablation()
select_stratified_xai_samples()
select_median_seed()
compute_xai_seed_stability()
build_xai_evidence_bundle()
```

### 2.5 `utils.py`

추가:

```text
run_id path helpers
paired confidence interval
Holm correction
Cohen's dz
bootstrap CI
manifest load/save
```

### 2.6 `experiment_dashboard.py`

수정:

```text
run_id 기반 dashboard bundle 생성
15 seed 통계표 표시
XAI seed stability 표시
evidence bundle 요약 표시
```

---

## 3. 검증해야 할 것

### 3.1 데이터 검증

```text
split hash 고정
post_id cross-split overlap 없음
text cross-split overlap 없음
label distribution 유지
source/target distribution 기록
```

### 3.2 실험 설계 검증

```text
모든 condition이 동일 seed 목록 사용
BERT family hyperparameter 동일
RoBERTa family hyperparameter 동일
attention alpha 동일 전파
VADER 유무 외 입력 차이 없음
```

### 3.3 실행 검증

```text
120 training runs 모두 완료
각 run metrics/checkpoint/predictions 존재
resume가 완료 run을 건너뜀
aggregate가 학습 없이 재실행 가능
```

### 3.4 통계 검증

```text
paired tests common seed 기준
Holm correction 적용
CI 계산 정상
ANOVA에서 seed block 처리
표준편차 ddof 기준 명시
```

### 3.5 XAI 검증

```text
SHAP token aggregation sanity check
same sample set across seeds
representative sample과 fixed_error sample 분리
median seed 선택 기준 기록
XAI 실패 run 재시도 가능
evidence bundle JSON/JSONL/CSV contract 검증
```

### 3.6 최종 출력 검증

```text
final_report.md 생성
final_report.docx 생성
dashboard/index.html 생성
manifest와 report 설정 일치
모든 표가 v2_15seed artifact를 참조
xai_claims.json과 xai_dashboard_bundle.json이 report/dashboard 입력과 일치
```

---

## 4. 리스크

### 4.1 실행 시간

15 seed benchmark는 약 12시간, freeze 포함 시 약 14시간이다. XAI는 별도 수 시간이 필요할 수 있다.

대응:

```text
resume 필수
stage 분리
dry-run 지원
run별 로그 저장
```

### 4.2 XAI 비용

8조건 x 15 seed x 500 sample XAI는 과도하다.

대응:

```text
Primary XAI: A_B vs D_B x 15 seed x 200
Deep XAI: A_B vs D_B x median seed x 500
Ablation XAI: 8조건 x median seed x 50
```

### 4.3 결과 해석 과장

통계적으로 유의하지 않은 효과를 강하게 말하면 안 된다.

대응:

```text
p-value
Holm-adjusted p-value
effect size
95% CI
모두 함께 보고
```

### 4.4 기존 산출물 혼입

기존 `outputs/reports`를 읽으면 3-seed 결과와 섞일 수 있다.

대응:

```text
e2e stage는 run_id output root만 읽는다.
dashboard/report도 run_id를 필수 인자로 받는다.
```

---

## 5. 완료 기준

### 5.1 코드 완료

```text
./run.sh e2e plan --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --dry-run
./run.sh e2e aggregate --run-id v2_15seed
```

위 명령이 정상 동작한다.

### 5.2 실험 완료

```text
benchmark_runs.csv contains 120 transformer rows
all 15 seeds exist for all 8 conditions
paired_tests_holm.csv exists
anova outputs exist
```

### 5.3 XAI 완료

```text
primary seed-level XAI metrics exist
seed_stability.csv exists
deep XAI case plots exist
ablation XAI matrix exists
```

### 5.4 보고 완료

```text
final_report.md
final_report.docx
dashboard/index.html
```

세 파일이 모두 `outputs/experiments/v2_15seed/` 아래에 존재한다.
