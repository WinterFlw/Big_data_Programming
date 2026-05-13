# 06. Execution Runbook

> 목적: v2 실험을 실제로 돌릴 때 어떤 순서로 실행하고, 어디서 실패를 확인하며, 언제 다음 단계로 넘어갈 수 있는지 정한다.

---

## 1. 실행 전 원칙

v2 실행은 한 번에 모든 것을 돌리는 방식보다 stage별로 끊어서 검증하는 방식을 기본으로 한다.

권장 순서:

```text
plan -> data -> benchmark -> aggregate -> xai-primary -> xai-deep -> xai-ablation -> report -> dashboard
```

선택 stage:

```text
freeze
```

`freeze`는 핵심 결론에 필요하면 포함하고, 시간이나 GPU 상황이 빠듯하면 별도 보조 실험으로 분리한다.

---

## 2. Preflight checklist

실행 전에 아래를 확인한다.

```text
Python environment가 활성화되어 있는가?
필요 패키지가 설치되어 있는가?
GPU/CUDA 사용 가능 여부가 확인되었는가?
HateXplain dataset 경로가 맞는가?
train/val/test split이 고정되어 있는가?
기존 outputs와 새 outputs가 섞이지 않는가?
manifest_template.json이 실제 manifest.json으로 복사/확정되었는가?
```

데이터 관련 preflight:

```text
sample count
label distribution
source distribution
target distribution
rationale availability
empty text count
token length distribution
split hash
```

모델 관련 preflight:

```text
BERT/RoBERTa model name
max_length
batch_size
learning_rate
epochs
attention alpha
sentiment feature dimension
random seed propagation
```

---

## 3. 권장 CLI 흐름

초기 계획 생성:

```bash
./run.sh e2e plan --run-id v2_15seed
```

데이터 검증:

```bash
./run.sh e2e data --run-id v2_15seed
```

짧은 dry run:

```bash
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
```

첫 실제 smoke run:

```bash
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --resume
```

전체 benchmark:

```bash
./run.sh e2e benchmark --run-id v2_15seed --resume
```

집계:

```bash
./run.sh e2e aggregate --run-id v2_15seed
```

XAI:

```bash
./run.sh e2e xai-primary --run-id v2_15seed --resume
./run.sh e2e xai-deep --run-id v2_15seed
./run.sh e2e xai-ablation --run-id v2_15seed
```

최종 산출물:

```bash
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

---

## 4. 완료 판정

개별 학습 run 완료 조건:

```text
metrics.json exists
history.csv exists
predictions.pkl or predictions.csv exists
checkpoint exists
run_config.json exists
stderr.log has no fatal error
```

benchmark stage 완료 조건:

```text
8 conditions x 15 seeds = 120 completed runs
benchmark_runs.csv contains 120 rows
benchmark_summary.csv contains all 8 conditions
no missing primary metric
```

aggregate stage 완료 조건:

```text
paired_tests.csv generated
paired_tests_holm.csv generated
CI columns generated
effect size columns generated
ANOVA tables generated when applicable
```

XAI 완료 조건:

```text
primary sample set fixed
same sample ids used across seeds
seed_level_metrics.csv generated
paired_xai_tests.csv generated
seed_stability.csv generated
case files generated for selected qualitative examples
```

---

## 5. Resume와 실패 복구

기본 정책:

```text
--resume: 완료된 run은 건너뛴다.
--force: 지정된 run을 다시 실행한다.
--only-failed: 실패한 run만 다시 실행한다.
```

실패 run 판정:

```text
metrics.json 없음
checkpoint 없음
metrics.json parsing 실패
primary metric 없음
log에 RuntimeError/CUDA out of memory/NaN loss 기록
```

실패 복구 순서:

```text
1. 실패 run 목록 생성
2. 실패 원인 분류
3. 데이터/설정 오류면 전체 중단
4. 일시적 GPU/OOM이면 batch_size 조정 또는 해당 run 재실행
5. 실패 run만 재실행
6. aggregate 재실행
```

---

## 6. 권장 로그 파일

각 run 폴더에는 아래 로그를 둔다.

```text
stdout.log
stderr.log
run_config.json
environment.json
metrics.json
history.csv
```

run_id 루트에는 아래 파일을 둔다.

```text
manifest.json
execution_status.csv
failed_runs.csv
completed_runs.csv
```

---

## 7. 예상 실행 규모

기본 benchmark:

```text
8 conditions x 15 seeds = 120 training runs
```

선택 freeze:

```text
2 variants x 15 seeds = 30 training runs
```

Primary XAI:

```text
2 models x 15 seeds x 200 samples
```

Deep XAI:

```text
2 models x 1 median seed x 500 samples
```

Ablation XAI:

```text
8 models x 1 median seed x 50 samples
```

실제 시간은 GPU 종류, max_length, batch_size, epoch 수에 크게 의존한다. 따라서 첫 smoke run에서 아래 값을 기록한 뒤 전체 시간을 추정한다.

```text
single_run_train_minutes
single_run_eval_minutes
xai_seconds_per_sample
```

추정식:

```text
benchmark_hours = 120 * single_run_total_minutes / 60 / parallel_gpu_count
primary_xai_hours = 2 * 15 * 200 * xai_seconds_per_sample / 3600 / parallel_gpu_count
```

---

## 8. 중간 점검 시점

전체 benchmark 중 아래 시점에서 중간 점검한다.

```text
2 runs completed: A_B seed 42, D_B seed 42
16 runs completed: 8 conditions x seed 42
60 runs completed: roughly half benchmark
120 runs completed: full benchmark
```

각 시점에서 확인할 것:

```text
metric scale이 정상인가?
loss가 NaN으로 터지지 않았는가?
condition별 결과 파일 구조가 동일한가?
A_B와 D_B의 prediction 파일 스키마가 같은가?
seed별 runtime 편차가 지나치게 크지 않은가?
```

---

## 9. 중단 기준

아래 상황이면 전체 실행을 멈추고 설계를 다시 확인한다.

```text
데이터 split hash가 실행마다 달라짐
같은 seed에서 조건별 데이터 순서가 달라짐
label leakage 의심
attention loss가 rationale 없는 샘플에 잘못 적용됨
VADER feature가 A/B 조건에도 들어감
metrics schema가 condition마다 다름
```

---

## 10. 완료 후 할 일

전체 실행 후에는 아래 순서로 확인한다.

```text
1. benchmark_summary.csv 확인
2. paired_tests_holm.csv 확인
3. XAI seed_stability.csv 확인
4. final_report.md 확인
5. dashboard/index.html 확인
6. manifest.json과 execution_status.csv 보관
```

top-level `outputs/dashboard`나 `outputs/reports`로 복사하는 것은 마지막 단계에서만 한다. canonical output은 항상 `outputs/experiments/v2_15seed/`다.

