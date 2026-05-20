# 06. Execution Runbook

> 목적: v2 실험을 실제로 돌릴 때 어떤 순서로 실행하고, 어디서 실패를 확인하며, 언제 다음 단계로 넘어갈 수 있는지 정한다.

---

## 1. 실행 전 원칙

v2 실행은 한 번에 모든 것을 돌리는 방식보다 stage별로 끊어서 검증하는 방식을 기본으로 한다.

권장 순서:

```text
plan -> data -> benchmark -> aggregate -> xai-primary -> xai-deep -> xai-ablation -> xai-bundle -> report -> dashboard
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

서버에서는 먼저 Python을 명시적으로 고정한다. `python3`가 시스템 Python으로
잡히면 `numpy`, `torch`, `matplotlib`가 없는 환경으로 preflight가 실패할 수 있다.

```bash
cd v2
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# RTX 5090은 Blackwell(sm_120)이므로 CUDA 12.8 wheel을 우선 설치한다.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
export PYTHON_BIN="$PWD/.venv/bin/python"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

`run.sh`, `daily.sh`, `run_5090_dual.sh`, `run_5090_e2e_batch.sh`는 공통으로
`scripts/env_defaults.sh`를 읽는다. RunPod에서 `/workspace`가 보이면
HuggingFace/Torch/pip/matplotlib 캐시는 자동으로 아래 영구 경로에 저장된다.

```text
/workspace/.cache/hatespeech-v2/
```

따라서 사람이 매번 `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`를 export할 필요는 없다.
다른 영구 경로를 쓰고 싶을 때만 실행 전에 `RUNPOD_PERSISTENT_ROOT` 또는
`HATESPEECH_CACHE_ROOT`를 명시한다.

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
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --execute --resume
```

전체 benchmark:

```bash
./run.sh e2e benchmark --run-id v2_15seed --execute --resume
```

RunPod에서 RTX 5090 두 장만 잡히는 경우에는 단일 명령 하나로 multi-GPU 학습을
시키지 않는다. v2 학습 코드는 condition x seed 단위의 단일 GPU 실행을 기준으로
작성되어 있으므로, GPU 0과 GPU 1에 서로 다른 condition shard를 맡긴다.

2x RTX 5090 권장 분할:

```text
GPU 0: A_B, B_B, C_B, D_B  # BERT family
GPU 1: A_R, B_R, C_R, D_R  # RoBERTa family
```

두 GPU가 모두 보이는지 먼저 확인한다.

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

자동 shard runner:

```bash
./scripts/run_5090_dual.sh
```

서버를 오래 잡아둘 수 있고, benchmark 이후 산출물까지 한 번에 이어서 만들고
싶다면 full E2E batch runner를 사용한다. 이 스크립트는 GPU 수를 자동 감지한다.
2장이 보이면 BERT/RoBERTa shard를 병렬 실행하고, 1장만 보이면 같은 8조건을
순차 실행한다. 따라서 2장짜리 RTX 5090이 unavailable이면 1장 Pod에서도 같은
명령으로 진행할 수 있다.

```bash
./scripts/run_5090_e2e_batch.sh
```

이 명령은 아래 순서로 실행한다.

```text
daily preflight
-> smoke benchmark(seed 42; 2GPU이면 병렬, 1GPU이면 순차)
-> full benchmark(2GPU이면 병렬, 1GPU이면 순차)
-> status
-> aggregate
-> xai-primary
-> xai-deep
-> xai-ablation
-> xai-bundle
-> report
-> dashboard
```

XAI는 benchmark보다 오래 걸릴 수 있으므로, 먼저 학습과 통계까지만 끝내고 싶으면
아래처럼 실행한다.

```bash
RUN_XAI=0 ./scripts/run_5090_e2e_batch.sh
```

smoke 없이 바로 full benchmark부터 들어가려면 아래처럼 실행한다. 단, 첫 서버
실행에서는 권장하지 않는다.

```bash
RUN_SMOKE=0 ./scripts/run_5090_e2e_batch.sh
```

1GPU fallback에서 실제 benchmark 명령은 아래와 같은 의미다.

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,B_B,C_B,D_B,A_R,B_R,C_R,D_R --execute --resume
```

smoke만 2GPU로 나눠 돌리고 싶으면 seed를 제한한다.

```bash
SEEDS=42 ./scripts/run_5090_dual.sh
```

수동 tmux 실행이 필요하면 터미널 두 개에서 아래처럼 나눠 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,B_B,C_B,D_B --execute --resume
CUDA_VISIBLE_DEVICES=1 ./run.sh e2e benchmark --run-id v2_15seed --conditions A_R,B_R,C_R,D_R --execute --resume
```

두 shard가 끝난 뒤에는 공유 상태표를 다시 만든다.

```bash
./run.sh e2e status --run-id v2_15seed
./run.sh e2e aggregate --run-id v2_15seed
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
./run.sh e2e xai-bundle --run-id v2_15seed
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
xai/evidence_bundle/xai_claims.json generated
xai/evidence_bundle/xai_dashboard_bundle.json generated
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

---

## 11. 학습 중 모니터링 — history.csv 28컬럼 활용 ★

작업 라운드 3차 갱신(2026-05-17)으로 `outputs/.../benchmark/runs/<cond>/seed_<n>/history.csv`가 에폭마다 28개 컬럼을 박는다. 학습 중·후 모니터링 명령.

### 11.1 학습 중 실시간 콘솔 출력

학습 시작 후 stdout에 다음 한 줄이 매 epoch 끝에 출력된다:

```text
[epoch] D_B | seed=42 | epoch=3/5 | train_loss=0.4521 | val_loss=0.6234 | val_macro_f1=0.6892
       | train_acc=0.7245 | val_acc=0.6960 | lr=1.20e-05 | grad_norm=2.314 | epoch_t=287.3s
```

**즉시 진단**:
- `train_acc - val_acc > 0.15` → overfit 의심
- `grad_norm > 5.0` 자주 → 학습률 너무 높음
- `epoch_t > 600s` (NVIDIA 기준) → 데이터 로더 병목 의심

### 11.2 학습 후 history.csv 직접 열기

```bash
# 한 condition × seed의 학습 곡선 한 눈에
column -t -s, outputs/experiments/v2_15seed/benchmark/runs/d_b/seed_42/history.csv | less -S

# Loss·F1만 빠르게
cut -d',' -f1,3,4,11,12 outputs/experiments/v2_15seed/benchmark/runs/d_b/seed_42/history.csv | column -t -s,

# Confusion matrix flatten 만
cut -d',' -f1,20-28 outputs/experiments/v2_15seed/benchmark/runs/d_b/seed_42/history.csv | column -t -s,
```

### 11.3 학습 곡선 시각화 (5번 Author용)

```python
# scripts/figures.py 안에서
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

run_id = "v2_15seed"
base = Path(f"outputs/experiments/{run_id}/benchmark/runs")

# 1. 한 조건의 학습 곡선 (loss + macro F1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for cond in ["A_B", "D_B"]:
    df = pd.read_csv(base / cond.lower() / "seed_42" / "history.csv")
    ax1.plot(df["epoch"], df["train_loss"], label=f"{cond} train", linestyle="--")
    ax1.plot(df["epoch"], df["val_loss"], label=f"{cond} val")
    ax2.plot(df["epoch"], df["val_macro_f1"], label=cond)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Macro F1"); ax2.legend()

# 2. Per-class F1 추이 (D_B 한 조건)
df = pd.read_csv(base / "d_b" / "seed_42" / "history.csv")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df["epoch"], df["val_f1_hatespeech"], label="hatespeech", color="C3")
ax.plot(df["epoch"], df["val_f1_offensive"], label="offensive", color="C1")
ax.plot(df["epoch"], df["val_f1_normal"], label="normal", color="C2")
ax.set_xlabel("Epoch"); ax.set_ylabel("Per-class F1"); ax.legend()
```

### 11.4 이상 학습 자동 감지 (1번 QA)

`daily.sh` 안에서 학습 결과 점검할 때 (선택 추가 가능):

```bash
# grad_norm_max가 너무 큰 unit 찾기
for f in outputs/experiments/v2_15seed/benchmark/runs/*/seed_*/history.csv; do
    max_grad=$(awk -F',' 'NR>1 {print $10}' "$f" | sort -n | tail -1)
    if (( $(echo "$max_grad > 5.0" | bc -l) )); then
        echo "[WARN] high grad_norm_max=$max_grad in $f"
    fi
done

# train_acc와 val_acc 격차가 큰 unit (overfit)
for f in outputs/experiments/v2_15seed/benchmark/runs/*/seed_*/history.csv; do
    last_train_acc=$(awk -F',' 'END {print $9}' "$f")
    last_val_acc=$(awk -F',' 'END {print $15}' "$f")
    # 격차 > 0.15 경고
    ...
done
```

---

## 12. 런팟 (또는 휘발성 GPU 인스턴스) 보존 정책 ★

NVIDIA 서버가 학교 PC가 아니라 런팟 같은 클라우드 GPU 인스턴스면 **인스턴스 종료 시 디스크가 휘발**한다. 산출물 손실을 막기 위한 정책.

### 무엇이 git으로 보존되나 (작아서 push OK)

`.gitignore` 정책 (2026-05-17 갱신)에 따라 다음 산출물은 매 stage 완료 후 git push로 보존 가능:

```text
outputs/experiments/v2_15seed/
├── execution_status.csv / failed_runs.csv / completed_runs.csv
├── manifest.json / plan_status.json
├── benchmark/
│   ├── benchmark_runs.csv / benchmark_summary.csv
│   ├── paired_tests.csv / paired_tests_holm.csv
│   ├── anova_2way_bert.csv / anova_2way_roberta.csv / anova_3way.csv
│   └── runs/<cond>/seed_<n>/
│       ├── metrics.json          ★ 평가 지표 핵심
│       ├── history.csv           ★ 에포크별 학습 곡선
│       ├── run_config.json       ★ 재학습 가능한 하이퍼파라미터
│       ├── predictions.csv       ★ 오분류 분석 입력
│       ├── stdout.log / stderr.log
├── xai/
│   ├── samples/{primary,deep,ablation}_samples.csv
│   ├── primary/{seed_level_metrics,sample_level_metrics,paired_xai_tests,seed_stability}.csv
│   ├── deep/case_summary.csv + xai_details.json
│   ├── ablation/xai_ablation_metrics.csv
│   ├── evidence_bundle/* (15 파일)
│   ├── xai_summary.json
│   └── .cache/<cond>_seed_<seed>.json  ★ SHAP/LIME 결과, 재계산 비용 큰 영역
├── reports/final_report.md/docx
└── dashboard/index.html
```

대략 50~200MB 정도. git push로 충분.

### 무엇이 git에 안 올라가나 (무거워서 별도 관리)

```text
outputs/experiments/v2_15seed/benchmark/checkpoints/*.pt    # 120개 × 400MB = 48GB
outputs/experiments/v2_15seed/benchmark/runs/**/*.pt        # epoch별 checkpoint
outputs/experiments/v2_15seed/benchmark/runs/**/predictions.pkl  # CSV로 대체됨
outputs/experiments/v2_15seed/xai/.cache/*.npy / *.pkl     # 재계산 가능
```

### 런팟에서 산출물 빼내는 흐름 (2번 학습 실행자 책임)

```bash
# 1. 학습 stage 완료 후 즉시 git push (작은 산출물 보존)
cd v2
git add outputs/experiments/v2_15seed/
git commit -m "chore(outputs): A_B seed 42 metrics + history + predictions push"
git push origin main

# 2. 다음 stage 진행 — aggregate / xai-primary / xai-bundle / report / dashboard
./run.sh e2e aggregate --run-id v2_15seed
git add outputs/experiments/v2_15seed/benchmark/*.csv
git commit -m "chore(outputs): aggregate CSVs"
git push

# 3. (선택) 체크포인트가 발표 후에도 필요하면 외부 스토리지로
#    a) 학교 NAS / Google Drive: rclone
#    b) S3: aws s3 sync
#    c) 단순 다운로드: scp / runpodctl
rclone copy outputs/experiments/v2_15seed/benchmark/checkpoints/ gdrive:hate_v2/

# 4. 인스턴스 종료 전 체크리스트
echo "=== 런팟 종료 전 점검 ==="
echo "1. git status — 모든 outputs CSV/JSON push 완료?"
echo "2. 체크포인트 외부 스토리지 동기화 완료?"
echo "3. final_report.md + dashboard/index.html git 반영 완료?"
```

### 매 stage 완료 후 git push 의무화 (2번)

```bash
# 권장 alias
alias push_outputs='git add outputs/experiments/v2_15seed/ && git commit -m "chore(outputs): stage progress" && git push origin main'

# 흐름
./run.sh e2e benchmark --execute --resume
push_outputs   # 학습 결과 즉시 보존

./run.sh e2e aggregate
push_outputs

./run.sh e2e xai-primary --resume
push_outputs   # .cache/까지 보존됨

./run.sh e2e xai-bundle && report && dashboard
push_outputs   # 최종 보고서·대시보드
```

### 발표 후 체크포인트 폐기 가능 여부

| 산출물 | 발표 후 필요? | 폐기 가능? |
|---|---|---|
| metrics.json / history.csv / run_config.json | ✓ 평가용 보관 | ✗ (작아서 git에 영구 보존) |
| predictions.csv | ✓ 오분류 분석용 | ✗ (작아서 git 보존) |
| `.cache/` SHAP/LIME JSON | ✓ XAI 재현용 | ✗ (git 보존) |
| **`.pt` checkpoint** | ✗ 재학습으로 복구 가능 | ✓ **학기 종료 후 폐기 OK** |

체크포인트 폐기해도 metrics + run_config + predictions로 재현 가능. 평가위원이 "다시 돌려봐" 하면 `./run.sh e2e benchmark --execute --resume`으로 재학습 (약 4~8시간 / seed × condition).

### 런팟 사용 안 할 때 (학교 PC GPU)

학교 PC GPU라면 인스턴스 종료 개념 없으므로 위 정책이 덜 critical. 다만 매 stage 후 git push는 그대로 유지 (1번/3번/4번/5번이 산출물 받는 통로).
