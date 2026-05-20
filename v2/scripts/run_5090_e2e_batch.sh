#!/usr/bin/env bash
# One-command RunPod batch runner for the v2 pipeline on RTX 5090.
#
# This script is intentionally boring:
#   1. Validate the Python/CUDA environment.
#   2. Optionally run a tiny smoke test.
#   3. Run the benchmark on 2 GPUs when available, or 1 GPU sequentially.
#   4. Merge outputs through status/aggregate.
#   5. Generate report and dashboard.
#
# The default is storage-safe for a small RunPod network volume: no XAI and no
# retained checkpoints. Enable XAI explicitly after increasing storage or when
# you accept the extra checkpoint/cache footprint.
#
# It does not make a single model use multiple GPUs. When 2 GPUs are available
# it runs independent condition x seed jobs in parallel. When only 1 GPU is
# available, it runs the same jobs sequentially with the same output contract.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)/.."

source "scripts/env_defaults.sh"

run_id="${RUN_ID:-v2_15seed}"
run_smoke="${RUN_SMOKE:-1}"
smoke_seeds="${SMOKE_SEEDS:-42}"
run_xai="${RUN_XAI:-0}"
xai_gpu="${XAI_GPU:-0}"
log_dir="${LOG_DIR:-outputs/experiments/${run_id}/server_logs}"
all_conditions="${ALL_CONDITIONS:-A_B,B_B,C_B,D_B,A_R,B_R,C_R,D_R}"
checkpoint_retention="${CHECKPOINT_RETENTION:-none}"
post_xai_prune="${POST_XAI_PRUNE:-1}"
export CHECKPOINT_RETENTION="$checkpoint_retention"

mkdir -p "$log_dir"
batch_timestamp="$(date +%Y%m%d_%H%M%S)"
batch_log="${log_dir}/${batch_timestamp}_e2e_batch.log"

exec > >(tee -a "$batch_log") 2>&1

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
    python_bin="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
else
    echo "Python interpreter not found in PATH."
    exit 1
fi
export PYTHON_BIN="$python_bin"

run_step() {
    local name="$1"
    shift
    echo ""
    echo "============================================================"
    echo "[$(date '+%F %T')] START ${name}"
    echo "============================================================"
    "$@"
    echo "[$(date '+%F %T')] DONE ${name}"
}

check_cuda() {
    "$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"gpu[{index}]:", torch.cuda.get_device_name(index))
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("STOP: at least 1 CUDA GPU is required for this batch runner.")
PY
}

cuda_device_count() {
    "$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

run_single_gpu_benchmark() {
    local name="$1"
    local seeds_value="${2:-}"
    local seed_args=()
    if [[ -n "$seeds_value" ]]; then
        seed_args=(--seeds "$seeds_value")
    fi
    env CUDA_VISIBLE_DEVICES=0 ./run.sh e2e benchmark \
        --run-id "$run_id" \
        --conditions "$all_conditions" \
        "${seed_args[@]}" \
        --execute \
        --resume
}

run_best_available_benchmark() {
    local name="$1"
    local seeds_value="${2:-}"
    local gpu_count
    gpu_count="$(cuda_device_count)"
    if [[ "$gpu_count" -ge 2 ]]; then
        if [[ -n "$seeds_value" ]]; then
            env SEEDS="$seeds_value" ./scripts/run_5090_dual.sh
        else
            ./scripts/run_5090_dual.sh
        fi
    else
        echo "Only ${gpu_count} CUDA GPU detected. Running ${name} sequentially on GPU 0."
        run_single_gpu_benchmark "$name" "$seeds_value"
    fi
}

echo "RunPod RTX 5090 v2 E2E batch"
echo "run_id=${run_id}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "HATESPEECH_CACHE_ROOT=${HATESPEECH_CACHE_ROOT}"
echo "RUN_SMOKE=${run_smoke}"
echo "SMOKE_SEEDS=${smoke_seeds}"
echo "RUN_XAI=${run_xai}"
echo "XAI_GPU=${xai_gpu}"
echo "CHECKPOINT_RETENTION=${CHECKPOINT_RETENTION}"
echo "POST_XAI_PRUNE=${post_xai_prune}"
echo "batch_log=${batch_log}"

if [[ "$run_xai" != "1" ]]; then
    echo "Storage-safe default: XAI is disabled and checkpoints are not retained."
    echo "For XAI, rerun with: RUN_XAI=1 CHECKPOINT_RETENTION=xai-minimal"
fi

if [[ "$run_xai" == "1" && "$CHECKPOINT_RETENTION" == "none" ]]; then
    echo "STOP: CHECKPOINT_RETENTION=none deletes checkpoints needed by XAI."
    echo "Use CHECKPOINT_RETENTION=xai-minimal, or set RUN_XAI=0 for metrics-only runs."
    exit 1
fi

run_step "cuda check" check_cuda
run_step "daily preflight" ./scripts/daily.sh

if [[ "$run_smoke" == "1" ]]; then
    echo ""
    echo "Running smoke test first. Completed smoke units will be skipped by the full run."
    run_step "best available GPU smoke benchmark" run_best_available_benchmark "smoke" "$smoke_seeds"
else
    echo "Skipping smoke test because RUN_SMOKE=${run_smoke}."
fi

run_step "best available GPU full benchmark" run_best_available_benchmark "full" ""
run_step "status after benchmark" ./run.sh e2e status --run-id "$run_id"
run_step "aggregate benchmark" ./run.sh e2e aggregate --run-id "$run_id"
if [[ "$CHECKPOINT_RETENTION" != "keep-all" && "$CHECKPOINT_RETENTION" != "all" ]]; then
    run_step "checkpoint prune (${CHECKPOINT_RETENTION})" "$PYTHON_BIN" scripts/prune_checkpoints.py --run-id "$run_id" --policy "$CHECKPOINT_RETENTION"
fi

if [[ "$run_xai" == "1" ]]; then
    echo ""
    echo "Running XAI sequentially on visible GPU index ${xai_gpu}."
    run_step "xai-primary" env CUDA_VISIBLE_DEVICES="$xai_gpu" ./run.sh e2e xai-primary --run-id "$run_id" --resume
    run_step "xai-deep" env CUDA_VISIBLE_DEVICES="$xai_gpu" ./run.sh e2e xai-deep --run-id "$run_id" --resume
    run_step "xai-ablation" env CUDA_VISIBLE_DEVICES="$xai_gpu" ./run.sh e2e xai-ablation --run-id "$run_id" --resume
    run_step "xai-bundle" ./run.sh e2e xai-bundle --run-id "$run_id"
    if [[ "$post_xai_prune" == "1" ]]; then
        run_step "post-XAI checkpoint prune" "$PYTHON_BIN" scripts/prune_checkpoints.py --run-id "$run_id" --policy none
    fi
else
    echo "Skipping XAI because RUN_XAI=${run_xai}."
fi

run_step "report" ./run.sh e2e report --run-id "$run_id"
run_step "dashboard" ./run.sh e2e dashboard --run-id "$run_id"
run_step "final status" ./run.sh e2e status --run-id "$run_id"

echo ""
echo "E2E batch complete."
echo "Log: ${batch_log}"
echo "Report: outputs/experiments/${run_id}/reports/final_report.md"
echo "Dashboard: outputs/experiments/${run_id}/dashboard/index.html"
