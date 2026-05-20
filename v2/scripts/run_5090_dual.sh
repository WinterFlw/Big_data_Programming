#!/usr/bin/env bash
# RunPod 2x RTX 5090 runner for the v2 benchmark stage.
#
# Design:
#   GPU 0 runs BERT-family conditions:    A_B,B_B,C_B,D_B
#   GPU 1 runs RoBERTa-family conditions: A_R,B_R,C_R,D_R
#
# Each process sees only one CUDA device through CUDA_VISIBLE_DEVICES, so the
# existing single-GPU training code does not need DistributedDataParallel.
# Results are written to disjoint condition/seed folders. The shared
# execution_status.csv can be refreshed after both shards finish.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)/.."

run_id="${RUN_ID:-v2_15seed}"
bert_conditions="${BERT_CONDITIONS:-A_B,B_B,C_B,D_B}"
roberta_conditions="${ROBERTA_CONDITIONS:-A_R,B_R,C_R,D_R}"
seeds="${SEEDS:-}"
log_dir="${LOG_DIR:-outputs/experiments/${run_id}/server_logs}"

mkdir -p "$log_dir"

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

gpu_count="$("$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

if [[ "$gpu_count" -lt 2 ]]; then
    echo "Expected at least 2 CUDA GPUs, but detected ${gpu_count}."
    echo "Check RunPod GPU count and the PyTorch CUDA installation before starting."
    exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"

run_shard() {
    local gpu="$1"
    local shard_name="$2"
    local conditions="$3"
    local logfile="${log_dir}/${timestamp}_${shard_name}_gpu${gpu}.log"
    local seed_args=()

    if [[ -n "$seeds" ]]; then
        seed_args=(--seeds "$seeds")
    fi

    (
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHON_BIN="$PYTHON_BIN"
        echo "[$(date '+%F %T')] shard=${shard_name} gpu=${gpu} run_id=${run_id}"
        echo "conditions=${conditions}"
        if [[ -n "$seeds" ]]; then
            echo "seeds=${seeds}"
        else
            echo "seeds=manifest default"
        fi
        "$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("visible_gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("visible_gpu_name:", torch.cuda.get_device_name(0))
PY
        nvidia-smi || true
        ./run.sh e2e benchmark \
            --run-id "$run_id" \
            --conditions "$conditions" \
            "${seed_args[@]}" \
            --execute \
            --resume
        echo "[$(date '+%F %T')] shard=${shard_name} finished"
    ) 2>&1 | tee "$logfile" &

    local pid="$!"
    echo "$pid" > "${log_dir}/${timestamp}_${shard_name}_gpu${gpu}.pid"
    echo "Started ${shard_name} shard on GPU ${gpu}: pid=${pid}, log=${logfile}"
}

echo "RunPod dual RTX 5090 benchmark runner"
echo "run_id=${run_id}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "Detected CUDA GPUs: ${gpu_count}"
echo "Logs: ${log_dir}"

run_shard 0 "bert" "$bert_conditions"
sleep 5
run_shard 1 "roberta" "$roberta_conditions"

status=0
for job in $(jobs -p); do
    if ! wait "$job"; then
        status=1
    fi
done

echo "Refreshing full benchmark status after both shards."
./run.sh e2e status --run-id "$run_id"

if [[ "$status" -ne 0 ]]; then
    echo "One or more shards failed. Check logs under ${log_dir}."
    exit "$status"
fi

echo "Both shards completed. Next: ./run.sh e2e aggregate --run-id ${run_id}"
