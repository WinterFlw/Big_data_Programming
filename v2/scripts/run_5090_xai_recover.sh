#!/usr/bin/env bash
# Regenerate only the checkpoints needed for v2 XAI, then run XAI/report steps.
#
# Use this after a storage-safe benchmark run finished with
# CHECKPOINT_RETENTION=none. It avoids rerunning all 120 checkpoints with
# keep-all, which is risky on a 50GB RunPod network volume.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)/.."

source "scripts/env_defaults.sh"

run_id="${RUN_ID:-v2_15seed}"
log_dir="${LOG_DIR:-outputs/experiments/${run_id}/server_logs}"
post_xai_prune="${POST_XAI_PRUNE:-1}"
run_backup="${RUN_BACKUP:-1}"
mkdir -p "$log_dir"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="${log_dir}/${timestamp}_xai_recover.log"
exec > >(tee -a "$log_path") 2>&1

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
export CHECKPOINT_RETENTION="${CHECKPOINT_RETENTION:-xai-minimal}"

if [[ "$CHECKPOINT_RETENTION" == "none" ]]; then
    echo "STOP: XAI recovery needs checkpoints. Use CHECKPOINT_RETENTION=xai-minimal."
    exit 1
fi

best_primary_pair_seeds() {
    "$PYTHON_BIN" - <<PY
import csv
from pathlib import Path

run_id = "${run_id}"
path = Path("outputs") / "experiments" / run_id / "benchmark" / "benchmark_runs.csv"
scores = {}

with path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        condition = row.get("condition", "")
        if condition not in {"A_B", "D_B"}:
            continue
        try:
            seed = int(row.get("seed", ""))
            macro_f1 = float(row.get("macro_f1", ""))
        except ValueError:
            continue
        scores.setdefault(seed, {})[condition] = macro_f1

ranked = []
for seed, values in scores.items():
    if "A_B" not in values or "D_B" not in values:
        continue
    paired_mean = (values["A_B"] + values["D_B"]) / 2
    ranked.append((paired_mean, values["D_B"], values["A_B"], seed))

ranked.sort(reverse=True)
top = [seed for _, _, _, seed in ranked[:2]]
if not top:
    raise SystemExit(f"No completed A_B/D_B benchmark rows found in {path}")
print(",".join(str(seed) for seed in top))
PY
}

if [[ "${XAI_TOP4:-0}" == "1" ]]; then
    # Qualitative emergency mode: benchmark statistics remain 15-seed, but XAI
    # explains only the two best paired A_B/D_B seeds, i.e. four checkpoints.
    export XAI_FAST=1
    export XAI_PRIMARY_SEEDS="${XAI_PRIMARY_SEEDS:-$(best_primary_pair_seeds)}"
    export RUN_XAI_DEEP="${RUN_XAI_DEEP:-0}"
    export RUN_XAI_ABLATION="${RUN_XAI_ABLATION:-0}"
fi

if [[ "${XAI_FAST:-0}" == "1" ]]; then
    # Emergency mode for maintenance windows: keep the benchmark's 15-seed result,
    # but downsize XAI evidence generation so report/dashboard artifacts finish.
    export XAI_PRIMARY_MAX_SEEDS="${XAI_PRIMARY_MAX_SEEDS:-3}"
    export XAI_PRIMARY_SAMPLE_SIZE="${XAI_PRIMARY_SAMPLE_SIZE:-30}"
    export XAI_DEEP_SAMPLE_SIZE="${XAI_DEEP_SAMPLE_SIZE:-60}"
    export XAI_ABLATION_SAMPLE_SIZE="${XAI_ABLATION_SAMPLE_SIZE:-20}"
    export XAI_SHAP_MAX_EVALS="${XAI_SHAP_MAX_EVALS:-150}"
    export XAI_LIME_NUM_SAMPLES="${XAI_LIME_NUM_SAMPLES:-150}"
    export XAI_INTERACTION_PAIRS="${XAI_INTERACTION_PAIRS:-15}"
    export XAI_ABLATION_METRIC_SAMPLE_SIZE="${XAI_ABLATION_METRIC_SAMPLE_SIZE:-20}"
fi

run_xai_primary="${RUN_XAI_PRIMARY:-1}"
run_xai_deep="${RUN_XAI_DEEP:-1}"
run_xai_ablation="${RUN_XAI_ABLATION:-1}"

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

cuda_device_count() {
    "$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

middle_seed() {
    "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

run_id = "${run_id}"
manifest_path = Path("outputs") / "experiments" / run_id / "manifest.json"
if not manifest_path.exists():
    manifest_path = Path("configs") / f"{run_id}.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
seeds = [int(seed) for seed in manifest["benchmark"]["seeds"]]
print(seeds[len(seeds) // 2])
PY
}

run_benchmark() {
    local gpu="$1"
    local conditions="$2"
    local seeds_value="${3:-}"
    local seed_args=()
    if [[ -n "$seeds_value" ]]; then
        seed_args=(--seeds "$seeds_value")
    fi
    env CUDA_VISIBLE_DEVICES="$gpu" CHECKPOINT_RETENTION="$CHECKPOINT_RETENTION" \
        ./run.sh e2e benchmark \
            --run-id "$run_id" \
            --conditions "$conditions" \
            "${seed_args[@]}" \
            --execute \
            --force
}

run_parallel_pair() {
    local left_name="$1"
    local left_gpu="$2"
    local left_conditions="$3"
    local left_seeds="${4:-}"
    local right_name="$5"
    local right_gpu="$6"
    local right_conditions="$7"
    local right_seeds="${8:-}"

    echo "Launching ${left_name} on GPU ${left_gpu}: ${left_conditions} seeds=${left_seeds:-all}"
    run_benchmark "$left_gpu" "$left_conditions" "$left_seeds" &
    local left_pid="$!"

    echo "Launching ${right_name} on GPU ${right_gpu}: ${right_conditions} seeds=${right_seeds:-all}"
    run_benchmark "$right_gpu" "$right_conditions" "$right_seeds" &
    local right_pid="$!"

    local status=0
    wait "$left_pid" || status=1
    wait "$right_pid" || status=1
    return "$status"
}

echo "RunPod RTX 5090 XAI recovery"
echo "run_id=${run_id}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CHECKPOINT_RETENTION=${CHECKPOINT_RETENTION}"
echo "POST_XAI_PRUNE=${post_xai_prune}"
echo "RUN_BACKUP=${run_backup}"
echo "XAI_FAST=${XAI_FAST:-0}"
echo "XAI_TOP4=${XAI_TOP4:-0}"
echo "SKIP_CHECKPOINT_RECOVERY=${SKIP_CHECKPOINT_RECOVERY:-0}"
echo "RUN_XAI_PRIMARY=${run_xai_primary}"
echo "RUN_XAI_DEEP=${run_xai_deep}"
echo "RUN_XAI_ABLATION=${run_xai_ablation}"
if [[ "${XAI_FAST:-0}" == "1" ]]; then
    echo "XAI_PRIMARY_SEEDS=${XAI_PRIMARY_SEEDS:-auto}"
    echo "XAI_PRIMARY_MAX_SEEDS=${XAI_PRIMARY_MAX_SEEDS}"
    echo "XAI_PRIMARY_SAMPLE_SIZE=${XAI_PRIMARY_SAMPLE_SIZE}"
    echo "XAI_DEEP_SAMPLE_SIZE=${XAI_DEEP_SAMPLE_SIZE}"
    echo "XAI_ABLATION_SAMPLE_SIZE=${XAI_ABLATION_SAMPLE_SIZE}"
    echo "XAI_SHAP_MAX_EVALS=${XAI_SHAP_MAX_EVALS}"
    echo "XAI_LIME_NUM_SAMPLES=${XAI_LIME_NUM_SAMPLES}"
fi
echo "log=${log_path}"

gpu_count="$(cuda_device_count)"
seed_mid="$(middle_seed)"
primary_recovery_seeds=""
if [[ "${XAI_TOP4:-0}" == "1" ]]; then
    primary_recovery_seeds="${XAI_PRIMARY_SEEDS}"
fi
echo "CUDA GPUs: ${gpu_count}"
echo "middle seed for deep/ablation XAI: ${seed_mid}"
if [[ -n "$primary_recovery_seeds" ]]; then
    echo "primary recovery seeds: ${primary_recovery_seeds}"
fi

if [[ "${SKIP_CHECKPOINT_RECOVERY:-0}" == "1" ]]; then
    echo "Skipping checkpoint recovery because SKIP_CHECKPOINT_RECOVERY=1."
elif [[ "$gpu_count" -ge 2 ]]; then
    run_step "regenerate primary XAI checkpoints A_B/D_B" \
        run_parallel_pair "A_B primary" 0 "A_B" "$primary_recovery_seeds" "D_B primary" 1 "D_B" "$primary_recovery_seeds"
    if [[ "$run_xai_deep" == "1" || "$run_xai_ablation" == "1" ]]; then
        run_step "regenerate median ablation checkpoints" \
            run_parallel_pair "BERT median" 0 "B_B,C_B" "$seed_mid" "RoBERTa median" 1 "A_R,B_R,C_R,D_R" "$seed_mid"
    fi
else
    run_step "regenerate primary XAI checkpoints A_B/D_B" \
        run_benchmark 0 "A_B,D_B" "$primary_recovery_seeds"
    if [[ "$run_xai_deep" == "1" || "$run_xai_ablation" == "1" ]]; then
        run_step "regenerate median ablation checkpoints" \
            run_benchmark 0 "B_B,C_B,A_R,B_R,C_R,D_R" "$seed_mid"
    fi
fi

run_step "status after checkpoint recovery" ./run.sh e2e status --run-id "$run_id"
run_step "aggregate after checkpoint recovery" ./run.sh e2e aggregate --run-id "$run_id"
if [[ "$run_xai_primary" == "1" ]]; then
    run_step "xai-primary" env CUDA_VISIBLE_DEVICES=0 ./run.sh e2e xai-primary --run-id "$run_id" --resume
else
    echo "Skipping xai-primary because RUN_XAI_PRIMARY=${run_xai_primary}."
fi
if [[ "$run_xai_deep" == "1" ]]; then
    run_step "xai-deep" env CUDA_VISIBLE_DEVICES=0 ./run.sh e2e xai-deep --run-id "$run_id" --resume
else
    echo "Skipping xai-deep because RUN_XAI_DEEP=${run_xai_deep}."
fi
if [[ "$run_xai_ablation" == "1" ]]; then
    run_step "xai-ablation" env CUDA_VISIBLE_DEVICES=0 ./run.sh e2e xai-ablation --run-id "$run_id" --resume
else
    echo "Skipping xai-ablation because RUN_XAI_ABLATION=${run_xai_ablation}."
fi
run_step "xai-bundle" ./run.sh e2e xai-bundle --run-id "$run_id"
run_step "report" ./run.sh e2e report --run-id "$run_id"
run_step "dashboard" ./run.sh e2e dashboard --run-id "$run_id"

if [[ "$post_xai_prune" == "1" ]]; then
    run_step "post-XAI checkpoint prune" "$PYTHON_BIN" scripts/prune_checkpoints.py --run-id "$run_id" --policy none
fi

if [[ "$run_backup" == "1" ]]; then
    run_step "essential backup" ./scripts/backup_essential_outputs.sh
fi

echo ""
echo "XAI recovery complete."
echo "Log: ${log_path}"
