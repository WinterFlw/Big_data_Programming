#!/bin/zsh
# Fresh end-to-end training runner
# - Removes generated artifacts
# - Rebuilds data artifacts
# - Runs tuning, benchmark, freeze-study, and XAI in order

set -euo pipefail
cd "${0:a:h}"
export PYTHONUNBUFFERED=1

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

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="outputs/logs"
log_path="$log_dir/fresh_full_${timestamp}.log"

mkdir -p "$log_dir"

run_step() {
  local title="$1"
  shift

  echo ""
  echo "============================================================"
  echo "[STEP] $title"
  echo "============================================================"
  echo "[CMD] $*" | tee -a "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

echo "Fresh full pipeline started at $(date)" | tee "$log_path"
echo "Workspace: $(pwd)" | tee -a "$log_path"
echo "Python: $python_bin" | tee -a "$log_path"

# Generated artifacts only. Raw source data in data/ is kept.
run_step "Clean generated artifacts" "$python_bin" -u run_experiments.py clean
mkdir -p "$log_dir"
echo "Fresh full pipeline started at $(date)" | tee "$log_path"
echo "Workspace: $(pwd)" | tee -a "$log_path"
echo "Python: $python_bin" | tee -a "$log_path"

run_step "Prepare data split" "$python_bin" -u run_experiments.py data --force
run_step "Extract VADER features" "$python_bin" -u run_experiments.py vader --force
run_step "Hyperparameter tuning" "$python_bin" -u run_experiments.py tune
run_step "Repeated benchmark" "$python_bin" -u run_experiments.py benchmark
run_step "Freeze study" "$python_bin" -u run_experiments.py freeze-study
run_step "XAI report" "$python_bin" -u run_experiments.py xai
run_step "Dashboard refresh" "$python_bin" -u run_experiments.py dashboard
run_step "Final status" "$python_bin" -u run_experiments.py status

echo ""
echo "============================================================" | tee -a "$log_path"
echo "Fresh full pipeline finished at $(date)" | tee -a "$log_path"
echo "Log saved to: $(pwd)/$log_path" | tee -a "$log_path"
echo "============================================================" | tee -a "$log_path"
