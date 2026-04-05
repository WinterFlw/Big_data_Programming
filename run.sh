#!/bin/zsh
# Report-aligned experiment runner

set -e
cd "${0:a:h}"

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

command="${1:-status}"
shift || true

case "$command" in
  data|vader|eda|tune|benchmark|freeze-study|xai|dashboard|all|status|clean)
    "$python_bin" -u run_experiments.py "$command" "$@"
    ;;
  fresh-all)
    ./run_fresh_full.sh "$@"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: ./run.sh [command]

Main commands:
  data           Prepare 70/10/20 stratified splits and data profile outputs
  vader          Extract VADER features for all splits
  eda            Exploratory data analysis (text length, VADER by class, targets, vocab overlap)
  tune           Run sequential hyperparameter tuning
  benchmark      Run repeated benchmarks (TF-IDF baselines + BERT-base + BERT+MLP + BERT+VADER + RoBERTa+VADER)
  freeze-study   Compare BERT+VADER with frozen vs fine-tuned encoder
  xai            Run SHAP + LIME comparison and Overlap@5 analysis
  dashboard      Build the interactive HTML dashboard from saved artifacts
  all            Run data -> vader -> eda -> benchmark -> freeze-study -> xai -> dashboard
  fresh-all      Clean generated artifacts and rerun the full pipeline from scratch
  status         Show current artifact status
  clean          Remove outputs/ and checkpoints/

Optional flags:
  --force        Rebuild cached artifacts for the current command
  --with-tuning  With 'all', run tuning before the benchmark
EOF
    ;;
  *)
    echo "Unknown command: $command"
    echo "Run './run.sh help' for available commands."
    exit 1
    ;;
esac
