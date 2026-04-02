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
  data|vader|tune|benchmark|freeze-study|xai|dashboard|all|status|clean)
    "$python_bin" -u run_experiments.py "$command" "$@"
    ;;
  fresh-all)
    ./run_fresh_full.sh "$@"
    ;;
  svm|bert|bert-v|roberta-v|baselines|improved)
    echo "[info] '$command' is now routed to the repeated benchmark suite."
    "$python_bin" -u run_experiments.py benchmark "$@"
    ;;
  xai1|xai2)
    echo "[info] '$command' is now routed to the integrated XAI stage."
    "$python_bin" -u run_experiments.py xai "$@"
    ;;
  hatebert|hatebert-v)
    echo "[info] '$command' was replaced by the report-aligned freeze-study / benchmark flow."
    "$python_bin" -u run_experiments.py freeze-study "$@"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: ./run.sh [command]

Main commands:
  data           Prepare 70/10/20 stratified splits and data profile outputs
  vader          Extract VADER features for all splits
  tune           Run sequential hyperparameter tuning
  benchmark      Run repeated benchmarks (TF-IDF+LR, TF-IDF+SVM, BERT-base, BERT+VADER, RoBERTa+VADER)
  freeze-study   Compare BERT+VADER with frozen vs fine-tuned encoder
  xai            Run SHAP + LIME comparison and Overlap@5 analysis
  dashboard      Build the interactive HTML dashboard from saved artifacts
  all            Run data -> vader -> benchmark -> freeze-study -> xai
  fresh-all      Clean generated artifacts and rerun the full pipeline from scratch
  status         Show current artifact status
  clean          Remove outputs/ and checkpoints/

Compatibility aliases:
  svm, bert, bert-v, roberta-v, baselines, improved -> benchmark
  xai1, xai2                                      -> xai
  hatebert, hatebert-v                            -> freeze-study

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
