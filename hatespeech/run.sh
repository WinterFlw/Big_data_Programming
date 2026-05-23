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
    "$python_bin" -u main.py "$command" "$@"
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
  tune           Run v2.1 tuning (BERT+MLP base, B_B alpha grid, D_B+Target beta grid)
  benchmark      Run v2.1 benchmarks (TF-IDF + A/B/C/D for BERT and RoBERTa + D_B target aux)
  freeze-study   Compare frozen vs fine-tuned encoder as a secondary control
  xai            Run v2.1 XAI 4-axis post-hoc verification
  dashboard      Build the interactive HTML dashboard from saved artifacts
  all            Run data -> vader -> eda -> [tune] -> benchmark -> freeze-study -> xai -> dashboard
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
