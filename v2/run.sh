#!/usr/bin/env bash

set -e

# Always run from the v2 workspace root, even if the user calls
# ./v2/run.sh from the repository root. This keeps relative config/output paths
# stable and prevents accidental writes to the old top-level outputs/.
# (bash-compatible — the previous "${0:a:h}" was a zsh-only idiom.)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
cd "$script_dir"

# Keep model/dataset/package caches on persistent storage when RunPod mounts
# /workspace. This must happen before Python imports HuggingFace/PyTorch.
source "$script_dir/scripts/env_defaults.sh"

# PYTHON_BIN lets the server operator pin a virtualenv interpreter:
#
#   PYTHON_BIN=/path/to/venv/bin/python ./run.sh e2e status
#
# If it is not set, we choose python first and python3 second.
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

# The only real command today is e2e. The wrapper still has a command layer so
# we can later add v2-specific helpers such as "preflight" or "collect".
command="${1:-e2e}"
shift || true

case "$command" in
  e2e)
    # Use module execution so Python resolves imports from the local v2/
    # workspace. The package name is "pipeline" inside this folder.
    "$python_bin" -u -m pipeline.cli "$@"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: ./run.sh e2e [stage] [options]

Examples:
  ./run.sh e2e status --run-id v2_15seed
  ./run.sh e2e plan --run-id v2_15seed --force
  ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
  ./run.sh e2e aggregate --run-id v2_15seed
  ./run.sh e2e xai-bundle --run-id v2_15seed
  ./run.sh e2e report --run-id v2_15seed
  ./run.sh e2e dashboard --run-id v2_15seed
EOF
    ;;
  *)
    echo "Unknown command: $command"
    echo "Run './run.sh help' for available commands."
    exit 1
    ;;
esac
