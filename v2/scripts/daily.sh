#!/usr/bin/env bash
# Daily preflight for v2_15seed pipeline (QA stage owner's daily tool).
# Run from repo root: ./v2/scripts/daily.sh

set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)/.."

echo "=== compile pipeline ==="
python3 -m compileall pipeline scripts/validate_commit_message.py

echo ""
echo "=== config validation ==="
python3 -m json.tool configs/v2_15seed.json > /tmp/v2_config_check.json && echo "config ok"

echo ""
echo "=== CLI help ==="
./run.sh e2e --help > /dev/null && echo "cli help ok"

echo ""
echo "=== plan / status ==="
./run.sh e2e plan --run-id v2_15seed --force | tail -5
./run.sh e2e status --run-id v2_15seed | tail -5

echo ""
echo "=== dry-run benchmark ==="
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run | tail -5

echo ""
echo "=== scaffolding stages (placeholder safe) ==="
./run.sh e2e aggregate --run-id v2_15seed | tail -5
./run.sh e2e xai-bundle --run-id v2_15seed | tail -3
./run.sh e2e report --run-id v2_15seed | tail -3
./run.sh e2e dashboard --run-id v2_15seed | tail -3

echo ""
echo "=== failed runs ==="
if [ -f outputs/experiments/v2_15seed/failed_runs.csv ]; then
    failed_count=$(tail -n +2 outputs/experiments/v2_15seed/failed_runs.csv | wc -l)
    echo "failed_runs.csv: $failed_count rows"
else
    echo "failed_runs.csv: not yet generated"
fi

echo ""
echo "=== completed runs ==="
if [ -f outputs/experiments/v2_15seed/completed_runs.csv ]; then
    completed_count=$(tail -n +2 outputs/experiments/v2_15seed/completed_runs.csv | wc -l)
    echo "completed_runs.csv: $completed_count rows"
else
    echo "completed_runs.csv: not yet generated"
fi

echo ""
echo "[daily preflight ok]"
