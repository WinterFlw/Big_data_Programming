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
echo "=== Full Run Gate 자동 점검 ==="
# sample 결정성 검사는 xai-primary 재실행을 동반하므로 daily.sh 안에서는 skip.
# Pilot이 별도로 `python3 scripts/gate_check.py` 단독 실행 시 검사.
if python3 scripts/gate_check.py --run-id v2_15seed --skip-sample-check; then
    gate_status="GO"
else
    gate_status="STOP"
fi

echo ""
echo "[daily preflight ok — Gate: $gate_status]"
