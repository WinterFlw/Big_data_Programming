#!/usr/bin/env bash
# Create a small backup archive for essential v2 results.
#
# This is for RunPod maintenance / low-storage runs. It intentionally excludes
# checkpoints, Python environments, and cache folders so the archive can be
# downloaded or pushed off the Pod quickly.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)/.."

run_id="${RUN_ID:-v2_15seed}"
timestamp="$(date +%Y%m%d_%H%M%S)"
backup_dir="${BACKUP_DIR:-/workspace/hatespeech_backups}"
archive_name="hatespeech_${run_id}_essential_${timestamp}.tar.gz"
archive_path="${backup_dir}/${archive_name}"

mkdir -p "$backup_dir"

git_commit="unknown"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_commit="$(git rev-parse HEAD)"
fi

manifest_path="outputs/experiments/${run_id}/backup_manifest_${timestamp}.txt"
mkdir -p "$(dirname "$manifest_path")"
{
    echo "run_id=${run_id}"
    echo "created_at=${timestamp}"
    echo "git_commit=${git_commit}"
    echo "pwd=$(pwd)"
    echo ""
    echo "Disk usage:"
    df -h . || true
    echo ""
    echo "Output size:"
    du -sh "outputs/experiments/${run_id}" 2>/dev/null || true
} > "$manifest_path"

tar \
    --exclude=".venv" \
    --exclude=".cache" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude="*.pt" \
    --exclude="*.pth" \
    --exclude="*.bin" \
    --exclude="*.pkl" \
    --exclude="checkpoints" \
    --exclude="outputs/experiments/${run_id}/benchmark/checkpoints" \
    --exclude="outputs/experiments/${run_id}/xai/.cache" \
    -czf "$archive_path" \
    configs \
    scripts \
    pipeline \
    runtime \
    docs \
    "outputs/experiments/${run_id}" \
    README.md \
    run.sh

echo "Created backup archive:"
echo "$archive_path"
du -h "$archive_path"
echo ""
echo "IMPORTANT: this archive is still on the RunPod network volume."
echo "Download it through JupyterLab, scp it, or move it to external storage before maintenance."
