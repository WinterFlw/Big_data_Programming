#!/usr/bin/env python3
"""Prune v2 benchmark checkpoints to reduce RunPod network storage pressure."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_RE = re.compile(r"^(?P<condition>.+)_seed_(?P<seed>\d+)\.pt$")


def _load_manifest(run_id: str) -> dict[str, Any]:
    run_manifest = BASE_DIR / "outputs" / "experiments" / run_id / "manifest.json"
    if run_manifest.exists():
        return json.loads(run_manifest.read_text(encoding="utf-8"))
    config_manifest = BASE_DIR / "configs" / f"{run_id}.json"
    if config_manifest.exists():
        return json.loads(config_manifest.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"manifest not found for run_id={run_id}")


def _needed_pairs(manifest: dict[str, Any], policy: str) -> set[tuple[str, int]]:
    if policy == "none":
        return set()
    seeds = [int(seed) for seed in manifest["benchmark"]["seeds"]]
    middle_seed = seeds[len(seeds) // 2]
    xai_config = manifest.get("xai", {})
    needed: set[tuple[str, int]] = set()

    for condition in xai_config.get("primary", {}).get("models", []):
        for seed in seeds:
            needed.add((str(condition).upper(), seed))

    for group_name in ("deep", "ablation"):
        for condition in xai_config.get(group_name, {}).get("models", []):
            needed.add((str(condition).upper(), middle_seed))
    return needed


def _iter_checkpoint_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*.pt"))


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


def prune(run_id: str, policy: str, dry_run: bool = False) -> dict[str, Any]:
    manifest = _load_manifest(run_id)
    if policy not in {"xai-minimal", "none"}:
        raise ValueError("policy must be xai-minimal or none")

    run_root = BASE_DIR / "outputs" / "experiments" / run_id
    benchmark_checkpoint_dir = run_root / "benchmark" / "checkpoints"
    runtime_checkpoint_dir = BASE_DIR / "checkpoints"
    needed = _needed_pairs(manifest, policy)

    deleted_files: list[Path] = []
    kept_files: list[Path] = []
    reclaimed_bytes = 0

    for path in _iter_checkpoint_files(benchmark_checkpoint_dir):
        match = CHECKPOINT_RE.match(path.name)
        if not match:
            kept_files.append(path)
            continue
        key = (match.group("condition").upper(), int(match.group("seed")))
        if key in needed:
            kept_files.append(path)
            continue
        size = path.stat().st_size
        deleted_files.append(path)
        reclaimed_bytes += size
        if not dry_run:
            path.unlink()

    # v2/runtime writes the temporary best checkpoint to BASE_DIR/checkpoints.
    # training_adapter now moves it into benchmark/checkpoints, but old runs may
    # still have duplicate files here. They are not part of the v2 XAI contract.
    for path in _iter_checkpoint_files(runtime_checkpoint_dir):
        size = path.stat().st_size
        deleted_files.append(path)
        reclaimed_bytes += size
        if not dry_run:
            path.unlink()

    return {
        "policy": policy,
        "dry_run": dry_run,
        "kept": len(kept_files),
        "deleted": len(deleted_files),
        "reclaimed_bytes": reclaimed_bytes,
        "reclaimed": _format_bytes(reclaimed_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune v2 benchmark checkpoints.")
    parser.add_argument("--run-id", default="v2_15seed")
    parser.add_argument("--policy", choices=["xai-minimal", "none"], default="xai-minimal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = prune(args.run_id, args.policy, dry_run=args.dry_run)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
