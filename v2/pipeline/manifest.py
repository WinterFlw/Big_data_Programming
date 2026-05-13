"""Manifest loading, validation, and writing for v2 experiments.

The manifest is the audit log for the run. It records the seed list, condition
matrix, XAI sampling plan, and output root. Every stage loads the manifest
rather than carrying hidden defaults in code.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import CONFIG_DIR, DEFAULT_CONFIG_PATH, DEFAULT_RUN_ID, DOC_MANIFEST_TEMPLATE, experiment_root
from .schema import CONDITION_METADATA


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict[str, Any], path: Path) -> Path:
    """Save JSON with stable formatting for reviewable diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def default_manifest_source() -> Path:
    """Return the preferred source manifest template."""
    # Runtime config wins over the documentation template. The docs copy is a
    # fallback so a fresh v2 workspace can still bootstrap itself.
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return DOC_MANIFEST_TEMPLATE


def load_manifest(path: Path | None = None, run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    """Load a manifest from an explicit path, run output, or default template."""
    # Priority order:
    #   1. explicit --manifest path
    #   2. already planned output manifest
    #   3. checked-in config/template
    #
    # This lets status/aggregate reuse the exact manifest that plan wrote.
    if path is not None:
        return normalize_manifest(load_json(path), run_id=run_id)

    planned_path = experiment_root(run_id) / "manifest.json"
    if planned_path.exists():
        return normalize_manifest(load_json(planned_path), run_id=run_id)

    return normalize_manifest(load_json(default_manifest_source()), run_id=run_id)


def normalize_manifest(manifest: dict[str, Any], run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    """Normalize run-specific fields without mutating the caller's dict."""
    # Deep-copy through JSON so callers can safely reuse their original dict.
    normalized = json.loads(json.dumps(manifest))
    # run_id and output_root are tied together. If a teammate passes
    # --run-id test_smoke, outputs must follow that run_id automatically.
    normalized["run_id"] = run_id
    normalized["output_root"] = f"outputs/experiments/{run_id}"
    normalized.setdefault("created_at", datetime.now(timezone.utc).date().isoformat())
    return normalized


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Return a short reproducibility hash for the manifest payload."""
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return validation errors. Empty list means the manifest is usable."""
    errors: list[str] = []
    # Missing top-level blocks usually mean someone passed the wrong JSON file.
    for key in ["run_id", "output_root", "benchmark", "statistics", "xai"]:
        if key not in manifest:
            errors.append(f"missing top-level key: {key}")

    benchmark = manifest.get("benchmark", {})
    seeds = benchmark.get("seeds", [])
    conditions = benchmark.get("conditions", [])
    if not seeds:
        errors.append("benchmark.seeds must not be empty")
    if not conditions:
        errors.append("benchmark.conditions must not be empty")

    unknown_conditions = [item for item in conditions if item not in CONDITION_METADATA]
    if unknown_conditions:
        errors.append(f"unknown benchmark conditions: {', '.join(unknown_conditions)}")

    # Duplicate seeds/conditions break paired-design assumptions and distort
    # status counts, so we reject them before any output tree is created.
    if len(set(seeds)) != len(seeds):
        errors.append("benchmark.seeds contains duplicates")
    if len(set(conditions)) != len(conditions):
        errors.append("benchmark.conditions contains duplicates")
    return errors


def write_planned_manifest(
    run_id: str = DEFAULT_RUN_ID,
    source_path: Path | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Create or refresh the canonical output manifest."""
    source = source_path or default_manifest_source()
    manifest = load_manifest(source, run_id=run_id)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Invalid manifest:\n- " + "\n- ".join(errors))

    root = experiment_root(run_id)
    target = root / "manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        # Do not silently overwrite a planned run. Reusing the output manifest
        # preserves the exact config that produced any existing artifacts.
        return load_manifest(target, run_id=run_id), target

    save_json(manifest, target)
    config_path = CONFIG_DIR / f"{run_id}.json"
    if source.resolve() != config_path.resolve():
        # Keep configs/<run_id>.json in sync when the user bootstraps from a
        # custom --manifest file. That gives teammates a stable config to read.
        config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, config_path)
    return manifest, target
