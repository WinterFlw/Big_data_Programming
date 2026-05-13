"""Artifact planning and status helpers for v2 pipeline runs.

This module answers the operational question "what exists on disk right now?"
It should not train models or compute statistics. Its job is to create folders,
write status ledgers, and decide whether a condition x seed unit is planned,
completed, or failed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .paths import display_path, experiment_root
from .schema import CONDITION_METADATA, RunUnit


# The output tree is created up front by the plan stage. That gives the team a
# visible map of where each stage will write before any expensive GPU job runs.
STAGE_DIRS = [
    "data",
    "benchmark/runs",
    "benchmark/checkpoints",
    "freeze/runs",
    "xai/samples",
    "xai/primary",
    "xai/deep/cases",
    "xai/ablation",
    "reports",
    "dashboard",
]


def ensure_experiment_tree(run_id: str) -> Path:
    """Create the full v2 output tree for a run_id."""
    root = experiment_root(run_id)
    for relative in STAGE_DIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def build_run_units(
    manifest: dict[str, Any],
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[RunUnit]:
    """Build condition x seed units from the manifest or CLI subsets.

    The full manifest produces 8 x 15 = 120 units. Smoke tests can pass a small
    subset, for example conditions=["A_B", "D_B"], seeds=[42].
    """
    benchmark = manifest["benchmark"]
    selected_conditions = conditions or list(benchmark["conditions"])
    selected_seeds = seeds or [int(seed) for seed in benchmark["seeds"]]
    root = experiment_root(manifest["run_id"])

    units: list[RunUnit] = []
    for condition in selected_conditions:
        # Unknown conditions are treated as a hard error. A typo like "DB" must
        # fail before we spend server time.
        if condition not in CONDITION_METADATA:
            raise ValueError(f"Unknown condition: {condition}")
        for seed in selected_seeds:
            units.append(RunUnit(manifest["run_id"], condition, int(seed), root))
    return units


def unit_status(unit: RunUnit) -> str:
    """Return planned/completed/failed for one run unit.

    A run is completed only when all minimal artifacts exist. This deliberately
    avoids treating a lone metrics.json as complete because interrupted jobs can
    leave partial files behind.
    """
    if unit.metrics_path.exists() and unit.history_path.exists() and unit.config_path.exists():
        return "completed"
    if (unit.run_dir / "stderr.log").exists():
        # The marker list is conservative. It catches common fatal failures
        # without trying to parse every possible framework-specific message.
        text = (unit.run_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")
        fatal_markers = ["RuntimeError", "Traceback", "CUDA out of memory", "nan loss"]
        if any(marker in text for marker in fatal_markers):
            return "failed"
    return "planned"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> Path:
    """Write a stable-column CSV file.

    DictWriter ignores any missing row keys by filling them with empty strings.
    That keeps placeholder outputs schema-compatible before real metrics exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def write_unit_plan(manifest: dict[str, Any], units: list[RunUnit]) -> Path:
    """Write execution_status.csv for the selected units.

    This file is the handoff surface for server operators. It tells them which
    units exist, where their folders are, and whether resume should skip them.
    """
    root = experiment_root(manifest["run_id"])
    columns = [
        "run_id",
        "condition",
        "seed",
        "backbone",
        "model_name",
        "use_attention_loss",
        "use_sentiment",
        "status",
        "run_dir",
    ]
    rows = []
    for unit in units:
        metadata = unit.metadata
        rows.append(
            {
                "run_id": unit.run_id,
                "condition": unit.condition,
                "seed": unit.seed,
                "backbone": metadata["backbone"],
                "model_name": metadata["model_name"],
                "use_attention_loss": metadata["use_attention_loss"],
                "use_sentiment": metadata["use_sentiment"],
                "status": unit_status(unit),
                "run_dir": display_path(unit.run_dir),
            }
        )
    return write_csv(root / "execution_status.csv", rows, columns)


def status_counts(units: list[RunUnit]) -> dict[str, int]:
    """Count planned/completed/failed units for CLI status output."""
    counts = {"planned": 0, "completed": 0, "failed": 0}
    for unit in units:
        counts[unit_status(unit)] += 1
    counts["total"] = len(units)
    return counts


def write_stage_marker(root: Path, stage: str, payload: dict[str, Any]) -> Path:
    """Write a small JSON marker that records a stage-level action."""
    marker = root / f"{stage}_status.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    return marker
