"""Stage orchestration for the v2 end-to-end pipeline.

Runner functions are the "control tower" for v2. The CLI calls this module, and
this module delegates to specialized helpers. Expensive model code should not
be placed directly here; instead, wire it through a small adapter so each stage
remains testable with dry-runs and placeholder artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import build_run_units, ensure_experiment_tree, status_counts, write_stage_marker, write_unit_plan
from .manifest import load_manifest, manifest_hash, validate_manifest, write_planned_manifest
from .paths import DEFAULT_RUN_ID, display_path, experiment_root
from .reporting import generate_dashboard, generate_markdown_report
from .schema import parse_csv_values, parse_seed_values
from .statistics import aggregate
from .xai import plan_ablation_xai, plan_deep_xai, plan_primary_xai


def plan(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None, force: bool = False) -> dict[str, Any]:
    """Create the canonical manifest, folder tree, and execution ledger."""
    manifest, path = write_planned_manifest(run_id=run_id, source_path=manifest_path, force=force)
    root = ensure_experiment_tree(run_id)
    # The execution ledger is generated immediately so the team can inspect the
    # full 120-unit plan before launching any server job.
    units = build_run_units(manifest)
    status_path = write_unit_plan(manifest, units)
    marker = write_stage_marker(
        root,
        "plan",
        {
            "run_id": run_id,
            "manifest": display_path(path),
            "manifest_hash": manifest_hash(manifest),
            "total_units": len(units),
        },
    )
    return {"manifest": path, "execution_status": status_path, "marker": marker, "total_units": len(units)}


def status(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None) -> dict[str, Any]:
    """Validate the manifest and refresh execution_status.csv."""
    manifest = load_manifest(manifest_path, run_id=run_id)
    errors = validate_manifest(manifest)
    if errors:
        return {"valid": False, "errors": errors}
    ensure_experiment_tree(run_id)
    units = build_run_units(manifest)
    status_path = write_unit_plan(manifest, units)
    counts = status_counts(units)
    return {"valid": True, "counts": counts, "execution_status": status_path}


def data(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None) -> dict[str, Any]:
    """Create a placeholder for the data verification stage.

    The actual implementation should later call the legacy data preparation
    code, record split hashes, and write split profiles under v2/outputs.
    """
    manifest = load_manifest(manifest_path, run_id=run_id)
    root = ensure_experiment_tree(run_id)
    marker = write_stage_marker(
        root / "data",
        "data",
        {
            "status": "planned",
            "split_source": manifest.get("data", {}).get("split_source"),
            "note": "Wire this stage to experiment_core.prepare_data and split hash profiling.",
        },
    )
    return {"marker": marker}


def benchmark(
    run_id: str = DEFAULT_RUN_ID,
    manifest_path: Path | None = None,
    conditions_value: str | None = None,
    seeds_value: str | None = None,
    dry_run: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Plan or execute benchmark units.

    Today this function is intentionally a safe scaffold. It can select units
    and write status ledgers, but it refuses --execute until the training
    adapter is implemented. This prevents someone from thinking the full
    120-run benchmark is already connected.
    """
    manifest = load_manifest(manifest_path, run_id=run_id)
    ensure_experiment_tree(run_id)
    default_conditions = list(manifest["benchmark"]["conditions"])
    default_seeds = [int(seed) for seed in manifest["benchmark"]["seeds"]]
    conditions = parse_csv_values(conditions_value, default_conditions)
    seeds = parse_seed_values(seeds_value, default_seeds)
    units = build_run_units(manifest, conditions=conditions, seeds=seeds)
    status_path = write_unit_plan(manifest, units)

    if execute:
        # Next implementation point:
        #   - import the legacy experiment_core training functions
        #   - map RunUnit metadata to the correct dataset/model factory
        #   - redirect checkpoints and run artifacts into unit.run_dir
        #   - honor --resume and --force without touching old outputs/
        raise NotImplementedError(
            "Benchmark execution is intentionally not wired yet. "
            "Next hook: call experiment_core.train_neural_model per RunUnit and write metrics.json."
        )

    return {
        "mode": "dry-run" if dry_run else "planned",
        "selected_units": len(units),
        "execution_status": status_path,
        "next_hook": "pipeline.runner.benchmark(..., execute=True)",
    }


def aggregate_stage(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None) -> dict[str, Any]:
    """Collect benchmark metrics and write summary/statistics CSVs."""
    manifest = load_manifest(manifest_path, run_id=run_id)
    ensure_experiment_tree(run_id)
    return aggregate(manifest)


def xai_primary(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Create or execute Primary XAI artifacts for A_B vs D_B across seeds."""
    manifest = load_manifest(manifest_path, run_id=run_id)
    ensure_experiment_tree(run_id)
    return plan_primary_xai(manifest, dry_run=dry_run)


def xai_deep(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Create or execute Deep XAI artifacts for median-seed case analysis."""
    manifest = load_manifest(manifest_path, run_id=run_id)
    ensure_experiment_tree(run_id)
    return plan_deep_xai(manifest, dry_run=dry_run)


def xai_ablation(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Create or execute lightweight XAI artifacts for all 8 conditions."""
    manifest = load_manifest(manifest_path, run_id=run_id)
    ensure_experiment_tree(run_id)
    return plan_ablation_xai(manifest, dry_run=dry_run)


def report(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None) -> dict[str, Any]:
    """Generate the Markdown report scaffold for this run_id."""
    manifest = load_manifest(manifest_path, run_id=run_id)
    ensure_experiment_tree(run_id)
    return {"report": generate_markdown_report(manifest)}


def dashboard(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None) -> dict[str, Any]:
    """Generate the HTML dashboard scaffold for this run_id."""
    manifest = load_manifest(manifest_path, run_id=run_id)
    ensure_experiment_tree(run_id)
    return {"dashboard": generate_dashboard(manifest)}


def all_stages(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None, force: bool = False) -> dict[str, Any]:
    """Run all non-expensive scaffolding stages.

    This is useful for local validation. It still does not train models; the
    benchmark call is a dry-run by design.
    """
    return {
        "plan": plan(run_id, manifest_path, force=force),
        "data": data(run_id, manifest_path),
        "benchmark": benchmark(run_id, manifest_path, dry_run=True),
        "aggregate": aggregate_stage(run_id, manifest_path),
        "xai_primary": xai_primary(run_id, manifest_path),
        "xai_deep": xai_deep(run_id, manifest_path),
        "xai_ablation": xai_ablation(run_id, manifest_path),
        "report": report(run_id, manifest_path),
        "dashboard": dashboard(run_id, manifest_path),
    }


def format_result(result: Any) -> str:
    """Format structured runner output for terminal display."""
    if isinstance(result, dict):
        lines = []
        for key, value in result.items():
            if isinstance(value, Path):
                lines.append(f"{key}: {display_path(value)}")
            elif isinstance(value, dict):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    return str(result)
