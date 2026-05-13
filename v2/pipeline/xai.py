"""XAI stage scaffolding for v2 experiments.

XAI is expensive and easy to over-interpret, so this module first creates
explicit sample/metric files for each XAI mode. Later, execution adapters can
fill those files with SHAP/LIME/rationale metrics without changing the output
contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import write_csv
from .paths import experiment_root
from .schema import XAI_SEED_METRIC_COLUMNS


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def plan_primary_xai(manifest: dict[str, Any], dry_run: bool = False) -> dict[str, Path | str]:
    """Prepare Primary XAI outputs.

    Primary XAI is the strict comparison: A_B vs D_B, all 15 seeds, same fixed
    sample set. The later implementation must not resample independently per
    seed because that would destroy explanation-stability comparisons.
    """
    root = experiment_root(manifest["run_id"])
    xai_config = manifest["xai"]["primary"]
    if dry_run:
        return {"status": "dry-run", "sample_size": str(xai_config["sample_size"])}

    # The sample CSV starts empty today. The future sampler should fill it with
    # stable sample_id values and keep this exact file reused across all seeds.
    sample_path = write_csv(
        root / "xai" / "samples" / "primary_samples.csv",
        [],
        ["sample_id", "label", "case_type", "source", "target"],
    )
    metric_path = write_csv(root / "xai" / "primary" / "seed_level_metrics.csv", [], XAI_SEED_METRIC_COLUMNS)
    paired_path = write_csv(
        root / "xai" / "primary" / "paired_xai_tests.csv",
        [],
        ["comparison", "metric", "n_pairs", "mean_diff", "p_value", "p_value_holm", "effect_size"],
    )
    stability_path = write_csv(
        root / "xai" / "primary" / "seed_stability.csv",
        [],
        ["condition", "metric", "mean", "std", "ci_low", "ci_high"],
    )
    return {
        "samples": sample_path,
        "seed_metrics": metric_path,
        "paired_tests": paired_path,
        "seed_stability": stability_path,
    }


def plan_deep_xai(manifest: dict[str, Any], dry_run: bool = False) -> dict[str, Path | str]:
    """Prepare Deep XAI outputs for qualitative case analysis."""
    root = experiment_root(manifest["run_id"])
    xai_config = manifest["xai"]["deep"]
    if dry_run:
        return {"status": "dry-run", "sample_size": str(xai_config["sample_size"])}

    sample_path = write_csv(
        root / "xai" / "samples" / "deep_samples.csv",
        [],
        ["sample_id", "label", "case_type", "source", "target"],
    )
    # xai_details.json is where detailed attribution payloads can later live.
    # It starts as a planned marker so report/dashboard can link to it now.
    details_path = _write_json(
        root / "xai" / "deep" / "xai_details.json",
        {"status": "planned", "models": xai_config["models"], "sample_size": xai_config["sample_size"]},
    )
    case_path = write_csv(
        root / "xai" / "deep" / "case_summary.csv",
        [],
        [
            "sample_id",
            "true_label",
            "baseline_prediction",
            "v2_prediction",
            "case_type",
            "baseline_confidence",
            "v2_confidence",
            "top_tokens_baseline",
            "top_tokens_v2",
            "human_rationale_tokens",
            "plot_path",
            "comment",
        ],
    )
    return {"samples": sample_path, "details": details_path, "cases": case_path}


def plan_ablation_xai(manifest: dict[str, Any], dry_run: bool = False) -> dict[str, Path | str]:
    """Prepare lightweight XAI outputs for all 8 ablation conditions."""
    root = experiment_root(manifest["run_id"])
    xai_config = manifest["xai"]["ablation"]
    if dry_run:
        return {"status": "dry-run", "sample_size": str(xai_config["sample_size"])}

    sample_path = write_csv(
        root / "xai" / "samples" / "ablation_samples.csv",
        [],
        ["sample_id", "label", "case_type", "source", "target"],
    )
    # Ablation XAI is intentionally smaller than Primary/Deep XAI. Its job is
    # to show directionality across all conditions, not to replace the primary
    # A_B vs D_B seed-stability analysis.
    metric_path = write_csv(
        root / "xai" / "ablation" / "xai_ablation_metrics.csv",
        [],
        [
            "condition",
            "backbone",
            "attention_loss",
            "sentiment_feature",
            "rationale_f1_at_5",
            "comprehensiveness",
            "sufficiency",
            "attention_entropy",
            "mss",
        ],
    )
    summary_path = _write_json(
        root / "xai" / "xai_summary.json",
        {"status": "planned", "primary": manifest["xai"]["primary"], "deep": manifest["xai"]["deep"], "ablation": xai_config},
    )
    return {"samples": sample_path, "metrics": metric_path, "summary": summary_path}
