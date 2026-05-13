"""Evidence bundle builder for v2 XAI outputs.

The XAI execution code and the evidence-bundle code are intentionally split.
Primary/deep/ablation XAI stages may become expensive model code, while this
module should stay a deterministic artifact combiner that report/dashboard can
depend on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import write_csv
from .paths import CONFIG_DIR, display_path, experiment_root


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def build_xai_evidence_bundle(manifest: dict[str, Any], dry_run: bool = False) -> dict[str, Path | str]:
    """Create the canonical XAI evidence bundle surface.

    This stage does not recompute SHAP, LIME, or attribution metrics. It reads
    primary/deep/ablation artifacts and exposes the small contract that
    report/dashboard should prefer. Until the expensive XAI adapters are
    implemented, it writes planned placeholders with the final filenames so
    downstream work can be built and tested now.
    """
    root = experiment_root(manifest["run_id"])
    bundle_dir = root / "xai" / "evidence_bundle"
    if dry_run:
        return {"status": "dry-run", "bundle_dir": bundle_dir}

    source_artifacts = {
        "primary_seed_metrics": "xai/primary/seed_level_metrics.csv",
        "primary_paired_tests": "xai/primary/paired_xai_tests.csv",
        "primary_seed_stability": "xai/primary/seed_stability.csv",
        "deep_case_summary": "xai/deep/case_summary.csv",
        "deep_details": "xai/deep/xai_details.json",
        "ablation_metrics": "xai/ablation/xai_ablation_metrics.csv",
        "xai_summary": "xai/xai_summary.json",
    }
    inventory_rows = [
        {
            "artifact": name,
            "path": relative_path,
            "exists": str((root / relative_path).exists()).lower(),
        }
        for name, relative_path in source_artifacts.items()
    ]

    inventory_path = write_csv(
        bundle_dir / "evidence_inventory.csv",
        inventory_rows,
        ["artifact", "path", "exists"],
    )
    metadata_path = _write_json(
        bundle_dir / "xai_run_metadata.json",
        {
            "status": "planned",
            "run_id": manifest["run_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "commit_hash": "pending",
            "config_path": display_path(CONFIG_DIR / f"{manifest['run_id']}.json"),
            "manifest_path": display_path(root / "manifest.json"),
            "data_split_hash": "pending",
            "conditions": manifest["benchmark"]["conditions"],
            "seeds": manifest["benchmark"]["seeds"],
            "source_artifacts": source_artifacts,
        },
    )
    sample_manifest_path = write_csv(
        bundle_dir / "xai_sample_manifest.csv",
        [],
        [
            "sample_id",
            "label",
            "split",
            "case_type",
            "selected_for_primary",
            "selected_for_deep",
            "selected_for_ablation",
            "rationale_available",
        ],
    )
    predictions_path = write_csv(
        bundle_dir / "xai_predictions.csv",
        [],
        ["sample_id", "condition", "seed", "true_label", "predicted_label", "probability", "checkpoint_path"],
    )
    method_agreement_path = write_csv(
        bundle_dir / "method_agreement.csv",
        [],
        ["sample_id", "condition", "seed", "overlap_at_5", "overlap_at_10", "rank_corr", "notes"],
    )
    faithfulness_path = write_csv(
        bundle_dir / "faithfulness_metrics.csv",
        [],
        ["sample_id", "condition", "seed", "comprehensiveness", "sufficiency", "loo_drop"],
    )
    context_path = write_csv(
        bundle_dir / "context_metrics.csv",
        [],
        ["sample_id", "condition", "target", "source", "context_window", "context_sensitivity"],
    )
    plausibility_path = write_csv(
        bundle_dir / "plausibility_metrics.csv",
        [],
        ["sample_id", "condition", "seed", "rationale_precision_at_5", "rationale_recall_at_5", "rationale_f1_at_5"],
    )
    subgroup_path = write_csv(
        bundle_dir / "subgroup_xai_metrics.csv",
        [],
        ["subgroup", "condition", "seed", "metric", "value"],
    )
    risk_flags_path = write_csv(
        bundle_dir / "xai_risk_flags.csv",
        [],
        ["sample_id", "condition", "seed", "flag_type", "severity", "evidence", "recommended_report_note"],
    )
    claims_path = _write_json(
        bundle_dir / "xai_claims.json",
        {
            "status": "planned",
            "run_id": manifest["run_id"],
            "purpose": "Report-ready XAI claims derived from primary/deep/ablation artifacts.",
            "claims": [],
            "source_artifacts": source_artifacts,
            "required_before_claiming": [
                "Fill primary seed-level XAI metrics.",
                "Fill paired XAI tests and seed stability.",
                "Fill deep qualitative case summaries.",
                "Fill ablation-level XAI metrics.",
            ],
        },
    )
    interpretation_cards_path = _write_json(
        bundle_dir / "xai_interpretation_cards.json",
        {
            "status": "planned",
            "run_id": manifest["run_id"],
            "cards": [],
        },
    )
    dashboard_bundle_path = _write_json(
        bundle_dir / "xai_dashboard_bundle.json",
        {
            "status": "planned",
            "run_id": manifest["run_id"],
            "summary_cards": [],
            "primary": {},
            "seed_stability": {},
            "deep_cases": [],
            "ablation": {},
            "artifact_links": source_artifacts,
        },
    )
    token_attributions_path = bundle_dir / "token_attributions.jsonl"
    token_attributions_path.write_text("", encoding="utf-8")
    readme_path = bundle_dir / "README.md"
    readme_path.write_text(
        f"""# XAI Evidence Bundle

run_id: `{manifest["run_id"]}`

This directory is the canonical XAI bundle for report/dashboard stages.
The placeholder files are created early so downstream code can be developed
before expensive SHAP/LIME/faithfulness runs are complete.

Report and dashboard code should prefer:

- `xai_claims.json`
- `xai_dashboard_bundle.json`
- `xai_interpretation_cards.json`

Raw per-sample evidence remains available through the CSV/JSONL files in this
directory.
""",
        encoding="utf-8",
    )

    return {
        "inventory": inventory_path,
        "metadata": metadata_path,
        "sample_manifest": sample_manifest_path,
        "predictions": predictions_path,
        "method_agreement": method_agreement_path,
        "faithfulness": faithfulness_path,
        "context": context_path,
        "plausibility": plausibility_path,
        "subgroup": subgroup_path,
        "risk_flags": risk_flags_path,
        "claims": claims_path,
        "interpretation_cards": interpretation_cards_path,
        "dashboard_bundle": dashboard_bundle_path,
        "token_attributions": token_attributions_path,
        "readme": readme_path,
    }
