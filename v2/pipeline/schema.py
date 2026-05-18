"""Data structures and fixed schemas for v2 orchestration.

The rest of the pipeline should treat this module as the contract layer. If a
CSV gains or loses a column, update the column list here first, then update the
stage that writes it. That makes schema drift easier to review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


# The 8-condition matrix is the center of the v2 experiment. The condition name
# encodes the ablation:
#
#   A = no attention loss, no sentiment
#   B = attention loss only
#   C = sentiment only
#   D = attention loss + sentiment
#   _B = BERT backbone
#   _R = RoBERTa backbone
#
# Keeping this metadata in one dict prevents different stages from silently
# disagreeing about what "D_B" means.
CONDITION_METADATA: dict[str, dict[str, Any]] = {
    "A_B": {"backbone": "BERT", "model_name": "bert-base-uncased", "use_attention_loss": False, "use_sentiment": False},
    "B_B": {"backbone": "BERT", "model_name": "bert-base-uncased", "use_attention_loss": True, "use_sentiment": False},
    "C_B": {"backbone": "BERT", "model_name": "bert-base-uncased", "use_attention_loss": False, "use_sentiment": True},
    "D_B": {"backbone": "BERT", "model_name": "bert-base-uncased", "use_attention_loss": True, "use_sentiment": True},
    "A_R": {"backbone": "RoBERTa", "model_name": "roberta-base", "use_attention_loss": False, "use_sentiment": False},
    "B_R": {"backbone": "RoBERTa", "model_name": "roberta-base", "use_attention_loss": True, "use_sentiment": False},
    "C_R": {"backbone": "RoBERTa", "model_name": "roberta-base", "use_attention_loss": False, "use_sentiment": True},
    "D_R": {"backbone": "RoBERTa", "model_name": "roberta-base", "use_attention_loss": True, "use_sentiment": True},
}


# One row per condition x seed run. This is the raw ledger that later stages
# use for summaries, paired tests, dashboards, and failure recovery.
BENCHMARK_RUN_COLUMNS = [
    "run_id",
    "condition",
    "backbone",
    "use_attention_loss",
    "use_sentiment",
    "seed",
    "status",
    "train_seconds",
    "best_epoch",
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "loss",
    "checkpoint_path",
    "metrics_path",
    "predictions_path",
]


# One row per condition after aggregating all completed seeds. The report must
# cite these mean/CI fields rather than cherry-picking best_seed.
BENCHMARK_SUMMARY_COLUMNS = [
    "condition",
    "backbone",
    "n_seeds",
    "macro_f1_mean",
    "macro_f1_std",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "weighted_f1_mean",
    "accuracy_mean",
    "best_seed",
    "median_seed",
    "failed_seed_count",
]


# Paired comparisons use the same seed on both sides. This is the statistical
# reason for running the exact same seed list across all 8 conditions.
PAIRED_TEST_COLUMNS = [
    "comparison",
    "metric",
    "condition_a",
    "condition_b",
    "n_pairs",
    "mean_diff",
    "std_diff",
    "ci_low",
    "ci_high",
    "test_name",
    "p_value",
    "p_value_holm",
    "effect_size",
    "significant_0_05",
]


# One row per condition x seed for Primary XAI. These columns intentionally
# mirror the XAI protocol document so the report layer can stay mechanical.
# v2.1 자동 XAI 4축 — Context Learning 축의 CI / MSS / IS / Attention Rollout Entropy
# 도 같이 둔다 (ablation 카드와 동일 메트릭이지만 primary는 sample 200 기준).
XAI_SEED_METRIC_COLUMNS = [
    "run_id",
    "condition",
    "seed",
    "sample_count",
    "shap_lime_overlap_at_5",
    "shap_lime_overlap_at_10",
    "rationale_precision_at_5",
    "rationale_recall_at_5",
    "rationale_f1_at_5",
    "comprehensiveness",
    "sufficiency",
    "loo_drop",
    "topk_jaccard_mean",
    "rank_corr_mean",
    "ci",
    "mss",
    "interaction_strength",
    "attention_entropy",
]


@dataclass(frozen=True)
class RunUnit:
    """A single condition x seed benchmark job.

    This object is the safest way to pass a benchmark job between modules.
    Instead of repeatedly recomputing paths from strings, every path derived
    from a run lives behind a property here.
    """

    run_id: str
    condition: str
    seed: int
    output_root: Path

    @property
    def metadata(self) -> dict[str, Any]:
        # Metadata lookup fails fast if a caller invents a condition name that
        # is not in CONDITION_METADATA. That is better than creating files for
        # a misspelled condition and discovering it after a long server run.
        return CONDITION_METADATA[self.condition]

    @property
    def run_dir(self) -> Path:
        # Example:
        #   v2/outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/
        #
        # The condition is lowercased only for folder readability. The canonical
        # condition value in CSV/JSON remains the uppercase form.
        return self.output_root / "benchmark" / "runs" / self.condition.lower() / f"seed_{self.seed}"

    @property
    def metrics_path(self) -> Path:
        # metrics.json is the minimal completion signal for aggregation.
        return self.run_dir / "metrics.json"

    @property
    def history_path(self) -> Path:
        # history.csv proves training progressed epoch by epoch, not just that
        # a final metric was written by hand or by a partial run.
        return self.run_dir / "history.csv"

    @property
    def config_path(self) -> Path:
        # run_config.json captures the exact hyperparameters used for this unit.
        # It is part of the completion criteria so resume can be trusted.
        return self.run_dir / "run_config.json"


def parse_csv_values(value: str | None, default: list[str]) -> list[str]:
    """Parse a comma-separated CLI option, preserving the default when omitted."""
    # Empty option means "use the manifest default", not "run nothing".
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_values(value: str | None, default: list[int]) -> list[int]:
    """Parse comma-separated integer seeds."""
    # Keep parsing strict: if a seed is not an integer, argparse should fail via
    # the ValueError instead of silently dropping it.
    if not value:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]
