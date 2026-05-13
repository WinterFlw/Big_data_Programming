"""Benchmark aggregation and paired statistics for v2 experiments.

This module is intentionally split into small steps because the statistics
stage will be one of the most reviewed parts of the project. It accepts partial
benchmark output during smoke tests, then fills the same CSV contracts with
confidence intervals, paired tests, effect sizes, and Holm correction once
completed run metrics exist.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .artifacts import build_run_units, unit_status, write_csv
from .paths import experiment_root
from .schema import BENCHMARK_RUN_COLUMNS, BENCHMARK_SUMMARY_COLUMNS, PAIRED_TEST_COLUMNS

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - exercised only on minimal installs.
    scipy_stats = None


def _read_metrics(path: Path) -> dict[str, Any]:
    """Read a run metrics file, returning an empty dict for missing runs."""
    # Missing metrics are normal before the benchmark has run. Aggregate should
    # still produce schema-correct CSVs so downstream report/dashboard work can
    # proceed with placeholders.
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_benchmark_runs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect one row per planned condition x seed unit."""
    rows: list[dict[str, Any]] = []
    for unit in build_run_units(manifest):
        metadata = unit.metadata
        metrics = _read_metrics(unit.metrics_path)
        rows.append(
            {
                "run_id": unit.run_id,
                "condition": unit.condition,
                "backbone": metadata["backbone"],
                "use_attention_loss": metadata["use_attention_loss"],
                "use_sentiment": metadata["use_sentiment"],
                "seed": unit.seed,
                "status": unit_status(unit),
                "train_seconds": metrics.get("train_seconds", ""),
                "best_epoch": metrics.get("best_epoch", ""),
                "macro_f1": metrics.get("macro_f1", ""),
                "weighted_f1": metrics.get("weighted_f1", ""),
                "accuracy": metrics.get("accuracy", ""),
                "precision_macro": metrics.get("precision_macro", ""),
                "recall_macro": metrics.get("recall_macro", ""),
                "loss": metrics.get("loss", ""),
                "checkpoint_path": metrics.get("checkpoint_path", ""),
                # Store the expected metrics path even for planned runs. That
                # makes failed-run diagnosis easier because the table points to
                # exactly where the missing file should have appeared.
                "metrics_path": unit.metrics_path,
                "predictions_path": metrics.get("predictions_path", ""),
            }
        )
    return rows


def summarize_benchmark(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize completed runs by condition.

    Planned and failed rows are preserved in benchmark_runs.csv but excluded
    from metric means. failed_seed_count is kept so the report can distinguish
    "no result yet" from "some seeds crashed".
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for condition, condition_rows in grouped.items():
        completed = [row for row in condition_rows if row["status"] == "completed" and row["macro_f1"] != ""]
        values = [float(row["macro_f1"]) for row in completed]
        accuracy_values = [float(row["accuracy"]) for row in completed if row["accuracy"] != ""]
        weighted_values = [float(row["weighted_f1"]) for row in completed if row["weighted_f1"] != ""]
        backbone = condition_rows[0]["backbone"] if condition_rows else ""
        best_seed = ""
        median_seed = ""
        if completed:
            # best_seed is useful for diagnosis, but final claims must use the
            # mean and confidence interval. median_seed is used by XAI to avoid
            # cherry-picking a best checkpoint for qualitative cases.
            sorted_completed = sorted(completed, key=lambda row: float(row["macro_f1"]))
            best_seed = sorted_completed[-1]["seed"]
            median_seed = sorted_completed[len(sorted_completed) // 2]["seed"]
        summaries.append(
            {
                "condition": condition,
                "backbone": backbone,
                "n_seeds": len(completed),
                "macro_f1_mean": mean(values) if values else "",
                "macro_f1_std": stdev(values) if len(values) > 1 else "",
                "macro_f1_ci_low": _mean_ci(values)[0] if values else "",
                "macro_f1_ci_high": _mean_ci(values)[1] if values else "",
                "weighted_f1_mean": mean(weighted_values) if weighted_values else "",
                "accuracy_mean": mean(accuracy_values) if accuracy_values else "",
                "best_seed": best_seed,
                "median_seed": median_seed,
                "failed_seed_count": sum(1 for row in condition_rows if row["status"] == "failed"),
            }
        )
    return summaries


def _mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided confidence interval for a mean."""
    if not values:
        return ("", "")  # type: ignore[return-value]
    if len(values) == 1:
        return (values[0], values[0])
    standard_error = stdev(values) / math.sqrt(len(values))
    if scipy_stats is not None:
        critical_value = float(scipy_stats.t.ppf((1 + confidence) / 2, df=len(values) - 1))
    else:
        critical_value = 1.96
    center = mean(values)
    margin = critical_value * standard_error
    return (center - margin, center + margin)


def _paired_p_value(differences: list[float]) -> tuple[str, str]:
    """Return the paired-test name and p-value for a difference vector."""
    if len(differences) < 2:
        return ("paired_t", "")
    if all(abs(value) < 1e-12 for value in differences):
        return ("paired_t", 1.0)
    if scipy_stats is None:
        # Normal approximation fallback. The project requirements include
        # scipy, but this keeps local syntax checks useful on thin installs.
        std_diff = stdev(differences)
        if std_diff == 0:
            return ("paired_t_normal_approx", 1.0)
        z_score = mean(differences) / (std_diff / math.sqrt(len(differences)))
        p_value = math.erfc(abs(z_score) / math.sqrt(2))
        return ("paired_t_normal_approx", p_value)
    result = scipy_stats.ttest_1samp(differences, popmean=0.0, nan_policy="omit")
    return ("paired_t", float(result.pvalue))


def _cohen_dz(differences: list[float]) -> str | float:
    """Compute paired Cohen's dz."""
    if len(differences) < 2:
        return ""
    std_diff = stdev(differences)
    if std_diff == 0:
        return 0.0
    return mean(differences) / std_diff


def _build_metric_lookup(rows: list[dict[str, Any]], metric: str) -> dict[tuple[str, int], float]:
    """Map (condition, seed) to one metric value for completed runs."""
    lookup: dict[tuple[str, int], float] = {}
    for row in rows:
        if row["status"] != "completed" or row.get(metric, "") == "":
            continue
        lookup[(str(row["condition"]), int(row["seed"]))] = float(row[metric])
    return lookup


def compute_paired_tests(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute same-seed paired comparisons requested by the manifest."""
    comparisons = manifest.get("statistics", {}).get("paired_tests", [])
    metrics = ["macro_f1", "accuracy", "weighted_f1"]
    test_rows: list[dict[str, Any]] = []
    for metric in metrics:
        lookup = _build_metric_lookup(rows, metric)
        for comparison in comparisons:
            condition_a, condition_b = comparison.split(":")
            seeds = sorted(
                {
                    seed
                    for condition, seed in lookup
                    if condition == condition_a and (condition_b, seed) in lookup
                }
            )
            differences = [lookup[(condition_b, seed)] - lookup[(condition_a, seed)] for seed in seeds]
            ci_low, ci_high = _mean_ci(differences) if differences else ("", "")
            test_name, p_value = _paired_p_value(differences)
            test_rows.append(
                {
                    "comparison": f"{condition_a} vs {condition_b}",
                    "metric": metric,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "n_pairs": len(differences),
                    "mean_diff": mean(differences) if differences else "",
                    "std_diff": stdev(differences) if len(differences) > 1 else "",
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "test_name": test_name,
                    "p_value": p_value,
                    "p_value_holm": "",
                    "effect_size": _cohen_dz(differences),
                    "significant_0_05": "",
                }
            )
    return test_rows


def apply_holm_correction(rows: list[dict[str, Any]], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni correction to rows with p-values."""
    corrected = [dict(row) for row in rows]
    indexed_p_values = [
        (index, float(row["p_value"]))
        for index, row in enumerate(corrected)
        if row.get("p_value") not in {"", None}
    ]
    total = len(indexed_p_values)
    if total == 0:
        return corrected

    ordered = sorted(indexed_p_values, key=lambda item: item[1])
    adjusted_by_index: dict[int, float] = {}
    running_max = 0.0
    for rank, (index, p_value) in enumerate(ordered, start=1):
        adjusted = min(1.0, (total - rank + 1) * p_value)
        running_max = max(running_max, adjusted)
        adjusted_by_index[index] = running_max

    for index, adjusted in adjusted_by_index.items():
        corrected[index]["p_value_holm"] = adjusted
        corrected[index]["significant_0_05"] = adjusted < alpha
    return corrected


def aggregate(manifest: dict[str, Any]) -> dict[str, Path]:
    """Run the benchmark aggregation stage."""
    root = experiment_root(manifest["run_id"])
    benchmark_dir = root / "benchmark"
    rows = collect_benchmark_runs(manifest)
    summary = summarize_benchmark(rows)
    paired_tests = compute_paired_tests(manifest, rows)
    holm_tests = apply_holm_correction(paired_tests)

    run_path = write_csv(benchmark_dir / "benchmark_runs.csv", rows, BENCHMARK_RUN_COLUMNS)
    summary_path = write_csv(benchmark_dir / "benchmark_summary.csv", summary, BENCHMARK_SUMMARY_COLUMNS)
    paired_path = write_csv(benchmark_dir / "paired_tests.csv", paired_tests, PAIRED_TEST_COLUMNS)
    holm_path = write_csv(benchmark_dir / "paired_tests_holm.csv", holm_tests, PAIRED_TEST_COLUMNS)
    return {
        "benchmark_runs": run_path,
        "benchmark_summary": summary_path,
        "paired_tests": paired_path,
        "paired_tests_holm": holm_path,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Small CSV reader for report/dashboard code."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
