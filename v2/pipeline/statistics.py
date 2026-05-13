"""Benchmark aggregation scaffold for v2 experiments.

This module is intentionally split into small steps because the statistics
stage will be one of the most reviewed parts of the project. The current code
already writes the expected CSV shells; the next implementation should fill in
confidence intervals, paired tests, effect sizes, and Holm correction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .artifacts import build_run_units, unit_status, write_csv
from .paths import experiment_root
from .schema import BENCHMARK_RUN_COLUMNS, BENCHMARK_SUMMARY_COLUMNS, PAIRED_TEST_COLUMNS


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
                "macro_f1_ci_low": "",
                "macro_f1_ci_high": "",
                "weighted_f1_mean": mean(weighted_values) if weighted_values else "",
                "accuracy_mean": mean(accuracy_values) if accuracy_values else "",
                "best_seed": best_seed,
                "median_seed": median_seed,
                "failed_seed_count": sum(1 for row in condition_rows if row["status"] == "failed"),
            }
        )
    return summaries


def write_empty_paired_tests(path: Path) -> Path:
    """Write a schema-correct paired-test CSV before real tests are implemented."""
    return write_csv(path, [], PAIRED_TEST_COLUMNS)


def aggregate(manifest: dict[str, Any]) -> dict[str, Path]:
    """Run the benchmark aggregation stage."""
    root = experiment_root(manifest["run_id"])
    benchmark_dir = root / "benchmark"
    rows = collect_benchmark_runs(manifest)
    summary = summarize_benchmark(rows)

    run_path = write_csv(benchmark_dir / "benchmark_runs.csv", rows, BENCHMARK_RUN_COLUMNS)
    summary_path = write_csv(benchmark_dir / "benchmark_summary.csv", summary, BENCHMARK_SUMMARY_COLUMNS)
    paired_path = write_empty_paired_tests(benchmark_dir / "paired_tests.csv")
    holm_path = write_empty_paired_tests(benchmark_dir / "paired_tests_holm.csv")
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
