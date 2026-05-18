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

try:
    import numpy as _np
    _NP_AVAILABLE = True
except ImportError:  # pragma: no cover.
    _np = None  # type: ignore[assignment]
    _NP_AVAILABLE = False

# Bootstrap CI 기본값. manifest.statistics.bootstrap_iterations 가 있으면 override.
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_BOOTSTRAP_SEED = 0

try:
    import pandas as _pd
    from statsmodels.formula.api import ols as _ols
    from statsmodels.stats.anova import anova_lm as _anova_lm
    _ANOVA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on minimal installs.
    _pd = None  # type: ignore[assignment]
    _ols = None  # type: ignore[assignment]
    _anova_lm = None  # type: ignore[assignment]
    _ANOVA_AVAILABLE = False


# ANOVA 출력 컬럼. statsmodels의 anova_lm은 'df' (degrees of freedom), 'sum_sq',
# 'F', 'PR(>F)' 컬럼을 갖는 DataFrame을 돌려준다. 우리 CSV로 흘려보낼 때는
# 'df' 컬럼명이 pandas DataFrame 변수명과 헷갈리지 않도록 그대로 두되,
# 'PR(>F)'는 'p_value'로 rename 한다.
# eta_squared / partial_eta_squared 는 효과 크기 — F/p만으로는 "얼마나 큰 효과인가"를
# 판단할 수 없으므로 ANOVA 보고에 함께 박는다 (Cohen 1988).
ANOVA_2WAY_COLUMNS = [
    "family", "metric", "factor", "sum_sq", "df", "F", "p_value",
    "eta_squared", "partial_eta_squared",
]
ANOVA_3WAY_COLUMNS = [
    "metric", "factor", "sum_sq", "df", "F", "p_value",
    "eta_squared", "partial_eta_squared",
]


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


def summarize_benchmark(
    rows: list[dict[str, Any]],
    bootstrap_iterations: int | None = None,
) -> list[dict[str, Any]]:
    """Summarize completed runs by condition.

    Planned and failed rows are preserved in benchmark_runs.csv but excluded
    from metric means. failed_seed_count is kept so the report can distinguish
    "no result yet" from "some seeds crashed".

    bootstrap_iterations: manifest.statistics.bootstrap_iterations. 양수이면
    percentile bootstrap CI, 0/None이면 t-분포 CI.
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
        ci = _mean_ci(values, bootstrap_iterations=bootstrap_iterations) if values else ("", "")
        summaries.append(
            {
                "condition": condition,
                "backbone": backbone,
                "n_seeds": len(completed),
                "macro_f1_mean": mean(values) if values else "",
                "macro_f1_std": stdev(values) if len(values) > 1 else "",
                "macro_f1_ci_low": ci[0],
                "macro_f1_ci_high": ci[1],
                "weighted_f1_mean": mean(weighted_values) if weighted_values else "",
                "accuracy_mean": mean(accuracy_values) if accuracy_values else "",
                "best_seed": best_seed,
                "median_seed": median_seed,
                "failed_seed_count": sum(1 for row in condition_rows if row["status"] == "failed"),
            }
        )
    return summaries


def _bootstrap_ci(
    values: list[float],
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float] | tuple[str, str]:
    """Percentile bootstrap CI for the mean.

    15 seed처럼 작은 표본에서 t-분포 CI는 정규성 가정이 약하므로 percentile
    bootstrap이 더 robust. numpy 미설치 환경에서는 _mean_ci_t로 fallback.
    """
    if not values:
        return ("", "")
    if len(values) == 1:
        return (values[0], values[0])
    if not _NP_AVAILABLE or _np is None:
        return _mean_ci_t(values, confidence=confidence)
    rng = _np.random.default_rng(seed)
    array = _np.asarray(values, dtype=float)
    n = array.size
    resamples = rng.choice(array, size=(iterations, n), replace=True)
    means = resamples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low = float(_np.quantile(means, alpha))
    high = float(_np.quantile(means, 1.0 - alpha))
    return (low, high)


def _mean_ci_t(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """t-distribution CI fallback (numpy 미설치 환경용)."""
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


def _mean_ci(
    values: list[float],
    confidence: float = 0.95,
    bootstrap_iterations: int | None = None,
) -> tuple[float, float] | tuple[str, str]:
    """기본 CI: bootstrap_iterations가 0보다 크면 bootstrap, 아니면 t-분포.

    manifest.statistics.bootstrap_iterations가 양수일 때 bootstrap CI 사용.
    """
    if bootstrap_iterations is None or bootstrap_iterations <= 0:
        return _mean_ci_t(values, confidence=confidence)
    return _bootstrap_ci(values, iterations=int(bootstrap_iterations), confidence=confidence)


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
    """Compute same-seed paired comparisons requested by the manifest.

    bootstrap_iterations 는 manifest.statistics.bootstrap_iterations 에서 읽음.
    """
    comparisons = manifest.get("statistics", {}).get("paired_tests", [])
    bootstrap_iterations = int(
        manifest.get("statistics", {}).get("bootstrap_iterations", 0) or 0
    )
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
            ci_low, ci_high = (
                _mean_ci(differences, bootstrap_iterations=bootstrap_iterations)
                if differences
                else ("", "")
            )
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


def _filter_anova_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    """Keep only completed rows that have a usable metric value."""
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        value = row.get(metric, "")
        if value == "" or value is None:
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            continue
        filtered.append(row)
    return filtered


def _anova_table_to_rows(table: Any, base_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a statsmodels ANOVA DataFrame into our flat row list.

    eta_squared = SS_factor / SS_total. partial_eta_squared = SS_factor / (SS_factor + SS_residual).
    두 효과 크기는 Cohen(1988) 기준으로 small=0.01, medium=0.06, large=0.14.
    Residual 행은 SS만 있고 효과 크기 X.
    """
    # SS_total = sum of all sum_sq (Residual 포함).
    try:
        all_ss = [float(record.get("sum_sq", 0.0)) for _, record in table.iterrows()]
        total_ss = sum(all_ss)
    except Exception:
        total_ss = 0.0
    # SS_residual은 보통 마지막 row.
    residual_ss = 0.0
    try:
        for factor_name, record in table.iterrows():
            if "Residual" in str(factor_name):
                residual_ss = float(record.get("sum_sq", 0.0))
                break
    except Exception:
        residual_ss = 0.0

    out_rows: list[dict[str, Any]] = []
    for factor_name, record in table.iterrows():
        row = dict(base_row)
        row["factor"] = str(factor_name)
        sum_sq_raw = record.get("sum_sq", None)
        sum_sq_value = float(sum_sq_raw) if sum_sq_raw is not None else 0.0
        row["sum_sq"] = sum_sq_value
        # 'df'는 degrees of freedom — 컬럼명을 그대로 유지한다.
        df_value = record.get("df", "")
        row["df"] = float(df_value) if df_value not in ("", None) else ""
        f_value = record.get("F", "")
        # Residual row는 F/p가 NaN으로 들어온다. 빈 문자열로 떨어뜨려 CSV가 깔끔하게.
        is_residual = "Residual" in str(factor_name)
        try:
            row["F"] = float(f_value) if f_value not in ("", None) and not (isinstance(f_value, float) and math.isnan(f_value)) else ""
        except (TypeError, ValueError):
            row["F"] = ""
        p_value = record.get("PR(>F)", "")
        try:
            row["p_value"] = float(p_value) if p_value not in ("", None) and not (isinstance(p_value, float) and math.isnan(p_value)) else ""
        except (TypeError, ValueError):
            row["p_value"] = ""

        # Effect size. Residual 행은 두 값 모두 빈 값.
        if is_residual or total_ss <= 0:
            row["eta_squared"] = ""
            row["partial_eta_squared"] = ""
        else:
            row["eta_squared"] = round(sum_sq_value / total_ss, 4)
            denominator = sum_sq_value + residual_ss
            row["partial_eta_squared"] = (
                round(sum_sq_value / denominator, 4) if denominator > 0 else ""
            )
        out_rows.append(row)
    return out_rows


def compute_two_way_anova(
    rows: list[dict[str, Any]],
    family: str = "BERT",
    metric: str = "macro_f1",
) -> list[dict[str, Any]]:
    """Two-way ANOVA inside one backbone family (A/B/C/D conditions).

    factor 1 = attention_loss (False/True)
    factor 2 = vader (False/True)
    """
    if not _ANOVA_AVAILABLE or _pd is None:
        # statsmodels 미설치 환경에서는 빈 리스트로 떨어뜨려 CSV가 헤더만 생성되게.
        return []

    family_rows = [
        row for row in _filter_anova_rows(rows, metric) if row.get("backbone") == family
    ]
    if len(family_rows) < 4:
        # 최소 2x2 셀 각 1개씩이라도 채워야 의미 있는 ANOVA.
        return []

    try:
        frame = _pd.DataFrame(
            [
                {
                    metric: float(row[metric]),
                    "attention_loss": str(row.get("use_attention_loss")),
                    "vader": str(row.get("use_sentiment")),
                }
                for row in family_rows
            ]
        )
        # 단일 cell에 모든 데이터가 몰리면 ANOVA가 무의미. cell unique 가 4개 미만이면 skip.
        cell_combinations = frame.groupby(["attention_loss", "vader"]).size()
        if len(cell_combinations) < 4:
            return []
        model = _ols(f"{metric} ~ C(attention_loss) * C(vader)", data=frame).fit()
        table = _anova_lm(model, typ=2)
    except Exception:  # pragma: no cover - statsmodels의 다양한 fit 실패 케이스 방어.
        return []

    base_row = {"family": family, "metric": metric}
    return _anova_table_to_rows(table, base_row)


def compute_three_way_anova(
    rows: list[dict[str, Any]],
    metric: str = "macro_f1",
) -> list[dict[str, Any]]:
    """Three-way ANOVA across both backbone families and both ablation factors.

    factor 1 = backbone (BERT / RoBERTa)
    factor 2 = attention_loss (False/True)
    factor 3 = vader (False/True)
    """
    if not _ANOVA_AVAILABLE or _pd is None:
        return []

    all_rows = _filter_anova_rows(rows, metric)
    if len(all_rows) < 8:
        return []

    try:
        frame = _pd.DataFrame(
            [
                {
                    metric: float(row[metric]),
                    "backbone": str(row.get("backbone")),
                    "attention_loss": str(row.get("use_attention_loss")),
                    "vader": str(row.get("use_sentiment")),
                }
                for row in all_rows
            ]
        )
        cell_combinations = frame.groupby(["backbone", "attention_loss", "vader"]).size()
        if len(cell_combinations) < 8:
            return []
        formula = f"{metric} ~ C(backbone) * C(attention_loss) * C(vader)"
        model = _ols(formula, data=frame).fit()
        table = _anova_lm(model, typ=2)
    except Exception:  # pragma: no cover.
        return []

    base_row = {"metric": metric}
    return _anova_table_to_rows(table, base_row)


def aggregate(manifest: dict[str, Any]) -> dict[str, Path]:
    """Run the benchmark aggregation stage."""
    root = experiment_root(manifest["run_id"])
    benchmark_dir = root / "benchmark"
    rows = collect_benchmark_runs(manifest)
    bootstrap_iterations = int(
        manifest.get("statistics", {}).get("bootstrap_iterations", 0) or 0
    )
    summary = summarize_benchmark(rows, bootstrap_iterations=bootstrap_iterations)
    paired_tests = compute_paired_tests(manifest, rows)
    holm_tests = apply_holm_correction(paired_tests)

    # ANOVA는 데이터 부족(시드 1개 미만, cell 불완전 등) 시 빈 리스트를 돌려준다.
    # 빈 리스트여도 CSV는 헤더만이라도 항상 생성한다 — 산출물 contract 유지.
    anova_bert = compute_two_way_anova(rows, family="BERT")
    anova_roberta = compute_two_way_anova(rows, family="RoBERTa")
    anova_3way = compute_three_way_anova(rows)

    run_path = write_csv(benchmark_dir / "benchmark_runs.csv", rows, BENCHMARK_RUN_COLUMNS)
    summary_path = write_csv(benchmark_dir / "benchmark_summary.csv", summary, BENCHMARK_SUMMARY_COLUMNS)
    paired_path = write_csv(benchmark_dir / "paired_tests.csv", paired_tests, PAIRED_TEST_COLUMNS)
    holm_path = write_csv(benchmark_dir / "paired_tests_holm.csv", holm_tests, PAIRED_TEST_COLUMNS)
    anova_bert_path = write_csv(
        benchmark_dir / "anova_2way_bert.csv", anova_bert, ANOVA_2WAY_COLUMNS
    )
    anova_roberta_path = write_csv(
        benchmark_dir / "anova_2way_roberta.csv", anova_roberta, ANOVA_2WAY_COLUMNS
    )
    anova_3way_path = write_csv(
        benchmark_dir / "anova_3way.csv", anova_3way, ANOVA_3WAY_COLUMNS
    )

    return {
        "benchmark_runs": run_path,
        "benchmark_summary": summary_path,
        "paired_tests": paired_path,
        "paired_tests_holm": holm_path,
        "anova_2way_bert": anova_bert_path,
        "anova_2way_roberta": anova_roberta_path,
        "anova_3way": anova_3way_path,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Small CSV reader for report/dashboard code."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
