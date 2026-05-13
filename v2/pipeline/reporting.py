"""Report and dashboard scaffolding for v2 experiments.

The reporting stage should be able to run before full results exist. That lets
the team verify links, sections, and output paths locally, then rerun the same
command after the server fills benchmark and XAI artifacts.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .artifacts import build_run_units, status_counts
from .manifest import manifest_hash
from .paths import experiment_root


def generate_markdown_report(manifest: dict[str, Any]) -> Path:
    """Write a Markdown report scaffold for one run_id."""
    root = experiment_root(manifest["run_id"])
    units = build_run_units(manifest)
    counts = status_counts(units)
    report_path = root / "reports" / "final_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # This text is deliberately conservative. Until real results exist, it only
    # states the design and current artifact status. The future report builder
    # should replace the hook section with tables loaded from benchmark/XAI CSVs.
    text = f"""# v2 Final Report

## Run

- run_id: `{manifest["run_id"]}`
- manifest_hash: `{manifest_hash(manifest)}`
- output_root: `{manifest["output_root"]}`

## Benchmark Status

- total units: {counts["total"]}
- completed: {counts["completed"]}
- failed: {counts["failed"]}
- planned: {counts["planned"]}

## Model

The v2 model family evaluates rationale-aware attention loss and VADER sentiment features through an 8-condition ablation matrix across BERT and RoBERTa backbones.

## Statistics

The primary comparison uses same-seed paired tests across 15 seeds. Holm correction is reserved for multiple comparisons.

## XAI

XAI is treated as post-hoc verification. Primary XAI compares A_B and D_B across all seeds, while deep XAI uses median-performing checkpoints for detailed cases.

## Next Implementation Hooks

- Fill `benchmark/benchmark_runs.csv` from completed condition x seed runs.
- Fill `benchmark/paired_tests_holm.csv` after aggregation.
- Fill `xai/primary/seed_level_metrics.csv` after XAI execution.
- Convert this Markdown report to DOCX after real results are populated.
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def generate_dashboard(manifest: dict[str, Any]) -> Path:
    """Write a minimal HTML dashboard for one run_id."""
    root = experiment_root(manifest["run_id"])
    units = build_run_units(manifest)
    counts = status_counts(units)
    dashboard_path = root / "dashboard" / "index.html"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)

    # Escape text that comes from the manifest so the dashboard remains safe if
    # a run_id or path contains punctuation.
    rows = "\n".join(
        f"<tr><th>{escape(key)}</th><td>{value}</td></tr>"
        for key, value in counts.items()
    )
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>{escape(manifest["run_id"])} dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; line-height: 1.5; }}
    table {{ border-collapse: collapse; min-width: 360px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{escape(manifest["run_id"])} dashboard</h1>
  <p>Output root: <code>{escape(manifest["output_root"])}</code></p>
  <table>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path
