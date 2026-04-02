"""Command-line entrypoint for the report-aligned experiment pipeline."""

from __future__ import annotations

import argparse

import pandas as pd

from experiment_dashboard import DASHBOARD_HTML_PATH, run_dashboard
from experiment_core import (
    BENCHMARK_SUMMARY_PATH,
    FREEZE_STUDY_PATH,
    OUTPUT_DIR,
    REPORT_DIR,
    TUNING_LOG_PATH,
    VADER_PICKLE_PATH,
    XAI_DIR,
    describe_status,
    extract_vader_features,
    get_config,
    prepare_data,
    run_benchmark,
    run_freeze_study,
    run_hyperparameter_tuning,
    save_config,
)
from experiment_xai import run_xai
from utils import CHECKPOINT_DIR, dataframe_to_markdown, remove_tree, save_text


def _print_status() -> None:
    status = describe_status()
    print("\nPipeline status")
    print("================")
    for key, value in status.items():
        print(f"{key:20s}: {'ready' if value else 'missing'}")


def _clean_outputs() -> None:
    for directory in [OUTPUT_DIR, CHECKPOINT_DIR]:
        remove_tree(directory)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("Removed outputs/ and checkpoints/ artifacts.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hate-speech experiments aligned with the report.")
    parser.add_argument(
        "command",
        choices=["data", "vader", "tune", "benchmark", "freeze-study", "xai", "dashboard", "all", "status", "clean"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild cached artifacts for this stage.")
    parser.add_argument(
        "--with-tuning",
        action="store_true",
        help="When running 'all', execute tuning before the benchmark.",
    )
    args = parser.parse_args(argv)

    config = get_config()
    save_config(config)

    if args.command == "data":
        prepare_data(config=config, force_refresh=args.force, force_download=False)
        dashboard_path = run_dashboard()
        print(f"Saved prepared splits under {OUTPUT_DIR}")
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    if args.command == "vader":
        splits = prepare_data(config=config)
        extract_vader_features(splits=splits, force_refresh=args.force)
        dashboard_path = run_dashboard()
        print(f"Saved VADER features to {VADER_PICKLE_PATH}")
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    if args.command == "tune":
        tuning_summary = run_hyperparameter_tuning(config=config)
        dashboard_path = run_dashboard()
        print("Tuning complete.")
        for model_name, params in tuning_summary.items():
            print(f"  {model_name}: {params}")
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    if args.command == "benchmark":
        summary = run_benchmark(config=config)
        dashboard_path = run_dashboard()
        print("Benchmark complete.")
        print(summary[["model", "macro_f1_display", "macro_precision_display", "macro_recall_display", "accuracy_display"]].to_string(index=False))
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    if args.command == "freeze-study":
        summary = run_freeze_study(config=config)
        dashboard_path = run_dashboard()
        print("Freeze study complete.")
        print(summary[["model", "macro_f1_display", "macro_precision_display", "macro_recall_display", "accuracy_display"]].to_string(index=False))
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    if args.command == "xai":
        summary = run_xai(config=config)
        dashboard_path = run_dashboard()
        print("XAI complete.")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    if args.command == "dashboard":
        dashboard_path = run_dashboard()
        print(f"Dashboard generated: {dashboard_path}")
        return 0

    if args.command == "all":
        prepare_data(config=config, force_refresh=args.force, force_download=False)
        extract_vader_features(force_refresh=args.force)
        if args.with_tuning:
            run_hyperparameter_tuning(config=config)
        benchmark_summary = run_benchmark(config=config)
        freeze_summary = run_freeze_study(config=config)
        xai_summary = run_xai(config=config)

        final_report = (
            "# Final Experiment Bundle\n\n"
            "## Benchmark Summary\n\n"
            + dataframe_to_markdown(
                benchmark_summary[
                    [
                        "model",
                        "macro_f1_display",
                        "macro_precision_display",
                        "macro_recall_display",
                        "accuracy_display",
                    ]
                ]
            )
            + "\n\n## Freeze Study\n\n"
            + dataframe_to_markdown(
                freeze_summary[
                    [
                        "model",
                        "macro_f1_display",
                        "macro_precision_display",
                        "macro_recall_display",
                        "accuracy_display",
                    ]
                ]
            )
            + "\n\n## XAI Summary\n\n"
            + dataframe_to_markdown(
                pd.DataFrame([xai_summary])
            )
        )
        save_text(final_report, REPORT_DIR / "final_bundle.md")
        dashboard_path = run_dashboard()
        print("Full pipeline complete.")
        print(f"Benchmark summary: {BENCHMARK_SUMMARY_PATH}")
        print(f"Freeze study: {FREEZE_STUDY_PATH}")
        print(f"Tuning log: {TUNING_LOG_PATH if TUNING_LOG_PATH.exists() else 'not generated'}")
        print(f"XAI outputs: {XAI_DIR}")
        print(f"Dashboard: {dashboard_path}")
        return 0

    if args.command == "status":
        _print_status()
        return 0

    if args.command == "clean":
        _clean_outputs()
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
