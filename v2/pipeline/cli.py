"""CLI for the v2 end-to-end pipeline.

This file translates shell commands into stage function calls. It should stay
thin: argument parsing belongs here, pipeline behavior belongs in runner.py.
That separation makes it easier for teammates to add stage logic without
accidentally changing the command-line contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import runner
from .paths import DEFAULT_RUN_ID


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    # Every stage accepts the same run selector. Keeping this helper small is
    # intentional: when we later add common options such as --verbose or
    # --manifest, one change updates every subcommand consistently.
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Experiment run id.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSON path.")


def build_parser() -> argparse.ArgumentParser:
    # The top-level parser only knows that we are running "e2e". The actual
    # stages are subcommands so a teammate can run one stage at a time on a
    # limited server allocation.
    parser = argparse.ArgumentParser(prog="./run.sh e2e", description="Run v2 end-to-end experiment stages.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    # plan is the first command to run. It creates the manifest copy, the output
    # tree, and the 120 condition x seed execution table. It does not train.
    plan_parser = subparsers.add_parser("plan", help="Create manifest, directories, and execution plan.")
    _add_common_options(plan_parser)
    plan_parser.add_argument("--force", action="store_true", help="Overwrite existing output manifest.")

    # These stages do not need specialized CLI arguments yet. They still accept
    # --run-id and --manifest through _add_common_options.
    for name in ["status", "data", "aggregate", "report", "dashboard"]:
        stage_parser = subparsers.add_parser(name, help=f"Run {name} stage.")
        _add_common_options(stage_parser)

    # benchmark is the expensive stage. Subset selectors are mandatory for
    # safe smoke tests because the full plan is 8 conditions x 15 seeds.
    benchmark_parser = subparsers.add_parser("benchmark", help="Plan or execute condition x seed benchmark units.")
    _add_common_options(benchmark_parser)
    benchmark_parser.add_argument("--conditions", default=None, help="Comma-separated condition subset.")
    benchmark_parser.add_argument("--seeds", default=None, help="Comma-separated seed subset.")
    benchmark_parser.add_argument("--dry-run", action="store_true", help="Only print/write planned units.")
    benchmark_parser.add_argument("--execute", action="store_true", help="Run the training adapter when implemented.")
    benchmark_parser.add_argument("--resume", action="store_true", help="Reserved for execution adapter.")
    benchmark_parser.add_argument("--force", action="store_true", help="Reserved for execution adapter.")

    # XAI can be even more expensive than training, so each XAI mode is a
    # separate command. That lets us run primary/deep/ablation only after the
    # benchmark and statistics outputs prove the checkpoints are usable.
    for name in ["xai-primary", "xai-deep", "xai-ablation"]:
        stage_parser = subparsers.add_parser(name, help=f"Plan or execute {name}.")
        _add_common_options(stage_parser)
        stage_parser.add_argument("--dry-run", action="store_true", help="Only show the planned XAI scope.")
        stage_parser.add_argument("--resume", action="store_true", help="Reserved for execution adapter.")
        stage_parser.add_argument("--force", action="store_true", help="Reserved for execution adapter.")

    all_parser = subparsers.add_parser("all", help="Create all planned scaffolding artifacts.")
    _add_common_options(all_parser)
    all_parser.add_argument("--force", action="store_true", help="Overwrite existing output manifest.")
    all_parser.add_argument("--resume", action="store_true", help="Reserved for execution adapter.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Dispatch stays explicit instead of using getattr magic. It is a little
    # longer, but a new teammate can search a stage name and see exactly which
    # runner function owns it.
    if args.stage == "plan":
        result = runner.plan(args.run_id, args.manifest, force=args.force)
    elif args.stage == "status":
        result = runner.status(args.run_id, args.manifest)
    elif args.stage == "data":
        result = runner.data(args.run_id, args.manifest)
    elif args.stage == "benchmark":
        result = runner.benchmark(
            args.run_id,
            args.manifest,
            conditions_value=args.conditions,
            seeds_value=args.seeds,
            dry_run=args.dry_run,
            execute=args.execute,
        )
    elif args.stage == "aggregate":
        result = runner.aggregate_stage(args.run_id, args.manifest)
    elif args.stage == "xai-primary":
        result = runner.xai_primary(args.run_id, args.manifest, dry_run=args.dry_run)
    elif args.stage == "xai-deep":
        result = runner.xai_deep(args.run_id, args.manifest, dry_run=args.dry_run)
    elif args.stage == "xai-ablation":
        result = runner.xai_ablation(args.run_id, args.manifest, dry_run=args.dry_run)
    elif args.stage == "report":
        result = runner.report(args.run_id, args.manifest)
    elif args.stage == "dashboard":
        result = runner.dashboard(args.run_id, args.manifest)
    elif args.stage == "all":
        result = runner.all_stages(args.run_id, args.manifest, force=args.force)
    else:
        parser.error(f"Unknown e2e stage: {args.stage}")

    # Runner functions return dictionaries so tests and agents can inspect
    # structured results. The CLI formats them only at the boundary.
    print(runner.format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
