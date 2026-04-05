"""Compatibility wrapper for the benchmark suite."""

from run_experiments import main


if __name__ == "__main__":
    raise SystemExit(main(["benchmark"]))
