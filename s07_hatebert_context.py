"""Compatibility wrapper for the freeze-study suite."""

from run_experiments import main


if __name__ == "__main__":
    raise SystemExit(main(["freeze-study"]))
