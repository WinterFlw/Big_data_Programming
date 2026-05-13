"""Shared filesystem paths for the v2 pipeline.

This module is deliberately tiny, but it is one of the most important files in
the v2 workspace. Every other pipeline module imports paths from here instead
of hard-coding folder names. That gives us one place to change the workspace
layout later if the server directory structure changes.
"""

from __future__ import annotations

from pathlib import Path


# BASE_DIR points to the top-level v2 workspace:
#
#   <repo>/v2/
#
# The pipeline package lives under <repo>/v2/pipeline, so parents[1] is the
# stable workspace root regardless of where the command was launched from.
BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_RUN_ID = "v2_15seed"

# All v2-owned inputs and outputs are kept under BASE_DIR. This is the point of
# the new v2/ folder: a teammate can open this folder and see the whole v2 line.
CONFIG_DIR = BASE_DIR / "configs"
DOCS_V2_DIR = BASE_DIR / "docs"
OUTPUT_DIR = BASE_DIR / "outputs"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"

# DEFAULT_CONFIG_PATH is the human-editable config checked into the workspace.
# DOC_MANIFEST_TEMPLATE is the documentation copy. The code prefers the config
# file, then falls back to the docs template if the config is missing.
DEFAULT_CONFIG_PATH = CONFIG_DIR / f"{DEFAULT_RUN_ID}.json"
DOC_MANIFEST_TEMPLATE = DOCS_V2_DIR / "manifest_template.json"


def experiment_root(run_id: str = DEFAULT_RUN_ID) -> Path:
    """Return the canonical root for a v2 experiment run.

    For the default run this resolves to:

        v2/outputs/experiments/v2_15seed/

    Every stage should write under this directory unless it is explicitly
    exporting a final copy elsewhere.
    """
    return EXPERIMENTS_DIR / run_id


def display_path(path: Path) -> str:
    """Return a v2-relative path when possible.

    CLI output should be readable and copy-paste friendly. Absolute paths make
    status tables noisy, so we shorten paths that live under BASE_DIR.
    """
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)
