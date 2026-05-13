#!/usr/bin/env python3
"""Validate v2 commit messages.

The validator is intentionally strict because v2 commits double as an audit log
for server runs, statistical changes, and XAI/reporting work.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ALLOWED_TYPES = {
    "feat",
    "fix",
    "docs",
    "refactor",
    "test",
    "chore",
    "ci",
    "perf",
    "build",
    "revert",
}

ALLOWED_SCOPES = {
    "v2",
    "pipeline",
    "benchmark",
    "stats",
    "xai",
    "report",
    "qa",
    "docs",
    "agent",
    "config",
    "scripts",
    "outputs",
    "root",
}

HEADER_RE = re.compile(
    rf"^({'|'.join(sorted(ALLOWED_TYPES))})\(({'|'.join(sorted(ALLOWED_SCOPES))})\): (.+)$"
)

REQUIRED_SECTIONS = ["Why:", "What:", "Validation:"]


def _strip_comment_lines(text: str) -> list[str]:
    """Return commit message lines excluding Git's commented template hints."""
    return [line.rstrip() for line in text.splitlines() if not line.lstrip().startswith("#")]


def _first_non_empty_line(lines: list[str]) -> tuple[int, str] | None:
    for index, line in enumerate(lines):
        if line.strip():
            return index, line
    return None


def validate_message(text: str) -> list[str]:
    """Return validation errors. Empty list means the message is accepted."""
    errors: list[str] = []
    lines = _strip_comment_lines(text)
    first = _first_non_empty_line(lines)

    if first is None:
        return ["commit message is empty"]

    header_index, header = first

    # Let Git-generated merge/revert/fixup messages pass. The normal v2 workflow
    # should still use the strict template for hand-written commits.
    if header.startswith(("Merge ", "Revert ", "fixup!", "squash!")):
        return []

    match = HEADER_RE.match(header)
    if not match:
        allowed_types = ", ".join(sorted(ALLOWED_TYPES))
        allowed_scopes = ", ".join(sorted(ALLOWED_SCOPES))
        errors.append("header must match <type>(<scope>): <summary>")
        errors.append(f"allowed types: {allowed_types}")
        errors.append(f"allowed scopes: {allowed_scopes}")
        return errors

    summary = match.group(3).strip()
    if len(summary) < 10:
        errors.append("summary must be at least 10 characters")
    if len(summary) > 72:
        errors.append("summary must be 72 characters or fewer")
    if summary.endswith("."):
        errors.append("summary must not end with a period")

    if len(lines) <= header_index + 1 or lines[header_index + 1].strip():
        errors.append("header must be followed by a blank line")

    body = "\n".join(lines[header_index + 1 :])
    for section in REQUIRED_SECTIONS:
        if section not in body:
            errors.append(f"missing required section: {section}")

    validation_index = body.find("Validation:")
    if validation_index != -1:
        validation_body = body[validation_index + len("Validation:") :].strip()
        if not validation_body:
            errors.append("Validation section must list commands or a Not run reason")
        elif "Not run:" in validation_body:
            # Explicitly allowed, but it must include an explanation after the
            # marker so reviewers know why validation was skipped.
            not_run_text = validation_body.split("Not run:", 1)[1].strip()
            if not not_run_text:
                errors.append("Not run must include a reason")

    return errors


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: validate_commit_message.py <commit-msg-file>", file=sys.stderr)
        return 2

    message_path = Path(args[0])
    errors = validate_message(message_path.read_text(encoding="utf-8"))
    if errors:
        print("Invalid commit message:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print("\nSee v2/docs/13_commit_message_policy.md", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

