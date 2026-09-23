#!/usr/bin/env python3
"""Validate the agentic-rag skill package without third-party dependencies."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
REQUIRED_PATHS = [
    "SKILL.md",
    "agents/openai.yaml",
    "references/agentic-rag-behavior.md",
    "references/agentic-rag-behavior-summary.md",
    "references/codex-completion-brief.md",
    "references/prompts-and-schemas.md",
    "references/source-map.md",
    "src/agentic_rag/contracts.py",
    "src/agentic_rag/orchestrator.py",
    "src/agentic_rag/adapters/in_memory.py",
]


def parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    try:
        _, frontmatter, _ = text.split("---", 2)
    except ValueError as exc:
        raise ValueError("SKILL.md frontmatter must be closed with ---") from exc

    fields: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in frontmatter.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" ") and ":" in line:
            if current_key:
                fields[current_key] = " ".join(current_lines).strip()
            key, value = line.split(":", 1)
            current_key = key.strip()
            current_lines = [value.strip().strip('"')]
            continue
        if current_key:
            current_lines.append(line.strip())

    if current_key:
        fields[current_key] = " ".join(current_lines).strip()
    return fields


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_PATHS:
        if not (root / rel).exists():
            errors.append(f"missing required file: {rel}")

    skill_path = root / "SKILL.md"
    if not skill_path.exists():
        return errors

    try:
        fields = parse_frontmatter(skill_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    unexpected = sorted(set(fields) - {"name", "description"})
    if unexpected:
        errors.append(f"unexpected SKILL.md frontmatter fields: {', '.join(unexpected)}")

    name = fields.get("name", "")
    if not NAME_RE.match(name):
        errors.append(f"invalid skill name: {name!r}")
    if name and name != root.name and root.name not in ("evidencefirst-demo", "Agentic-RAG-Skill"):
        errors.append(f"skill name {name!r} should match install directory {root.name!r}")

    description = fields.get("description", "")
    if not description:
        errors.append("missing description")
    elif len(description) > 1024:
        errors.append("description exceeds 1024 characters")

    if len(skill_path.read_text(encoding="utf-8").splitlines()) >= 200:
        errors.append("SKILL.md should stay under 200 lines and use references for detail")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Path to the agentic-rag skill directory",
    )
    args = parser.parse_args()
    root = args.path.resolve()
    errors = validate(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: {root} is a valid agentic-rag skill package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
