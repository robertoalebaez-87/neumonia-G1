"""Muestra los objetivos del Makefile con sus descripciones (``make help``)."""

from __future__ import annotations

import re
from pathlib import Path

PATTERN = re.compile(r"^([a-zA-Z_-]+):.*?## (.*)$")


def main() -> int:
    makefile = Path(__file__).resolve().parent.parent / "Makefile"
    print("Objetivos disponibles:\n")
    width = 0
    entries: list[tuple[str, str]] = []
    for line in makefile.read_text(encoding="utf-8").splitlines():
        match = PATTERN.match(line)
        if match:
            name, description = match.groups()
            entries.append((name, description))
            width = max(width, len(name))
    for name, description in sorted(entries):
        print(f"  {name.ljust(width)}  {description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
