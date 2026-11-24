"""Limpia cachés y artefactos del proyecto (``make clean``)."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = (".pytest_cache", ".ruff_cache", ".coverage", "htmlcov", "build", "dist", ".venv")
PATTERNS = ("__pycache__", "*.egg-info")


def main() -> int:
    for name in TARGETS:
        _remove(ROOT / name)
    for pattern in PATTERNS:
        for path in ROOT.rglob(pattern):
            _remove(path)
    print("Limpieza completada.")
    return 0


def _remove(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        print(f"  eliminado dir: {path.relative_to(ROOT)}")
    else:
        path.unlink(missing_ok=True)
        print(f"  eliminado archivo: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
