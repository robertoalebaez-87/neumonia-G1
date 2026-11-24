#!/usr/bin/env python
"""Lanzador de la interfaz web local de detección de neumonía.

La lógica vive en el paquete :mod:`neumonia` (``src/neumonia``). Este archivo se
conserva por compatibilidad con la invocación histórica
``python detector_neumonia.py``.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Punto de entrada compatible con la versión de un solo archivo."""
    try:
        from neumonia.app import main as app_main
    except ImportError as exc:  # pragma: no cover - ayuda de arranque
        print(
            "No se pudo importar el paquete 'neumonia'. "
            "Ejecuta 'uv sync --extra ml' o 'make install' y vuelve a intentarlo.",
            file=sys.stderr,
        )
        print(f"Detalle: {exc}", file=sys.stderr)
        return 1
    app_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
