"""Pruebas de la regla de dependencia: ui.py solo depende de integrator."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from integrator import analyze

SRC = Path(__file__).resolve().parents[1] / "src"
INTERNOS = {
    "integrator",
    "load_model",
    "read_img",
    "preprocess_img",
    "grad_cam",
    "ui",
    "neumonia",
}


def _imports(ruta: Path) -> tuple[set[str], set[str]]:
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    modulos: set[str] = set()
    nombres: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                modulos.add(alias.name)
                nombres.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module)
            nombres.update(alias.name for alias in nodo.names)
    return modulos, nombres


def test_ui_solo_importa_integrator_entre_modulos_internos() -> None:
    modulos, _nombres = _imports(SRC / "ui.py")
    internos = {
        modulo for modulo in modulos if modulo in INTERNOS or modulo.startswith("neumonia.")
    }
    assert internos == {"integrator"}


def test_ui_no_importa_tensorflow_ni_load_model() -> None:
    modulos, nombres = _imports(SRC / "ui.py")
    assert not any(modulo == "tensorflow" or modulo.startswith("tensorflow.") for modulo in modulos)
    assert "load_model" not in modulos
    assert "load_model" not in nombres


def test_neumonia_app_reexporta_ui() -> None:
    from neumonia.app import App as AppLegacy
    from ui import App

    assert AppLegacy is App


def test_analyze_en_modo_demo_no_necesita_modelo() -> None:
    etiqueta, probabilidad, mapa = analyze(np.zeros((3, 3, 3), dtype=np.uint8), demo=True)
    assert isinstance(etiqueta, str)
    assert 0.0 <= probabilidad <= 100.0
    assert mapa.shape == (3, 3, 3)
