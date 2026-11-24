"""Pruebas del modo demostración sin modelo entrenado."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from neumonia.constants import CLASS_NAMES
from neumonia.demo import DEMO_NOTICE, demo_heatmap, demo_predictions, run_demo, should_use_demo
from neumonia.images import read_jpg_file
from neumonia.predict import interpret_predictions


def test_should_use_demo_sin_modelo(tmp_path: Path) -> None:
    assert should_use_demo(tmp_path / "WilhemNet86.h5") is True


def test_should_use_demo_con_modelo(tmp_path: Path) -> None:
    modelo = tmp_path / "modelo.h5"
    modelo.touch()
    assert should_use_demo(modelo) is False


def test_should_use_demo_forzado(tmp_path: Path) -> None:
    modelo = tmp_path / "modelo.h5"
    modelo.touch()
    assert should_use_demo(modelo, force=True) is True


def test_demo_predictions_devuelve_tres_probabilidades() -> None:
    assert demo_predictions(np.zeros((2, 2, 3))).shape == (3,)


def test_demo_predictions_suma_uno() -> None:
    assert demo_predictions(np.full((2, 2, 3), 100)).sum() == pytest.approx(1.0)


def test_demo_predictions_es_determinista() -> None:
    array = np.full((2, 2, 3), 100, dtype=np.uint8)
    assert np.array_equal(demo_predictions(array), demo_predictions(array))


@pytest.mark.parametrize(
    ("valor", "etiqueta"),
    [(0, "bacteriana"), (128, "normal"), (255, "viral")],
)
def test_demo_predictions_varia_con_brillo(valor: int, etiqueta: str) -> None:
    array = np.full((2, 2, 3), valor, dtype=np.uint8)
    label, _proba, _index = interpret_predictions(demo_predictions(array))
    assert label == etiqueta


def test_demo_predictions_rechaza_arreglo_vacio() -> None:
    with pytest.raises(ValueError):
        demo_predictions(np.array([]))


def test_demo_heatmap_devuelve_rgb_uint8() -> None:
    result = demo_heatmap(np.zeros((4, 5, 3), dtype=np.uint8))
    assert result.shape == (4, 5, 3)
    assert result.dtype == np.uint8


def test_demo_heatmap_acepta_imagen_en_gris() -> None:
    assert demo_heatmap(np.zeros((4, 5), dtype=np.uint8)).shape == (4, 5, 3)


def test_demo_heatmap_no_modifica_la_entrada() -> None:
    array = np.full((2, 2, 3), 50, dtype=np.uint8)
    original = array.copy()
    demo_heatmap(array)
    assert np.array_equal(array, original)


def test_demo_heatmap_rechaza_cuatro_dimensiones() -> None:
    with pytest.raises(ValueError):
        demo_heatmap(np.zeros((2, 2, 2, 2)))


def test_run_demo_devuelve_tripleta() -> None:
    label, proba, heatmap = run_demo(np.zeros((3, 3, 3), dtype=np.uint8))
    assert isinstance(label, str)
    assert isinstance(proba, float)
    assert heatmap.shape == (3, 3, 3)


def test_run_demo_etiqueta_conocida() -> None:
    label, _proba, _heatmap = run_demo(np.zeros((3, 3, 3), dtype=np.uint8))
    assert label in CLASS_NAMES.values()


def test_run_demo_probabilidad_en_rango() -> None:
    _label, proba, _heatmap = run_demo(np.zeros((3, 3, 3), dtype=np.uint8))
    assert 0.0 <= proba <= 100.0


def test_notice_adverte_sobre_diagnostico() -> None:
    assert "diagnóstico" in DEMO_NOTICE


def test_read_jpg_sin_cv2_usa_pillow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ruta = tmp_path / "imagen.jpg"
    Image.new("RGB", (3, 2), (10, 20, 30)).save(ruta)
    monkeypatch.setitem(sys.modules, "cv2", None)
    _normalized, display = read_jpg_file(ruta)
    assert display.shape == (2, 3, 3)
