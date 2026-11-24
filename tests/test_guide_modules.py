"""Pruebas de los módulos públicos exigidos por la guía y de la caché del modelo."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from neumonia.model import clear_model_cache, load_model


@pytest.fixture(autouse=True)
def _limpiar_cache_modelo() -> Iterator[None]:
    clear_model_cache()
    yield
    clear_model_cache()


@pytest.mark.parametrize(
    ("modulo", "funcion"),
    [
        ("load_model", "load_model"),
        ("read_img", "read_jpg_file"),
        ("preprocess_img", "preprocess"),
        ("grad_cam", "grad_cam"),
        ("integrator", "predict"),
        ("ui", "App"),
    ],
)
def test_modulo_gui_exponen_api(modulo: str, funcion: str) -> None:
    assert callable(getattr(import_module(modulo), funcion))


def test_modulo_gui_read_img_expone_dicom() -> None:
    from read_img import read_dicom_file

    assert callable(read_dicom_file)


def test_modulo_gui_preprocess_expone_normalizacion() -> None:
    from preprocess_img import normalize_to_uint8

    assert callable(normalize_to_uint8)


def test_modulo_gui_grad_cam_expone_matematica() -> None:
    from grad_cam import compute_gradcam

    assert callable(compute_gradcam)


def test_modulo_gui_integrator_expone_demo() -> None:
    from integrator import run_demo

    assert callable(run_demo)


def test_modulo_gui_ui_expone_titulo() -> None:
    from ui import APP_TITLE

    assert isinstance(APP_TITLE, str)


def test_modulo_gui_load_model_reexporta_implementacion() -> None:
    from load_model import load_model as cargador_gui

    assert cargador_gui is load_model


class _TensorFlowFalso(ModuleType):
    compat: SimpleNamespace
    keras: SimpleNamespace


def _tensorflow_falso(llamadas: list[str]) -> ModuleType:
    tf = _TensorFlowFalso("tensorflow")

    def cargar(path: str) -> object:
        llamadas.append(path)
        return object()

    tf.compat = SimpleNamespace(
        v1=SimpleNamespace(
            disable_eager_execution=lambda: None,
            experimental=SimpleNamespace(output_all_intermediates=lambda _valor: None),
        )
    )
    tf.keras = SimpleNamespace(models=SimpleNamespace(load_model=cargar))
    return tf


def test_load_model_carga_una_sola_vez(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llamadas: list[str] = []
    monkeypatch.setitem(sys.modules, "tensorflow", _tensorflow_falso(llamadas))
    ruta = tmp_path / "modelo.h5"

    primero = load_model(ruta)
    segundo = load_model(ruta)

    assert primero is segundo
    assert llamadas == [str(ruta)]


def test_load_model_iguala_str_y_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llamadas: list[str] = []
    monkeypatch.setitem(sys.modules, "tensorflow", _tensorflow_falso(llamadas))
    ruta = tmp_path / "modelo.h5"

    load_model(ruta)
    load_model(str(ruta))

    assert llamadas == [str(ruta)]


def test_clear_model_cache_permite_recargar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    llamadas: list[str] = []
    monkeypatch.setitem(sys.modules, "tensorflow", _tensorflow_falso(llamadas))
    ruta = tmp_path / "modelo.h5"

    load_model(ruta)
    clear_model_cache()
    load_model(ruta)

    assert len(llamadas) == 2
