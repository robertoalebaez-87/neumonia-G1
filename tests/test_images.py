"""Pruebas de lectura y preprocesamiento de imágenes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neumonia.images import (
    normalize_to_uint8,
    preprocess,
    read_dicom_file,
    read_jpg_file,
    resize_for_display,
)


def test_normalize_imagen_negra_devuelve_ceros() -> None:
    result = normalize_to_uint8(np.zeros((2, 2)))
    assert np.array_equal(result, np.zeros((2, 2), dtype=np.uint8))


def test_normalize_escala_maximo_a_255() -> None:
    result = normalize_to_uint8(np.array([[0, 50], [100, 200]], dtype=float))
    assert np.array_equal(result, np.array([[0, 63], [127, 255]], dtype=np.uint8))


def test_normalize_recorta_negativos() -> None:
    result = normalize_to_uint8(np.array([-5, 0, 10], dtype=float))
    assert np.array_equal(result, np.array([0, 0, 255], dtype=np.uint8))


def test_normalize_devuelve_dtype_uint8() -> None:
    assert normalize_to_uint8(np.array([1.0, 2.0])).dtype == np.uint8


def test_normalize_conserva_la_forma() -> None:
    result = normalize_to_uint8(np.arange(12, dtype=float).reshape(3, 4))
    assert result.shape == (3, 4)


def test_normalize_arreglo_vacio() -> None:
    result = normalize_to_uint8(np.array([], dtype=float))
    assert result.size == 0
    assert result.dtype == np.uint8


def test_normalize_todo_negativo_devuelve_ceros() -> None:
    result = normalize_to_uint8(np.array([-1.0, -2.0, -3.0]))
    assert np.array_equal(result, np.zeros(3, dtype=np.uint8))


def test_normalize_constante_positiva_es_toda_blanca() -> None:
    result = normalize_to_uint8(np.full((2, 2), 7.0))
    assert np.array_equal(result, np.full((2, 2), 255, dtype=np.uint8))


def test_normalize_no_modifica_la_entrada() -> None:
    data = np.array([0.0, 50.0, 100.0])
    original = data.copy()
    normalize_to_uint8(data)
    assert np.array_equal(data, original)


def test_normalize_conserva_proporcion() -> None:
    result = normalize_to_uint8(np.array([1.0, 2.0, 4.0]))
    assert result[0] == 63
    assert result[1] == 127
    assert result[2] == 255


def _fake_ops(size: tuple[int, int], calls: dict[str, int]):
    def resize(array: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        calls["resize"] = calls.get("resize", 0) + 1
        return np.full((target[1], target[0], 3), 51.0)

    def to_gray(array: np.ndarray) -> np.ndarray:
        calls["to_gray"] = calls.get("to_gray", 0) + 1
        return array[:, :, 0]

    def clahe_apply(array: np.ndarray) -> np.ndarray:
        calls["clahe"] = calls.get("clahe", 0) + 1
        return array

    return resize, to_gray, clahe_apply


def test_preprocess_devuelve_cuatro_dimensiones() -> None:
    resize, to_gray, clahe = _fake_ops((3, 2), {})
    result = preprocess(np.zeros((4, 4, 3)), resize=resize, to_gray=to_gray, clahe_apply=clahe)
    assert result.ndim == 4


def test_preprocess_primera_dimension_es_uno() -> None:
    resize, to_gray, clahe = _fake_ops((3, 2), {})
    result = preprocess(np.zeros((4, 4, 3)), resize=resize, to_gray=to_gray, clahe_apply=clahe)
    assert result.shape[0] == 1


def test_preprocess_ultima_dimension_es_uno() -> None:
    resize, to_gray, clahe = _fake_ops((3, 2), {})
    result = preprocess(np.zeros((4, 4, 3)), resize=resize, to_gray=to_gray, clahe_apply=clahe)
    assert result.shape[-1] == 1


def test_preprocess_respeta_tamano_personalizado() -> None:
    resize, to_gray, clahe = _fake_ops((3, 2), {})
    result = preprocess(
        np.zeros((4, 4, 3)), resize=resize, to_gray=to_gray, clahe_apply=clahe, size=(3, 2)
    )
    assert result.shape == (1, 2, 3, 1)


def test_preprocess_normaliza_entre_cero_y_uno() -> None:
    resize, to_gray, clahe = _fake_ops((3, 2), {})
    result = preprocess(np.zeros((4, 4, 3)), resize=resize, to_gray=to_gray, clahe_apply=clahe)
    assert np.allclose(result, 51.0 / 255.0)


def test_preprocess_invoca_resize_inyectado() -> None:
    calls: dict[str, int] = {}
    resize, to_gray, clahe = _fake_ops((3, 2), calls)
    preprocess(np.zeros((4, 4, 3)), resize=resize, to_gray=to_gray, clahe_apply=clahe)
    assert calls.get("resize") == 1


def test_preprocess_invoca_to_gray_inyectado() -> None:
    calls: dict[str, int] = {}
    resize, to_gray, clahe = _fake_ops((3, 2), calls)
    preprocess(np.zeros((4, 4, 3)), resize=resize, to_gray=to_gray, clahe_apply=clahe)
    assert calls.get("to_gray") == 1


def test_preprocess_invoca_clahe_inyectado() -> None:
    calls: dict[str, int] = {}
    resize, to_gray, clahe = _fake_ops((3, 2), calls)
    preprocess(np.zeros((4, 4, 3)), resize=resize, to_gray=to_gray, clahe_apply=clahe)
    assert calls.get("clahe") == 1


def test_read_jpg_devuelve_dos_arreglos() -> None:
    original = np.array([[[0, 0, 0], [100, 100, 100]]], dtype=np.uint8)
    normalized, display = read_jpg_file("img.jpg", imread=lambda _path: original)
    assert normalized.shape == original.shape
    assert display.shape == original.shape


def test_read_jpg_normaliza_uint8() -> None:
    original = np.array([[[0, 0, 0], [100, 100, 100]]], dtype=np.uint8)
    normalized, _ = read_jpg_file("img.jpg", imread=lambda _path: original)
    assert normalized.dtype == np.uint8


def test_read_jpg_original_intacto() -> None:
    original = np.array([[[0, 0, 0], [100, 100, 100]]], dtype=np.uint8)
    _, display = read_jpg_file("img.jpg", imread=lambda _path: original)
    assert np.array_equal(display, original)


def test_read_jpg_escala_al_maximo() -> None:
    original = np.array([[[0, 0, 0], [100, 100, 100]]], dtype=np.uint8)
    normalized, _ = read_jpg_file("img.jpg", imread=lambda _path: original)
    assert int(normalized.max()) == 255


def test_read_jpg_lanza_error_si_imread_falla() -> None:
    with pytest.raises(FileNotFoundError):
        read_jpg_file("noexiste.jpg", imread=lambda _path: None)


def test_read_jpg_pasa_la_ruta_a_imread() -> None:
    recibido: list[object] = []
    read_jpg_file("ruta.png", imread=lambda path: recibido.append(path) or np.ones((1, 1, 3)))
    assert recibido == ["ruta.png"]


def test_read_jpg_acepta_path() -> None:
    recibido: list[object] = []
    ruta = Path("ruta.png")
    read_jpg_file(ruta, imread=lambda path: recibido.append(path) or np.ones((1, 1, 3)))
    assert recibido == [ruta]


class _FakeDataset:
    def __init__(self, pixels: np.ndarray) -> None:
        self.pixel_array = pixels


def test_read_dicom_devuelve_rgb_y_gris() -> None:
    pixels = np.array([[0.0, 100.0], [200.0, 400.0]])
    rgb, gray = read_dicom_file(
        "scan.dcm",
        reader=lambda _path: _FakeDataset(pixels),
        to_rgb=lambda arr: np.stack([arr, arr, arr], axis=-1),
    )
    assert gray.shape == (2, 2)
    assert rgb.shape == (2, 2, 3)


def test_read_dicom_gris_es_uint8() -> None:
    _rgb, gray = read_dicom_file(
        "scan.dcm",
        reader=lambda _path: _FakeDataset(np.array([[0.0, 400.0]])),
        to_rgb=lambda arr: np.stack([arr, arr, arr], axis=-1),
    )
    assert gray.dtype == np.uint8


def test_read_dicom_normaliza_pixeles() -> None:
    _, gray = read_dicom_file(
        "scan.dcm",
        reader=lambda _path: _FakeDataset(np.array([[0.0, 400.0]])),
        to_rgb=lambda arr: arr,
    )
    assert np.array_equal(gray, np.array([[0, 255]], dtype=np.uint8))


def test_read_dicom_pasa_ruta_al_reader() -> None:
    recibido: list[object] = []

    def reader(path: object) -> _FakeDataset:
        recibido.append(path)
        return _FakeDataset(np.ones((2, 2)))

    read_dicom_file("scan.dcm", reader=reader, to_rgb=lambda arr: arr)
    assert recibido == ["scan.dcm"]


def test_read_dicom_usa_to_rgb_inyectado() -> None:
    calls: list[int] = []

    def to_rgb(arr: np.ndarray) -> np.ndarray:
        calls.append(1)
        return arr

    read_dicom_file("scan.dcm", reader=lambda _path: _FakeDataset(np.ones((2, 2))), to_rgb=to_rgb)
    assert len(calls) == 1


def test_read_dicom_usa_dcmread_de_pydicom3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace(
        dcmread=lambda _path: _FakeDataset(np.array([[0.0, 400.0]])),
    )
    monkeypatch.setitem(sys.modules, "pydicom", fake)
    _rgb, gray = read_dicom_file("scan.dcm", to_rgb=lambda arr: arr)
    assert np.array_equal(gray, np.array([[0, 255]], dtype=np.uint8))


def test_read_dicom_usa_read_file_en_pydicom_antiguo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace(
        read_file=lambda _path: _FakeDataset(np.array([[0.0, 400.0]])),
    )
    monkeypatch.setitem(sys.modules, "pydicom", fake)
    _rgb, gray = read_dicom_file("scan.dcm", to_rgb=lambda arr: arr)
    assert np.array_equal(gray, np.array([[0, 255]], dtype=np.uint8))


def test_resize_for_display_devuelve_tamano_por_defecto() -> None:
    result = resize_for_display(np.zeros((10, 10), dtype=np.uint8))
    assert result.shape == (250, 250)


def test_resize_for_display_acepta_tamano_personalizado() -> None:
    result = resize_for_display(np.zeros((10, 10), dtype=np.uint8), size=(32, 16))
    assert result.shape == (16, 32)
