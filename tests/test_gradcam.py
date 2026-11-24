"""Pruebas de la matemática de Grad-CAM."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from neumonia.gradcam import (
    compute_gradcam,
    compute_heatmap,
    normalize_heatmap,
    overlay_heatmap,
    scale_heatmap,
)


def test_compute_heatmap_forma_bidimendional() -> None:
    conv = np.ones((2, 2, 1))
    heatmap = compute_heatmap(conv, np.array([1.0]))
    assert heatmap.shape == (2, 2)


def test_compute_heatmap_es_no_negativo() -> None:
    conv = -np.ones((2, 2, 1))
    heatmap = compute_heatmap(conv, np.array([1.0]))
    assert np.all(heatmap >= 0)


def test_compute_heatmap_un_canal_uno() -> None:
    conv = np.ones((2, 2, 1))
    heatmap = compute_heatmap(conv, np.array([1.0]))
    assert np.allclose(heatmap, 1.0)


def test_compute_heatmap_promedia_canales() -> None:
    conv = np.zeros((1, 1, 2))
    conv[..., 0] = 2.0
    conv[..., 1] = 4.0
    heatmap = compute_heatmap(conv, np.array([0.5, 0.5]))
    assert np.allclose(heatmap, 1.5)


def test_compute_heatmap_limita_filtros() -> None:
    conv = np.zeros((1, 1, 3))
    conv[..., 0] = 1.0
    conv[..., 1] = 10.0
    conv[..., 2] = 100.0
    heatmap = compute_heatmap(conv, np.ones(3), filters=1)
    assert np.allclose(heatmap, 1.0)


def test_compute_heatmap_rechaza_conv_bidimensional() -> None:
    with pytest.raises(ValueError):
        compute_heatmap(np.ones((2, 2)), np.array([1.0]))


def test_compute_heatmap_rechaza_grads_escalar() -> None:
    with pytest.raises(ValueError):
        compute_heatmap(np.ones((2, 2, 1)), np.array(1.0))


def test_compute_heatmap_devuelve_float() -> None:
    heatmap = compute_heatmap(np.ones((2, 2, 1)), np.array([1]))
    assert np.issubdtype(heatmap.dtype, np.floating)


def test_compute_heatmap_salida_negativa_es_cero() -> None:
    conv = -np.ones((2, 2, 1))
    heatmap = compute_heatmap(conv, np.array([1.0]))
    assert np.array_equal(heatmap, np.zeros((2, 2)))


def test_compute_heatmap_cero_es_cero() -> None:
    heatmap = compute_heatmap(np.zeros((2, 2, 1)), np.array([1.0]))
    assert np.allclose(heatmap, 0.0)


def test_normalize_heatmap_maximo_es_uno() -> None:
    result = normalize_heatmap(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert result.max() == pytest.approx(1.0)


def test_normalize_heatmap_valores_en_rango() -> None:
    result = normalize_heatmap(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.all((result >= 0) & (result <= 1))


def test_normalize_heatmap_ceros() -> None:
    result = normalize_heatmap(np.zeros((2, 2)))
    assert np.array_equal(result, np.zeros((2, 2)))


def test_normalize_heatmap_todo_negativo_es_cero() -> None:
    result = normalize_heatmap(-np.ones((2, 2)))
    assert np.array_equal(result, np.zeros((2, 2)))


def test_normalize_heatmap_no_modifica_entrada() -> None:
    data = np.array([[1.0, 2.0]])
    original = data.copy()
    normalize_heatmap(data)
    assert np.array_equal(data, original)


def test_normalize_heatmap_conserva_forma() -> None:
    result = normalize_heatmap(np.arange(6, dtype=float).reshape(2, 3))
    assert result.shape == (2, 3)


def test_normalize_heatmap_es_proporcional() -> None:
    result = normalize_heatmap(np.array([2.0, 4.0, 8.0]))
    assert np.allclose(result, [0.25, 0.5, 1.0])


def test_scale_heatmap_dtype_uint8() -> None:
    result = scale_heatmap(np.array([0.0, 0.5, 1.0]))
    assert result.dtype == np.uint8


def test_scale_heatmap_mapea_cero() -> None:
    assert int(scale_heatmap(np.array([0.0]))[0]) == 0


def test_scale_heatmap_mapea_medio() -> None:
    assert int(scale_heatmap(np.array([0.5]))[0]) == 127


def test_scale_heatmap_mapea_uno() -> None:
    assert int(scale_heatmap(np.array([1.0]))[0]) == 255


def test_overlay_devuelve_uint8() -> None:
    result = overlay_heatmap(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)))
    assert result.dtype == np.uint8


def test_overlay_recorta_a_255() -> None:
    result = overlay_heatmap(np.full((1, 1, 3), 255.0), np.full((1, 1, 3), 100.0))
    assert np.all(result == 255)


def test_overlay_alpha_cero_devuelve_base() -> None:
    base = np.full((1, 1, 3), 10, dtype=np.uint8)
    result = overlay_heatmap(np.full((1, 1, 3), 255.0), base, alpha=0.0)
    assert np.array_equal(result, base)


def test_overlay_alpha_uno_suma() -> None:
    base = np.full((1, 1, 3), 10, dtype=np.uint8)
    result = overlay_heatmap(np.full((1, 1, 3), 20.0), base, alpha=1.0)
    assert np.all(result == 30)


def test_overlay_conserva_forma() -> None:
    result = overlay_heatmap(np.zeros((4, 5, 3)), np.zeros((4, 5, 3)))
    assert result.shape == (4, 5, 3)


def test_overlay_base_negativa_recortada() -> None:
    result = overlay_heatmap(np.zeros((1, 1, 3)), np.full((1, 1, 3), -5.0))
    assert np.all(result == 0)


class _FakeLayer:
    def __init__(self) -> None:
        self.output = np.zeros((4, 4, 8))


class _FakeModel:
    def __init__(self, layers: dict[str, _FakeLayer]) -> None:
        self.output = np.zeros((1, 3))
        self._layers = layers
        self.requested: list[str] = []

    def predict(self, _batch: np.ndarray) -> np.ndarray:
        return np.array([[0.1, 0.7, 0.2]])

    def get_layer(self, name: str) -> _FakeLayer:
        self.requested.append(name)
        return self._layers[name]


def _build_gradcam(**overrides):
    calls: dict[str, object] = {}
    layer = _FakeLayer()
    model = _FakeModel({overrides.pop("layer_name", "conv10_thisone"): layer})

    def preprocess_fn(array: np.ndarray) -> np.ndarray:
        calls["preprocess"] = array
        return np.zeros((1, 4, 4, 1))

    def model_factory() -> _FakeModel:
        calls["model"] = True
        return model

    def gradient_fn(_output, _layer_output):
        return [np.ones((1, 4, 4, 8))]

    def iterate_factory(_model, _layer):
        def iterate(_batch):
            return np.ones(8), np.ones((4, 4, 8))

        return iterate

    params: dict[str, Any] = {
        "preprocess_fn": preprocess_fn,
        "model_factory": model_factory,
        "gradient_fn": gradient_fn,
        "iterate_factory": iterate_factory,
        "layer_name": "conv10_thisone",
    }
    params.update(overrides)
    array = np.full((4, 4, 3), 10, dtype=np.uint8)
    return compute_gradcam(array, **params), calls, model


def test_compute_gradcam_devuelve_tres_canales() -> None:
    result, _calls, _model = _build_gradcam()
    assert result.shape == (4, 4, 3)


def test_compute_gradcam_devuelve_uint8() -> None:
    result, _calls, _model = _build_gradcam()
    assert result.dtype == np.uint8


def test_compute_gradcam_invoca_preprocess() -> None:
    _result, calls, _model = _build_gradcam()
    assert "preprocess" in calls


def test_compute_gradcam_invoca_model_factory() -> None:
    _result, calls, _model = _build_gradcam()
    assert calls.get("model") is True


def test_compute_gradcam_solicita_la_capa_indicada() -> None:
    _result, _calls, model = _build_gradcam(layer_name="conv10_thisone")
    assert model.requested == ["conv10_thisone"]


def test_compute_gradcam_alpha_cero_usa_base() -> None:
    result, _calls, _model = _build_gradcam(alpha=0.0)
    assert np.all(result == 10)


def test_compute_gradcam_usa_colormap_inyectado() -> None:
    def colormap(heatmap: np.ndarray) -> np.ndarray:
        return np.stack([heatmap, heatmap, heatmap], axis=-1)

    result, _calls, _model = _build_gradcam(colormap_fn=colormap)
    assert result.shape == (4, 4, 3)


def test_compute_gradcam_usa_resize_inyectado() -> None:
    calls: list[tuple[int, ...]] = []

    def resize(arr: np.ndarray, _size: tuple[int, int]) -> np.ndarray:
        calls.append(arr.shape)
        return arr

    _result, _calls, _model = _build_gradcam(resize_fn=resize, target_size=(4, 4))
    assert len(calls) == 2
