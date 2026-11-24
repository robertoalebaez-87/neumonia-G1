"""Pruebas de inferencia y orquestación."""

from __future__ import annotations

import numpy as np
import pytest

from neumonia.predict import interpret_predictions, predict


@pytest.mark.parametrize(
    ("predictions", "expected_label", "expected_index"),
    [
        ([0.9, 0.05, 0.05], "bacteriana", 0),
        ([0.05, 0.9, 0.05], "normal", 1),
        ([0.05, 0.05, 0.9], "viral", 2),
    ],
)
def test_interpret_cada_clase(
    predictions: list[float], expected_label: str, expected_index: int
) -> None:
    label, _proba, index = interpret_predictions(np.array(predictions))
    assert label == expected_label
    assert index == expected_index


def test_interpret_devuelve_porcentaje() -> None:
    _label, proba, _index = interpret_predictions(np.array([0.5, 0.25, 0.25]))
    assert proba == pytest.approx(50.0)


def test_interpret_elige_el_maximo() -> None:
    _label, proba, index = interpret_predictions(np.array([0.1, 0.2, 0.3]))
    assert index == 2
    assert proba == pytest.approx(30.0)


def test_interpret_acepta_forma_batch() -> None:
    label, _proba, index = interpret_predictions(np.array([[0.1, 0.7, 0.2]]))
    assert label == "normal"
    assert index == 1


def test_interpret_rechaza_vacio() -> None:
    with pytest.raises(ValueError):
        interpret_predictions(np.array([]))


def test_interpret_rechaza_tres_dimensiones() -> None:
    with pytest.raises(ValueError):
        interpret_predictions(np.zeros((1, 1, 3)))


def test_interpret_devuelve_tipos_correctos() -> None:
    label, proba, index = interpret_predictions(np.array([0.1, 0.7, 0.2]))
    assert isinstance(label, str)
    assert isinstance(proba, float)
    assert isinstance(index, int)


def test_interpret_empate_elige_el_primero() -> None:
    _label, _proba, index = interpret_predictions(np.array([0.5, 0.5, 0.0]))
    assert index == 0


def test_interpret_probabilidad_es_el_maximo() -> None:
    _label, proba, _index = interpret_predictions(np.array([0.1, 0.7, 0.2]))
    assert proba == pytest.approx(70.0)


class _FakeModel:
    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.batch = batch
        return np.array([[0.2, 0.3, 0.5]])


def _run_predict():
    calls: dict[str, object] = {}
    model = _FakeModel()

    def preprocess_fn(array: np.ndarray) -> np.ndarray:
        calls["preprocess"] = array
        return array[None, ...]

    def model_factory() -> _FakeModel:
        calls["model"] = True
        return model

    heatmap = np.full((2, 2, 3), 7, dtype=np.uint8)
    array = np.zeros((2, 2, 3), dtype=np.uint8)
    result = predict(
        array,
        preprocess_fn=preprocess_fn,
        model_factory=model_factory,
        grad_cam_fn=lambda _array: heatmap,
    )
    return result, calls, model, array, heatmap


def test_predict_devuelve_tres_elementos() -> None:
    label, proba, heatmap = _run_predict()[0]
    assert isinstance(label, str)
    assert isinstance(proba, float)
    assert heatmap is not None


def test_predict_etiqueta_correcta() -> None:
    label, _proba, _heatmap = _run_predict()[0]
    assert label == "viral"


def test_predict_probabilidad_correcta() -> None:
    _label, proba, _heatmap = _run_predict()[0]
    assert proba == pytest.approx(50.0)


def test_predict_invoca_preprocess() -> None:
    _result, calls, _model, _array, _heatmap = _run_predict()
    assert "preprocess" in calls


def test_predict_invoca_model_factory() -> None:
    _result, calls, _model, _array, _heatmap = _run_predict()
    assert calls.get("model") is True


def test_predict_pasa_el_array_a_preprocess() -> None:
    _result, calls, _model, array, _heatmap = _run_predict()
    assert calls["preprocess"] is array


def test_predict_devuelve_el_heatmap_del_gradcam() -> None:
    _result, _calls, _model, _array, heatmap = _run_predict()
    assert _result[2] is heatmap


def test_predict_pasa_el_lote_al_modelo() -> None:
    result, _calls, model, _array, _heatmap = _run_predict()
    assert model.batch.shape == (1, 2, 2, 3)
    assert result[0] == "viral"
