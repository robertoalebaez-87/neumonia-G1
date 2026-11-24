"""Inferencia y orquestación de la predicción."""

from __future__ import annotations

import numpy as np

from .labels import label_for_index


def interpret_predictions(predictions: np.ndarray) -> tuple[str, float, int]:
    """Interpreta la salida de la CNN.

    Args:
        predictions: Arreglo con probabilidades por clase; acepta forma
            ``(N,)`` o ``(1, N)``.

    Returns:
        ``(etiqueta, probabilidad_porcentual, índice)``.
    """
    data = np.asarray(predictions, dtype=float)
    if data.ndim == 2:
        data = data[0]
    if data.ndim != 1 or data.size == 0:
        raise ValueError("predictions debe ser un vector no vacío")
    index = int(np.argmax(data))
    probability = float(np.max(data)) * 100
    return label_for_index(index), probability, index


def predict(
    array: np.ndarray,
    *,
    preprocess_fn,
    model_factory,
    grad_cam_fn,
) -> tuple[str, float, np.ndarray]:
    """Ejecuta el flujo completo: preprocesar, predecir y generar Grad-CAM.

    Los colaboradores se inyectan para poder probar sin TensorFlow.
    """
    batch = preprocess_fn(array)
    model = model_factory()
    predictions = model.predict(batch)
    label, probability, _index = interpret_predictions(predictions)
    heatmap = grad_cam_fn(array)
    return label, probability, heatmap


__all__ = ["interpret_predictions", "predict"]
