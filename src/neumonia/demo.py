"""Modo demostración: resultados ilustrativos sin modelo entrenado."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import DEFAULT_MODEL_PATH, HEATMAP_ALPHA, NUM_CLASSES
from .gradcam import normalize_heatmap, overlay_heatmap
from .images import normalize_to_uint8
from .predict import interpret_predictions

DEMO_NOTICE = "MODO DEMO — resultado ilustrativo, no válido para diagnóstico"
_TOP_PROBABILITY = 0.7
_OTHER_PROBABILITY = (1.0 - _TOP_PROBABILITY) / (NUM_CLASSES - 1)


def should_use_demo(model_path: str | Path = DEFAULT_MODEL_PATH, *, force: bool = False) -> bool:
    """Indica si corresponde usar el modo demo en lugar del modelo real."""
    return force or not Path(model_path).is_file()


def demo_predictions(array: np.ndarray) -> np.ndarray:
    """Devuelve probabilidades deterministas e ilustrativas según el brillo."""
    data = np.asarray(array)
    if data.size == 0:
        raise ValueError("array debe contener al menos un píxel")
    brightness = min(max(float(np.mean(data)) / 255.0, 0.0), 1.0)
    index = min(int(brightness * NUM_CLASSES), NUM_CLASSES - 1)
    probabilities = np.full(NUM_CLASSES, _OTHER_PROBABILITY)
    probabilities[index] = _TOP_PROBABILITY
    return probabilities


def _as_rgb(array: np.ndarray) -> np.ndarray:
    data = normalize_to_uint8(array)
    if data.ndim == 2:
        return np.stack([data, data, data], axis=-1)
    if data.ndim != 3:
        raise ValueError("array debe tener forma (H, W) o (H, W, C)")
    if data.shape[-1] == 1:
        return np.repeat(data, 3, axis=-1)
    if data.shape[-1] >= 3:
        return data[..., :3]
    raise ValueError("array debe tener al menos un canal")


def demo_heatmap(array: np.ndarray) -> np.ndarray:
    """Genera un mapa de calor sintético superpuesto sobre ``array``."""
    base = _as_rgb(array)
    height, width = base.shape[:2]
    y = np.linspace(-1.0, 1.0, height)[:, None]
    x = np.linspace(-1.0, 1.0, width)[None, :]
    focus = np.exp(-((x - 0.15) ** 2 + (y + 0.10) ** 2) / 0.10)
    secondary = 0.6 * np.exp(-((x + 0.40) ** 2 + (y - 0.35) ** 2) / 0.05)
    heat = normalize_heatmap(focus + secondary)
    colored = np.stack([255 * heat, 70 * (1 - heat), 255 * (1 - heat)], axis=-1)
    return overlay_heatmap(colored, base, alpha=HEATMAP_ALPHA)


def run_demo(array: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Ejecuta el flujo de demostración sin TensorFlow ni OpenCV."""
    label, probability, _index = interpret_predictions(demo_predictions(array))
    return label, probability, demo_heatmap(array)


__all__ = ["DEMO_NOTICE", "demo_heatmap", "demo_predictions", "run_demo", "should_use_demo"]
