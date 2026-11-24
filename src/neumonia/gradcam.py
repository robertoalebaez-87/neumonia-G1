"""Matemática de Grad-CAM independiente del framework de Deep Learning."""

from __future__ import annotations

import numpy as np

from .constants import GRADCAM_FILTERS, HEATMAP_ALPHA


def compute_heatmap(
    conv_output: np.ndarray,
    pooled_grads: np.ndarray,
    filters: int = GRADCAM_FILTERS,
) -> np.ndarray:
    """Combina las activaciones con los gradientes medios para obtener el mapa.

    Equivale a multiplicar cada filtro por su gradiente medio y promediar a lo
    largo del eje de canales, seguido de una ReLU.
    """
    output = np.asarray(conv_output, dtype=float)
    grads = np.asarray(pooled_grads, dtype=float)
    if output.ndim != 3:
        raise ValueError("conv_output debe tener forma (H, W, C)")
    if grads.ndim != 1:
        raise ValueError("pooled_grads debe ser un vector 1-D")
    channels = min(filters, output.shape[-1], grads.shape[0])
    weighted = output[..., :channels] * grads[:channels]
    return np.maximum(np.mean(weighted, axis=-1), 0.0)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normaliza un mapa de calor al rango ``[0, 1]``."""
    data = np.asarray(heatmap, dtype=float)
    data = np.maximum(data, 0.0)
    peak = float(data.max()) if data.size else 0.0
    if peak == 0.0:
        return np.zeros_like(data)
    return data / peak


def scale_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Convierte un mapa normalizado ``[0, 1]`` a ``uint8`` ``[0, 255]``."""
    return (255 * np.asarray(heatmap, dtype=float)).astype(np.uint8)


def overlay_heatmap(
    heatmap_rgb: np.ndarray,
    image: np.ndarray,
    alpha: float = HEATMAP_ALPHA,
) -> np.ndarray:
    """Superpone el mapa de calor de color sobre la imagen original."""
    colored = np.asarray(heatmap_rgb, dtype=float) * alpha
    base = np.asarray(image, dtype=float)
    return np.clip(colored + base, 0, 255).astype(np.uint8)


def compute_gradcam(
    array: np.ndarray,
    *,
    preprocess_fn,
    model_factory,
    gradient_fn,
    iterate_factory,
    layer_name: str,
    filters: int = GRADCAM_FILTERS,
    resize_fn=None,
    colormap_fn=None,
    target_size: tuple[int, int] | None = None,
    alpha: float = HEATMAP_ALPHA,
) -> np.ndarray:
    """Orquesta el cálculo completo de Grad-CAM.

    Los colaboradores pesados se inyectan para poder probar la orquestación
    sin cargar TensorFlow ni OpenCV.

    ``iterate_factory(model, layer)`` debe devolver una función que reciba el
    lote y retorne ``(pooled_grads, conv_output)``.
    """
    batch = preprocess_fn(array)
    model = model_factory()
    predictions = model.predict(batch)
    argmax = int(np.argmax(predictions[0]))
    layer = model.get_layer(layer_name)
    grads = gradient_fn(model.output[:, argmax], layer.output)[0]
    pooled_grads = np.mean(grads, axis=(0, 1, 2))
    iterate = iterate_factory(model, layer)
    _pooled_value, conv_value = iterate(batch)
    heatmap = compute_heatmap(conv_value, np.asarray(pooled_grads), filters)
    heatmap = normalize_heatmap(heatmap)
    if resize_fn is not None and target_size is not None:
        heatmap = resize_fn(heatmap, target_size)
    heatmap = scale_heatmap(heatmap)
    if colormap_fn is not None:
        heatmap = colormap_fn(heatmap)
    elif heatmap.ndim == 2:
        heatmap = np.stack([heatmap, heatmap, heatmap], axis=-1)
    base = resize_fn(array, target_size) if resize_fn and target_size else array
    return overlay_heatmap(heatmap, base, alpha)[:, :, ::-1]


__all__ = [
    "compute_gradcam",
    "compute_heatmap",
    "normalize_heatmap",
    "overlay_heatmap",
    "scale_heatmap",
]
