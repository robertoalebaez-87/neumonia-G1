"""Implementación de Grad-CAM con TensorFlow (import perezoso)."""

from __future__ import annotations

import numpy as np

from .constants import GRADCAM_FILTERS, HEATMAP_ALPHA, HEATMAP_LAYER
from .gradcam import compute_heatmap, overlay_heatmap, scale_heatmap
from .images import preprocess
from .model import load_model


def grad_cam(array: np.ndarray, model_path: str | None = None) -> np.ndarray:
    """Genera el mapa de calor Grad-CAM para ``array``.

    Requiere TensorFlow y OpenCV instalados (``uv sync --extra ml``).
    """
    import cv2
    from tensorflow.keras import backend as keras_backend

    img = preprocess(array)
    model = load_model(model_path) if model_path else load_model()
    predictions = model.predict(img)
    argmax = int(np.argmax(predictions[0]))
    output = model.output[:, argmax]
    last_conv_layer = model.get_layer(HEATMAP_LAYER)
    grads = keras_backend.gradients(output, last_conv_layer.output)[0]
    pooled_grads = keras_backend.mean(grads, axis=(0, 1, 2))
    iterate = keras_backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_value, conv_value = iterate(img)
    heatmap = compute_heatmap(conv_value, np.asarray(pooled_value), GRADCAM_FILTERS)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = scale_heatmap(np.maximum(heatmap, 0) / np.max(heatmap))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array, (512, 512))
    return overlay_heatmap(heatmap, img2, HEATMAP_ALPHA)[:, :, ::-1]


__all__ = ["grad_cam"]
