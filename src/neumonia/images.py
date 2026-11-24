"""Lectura y preprocesamiento de imágenes radiográficas.

Las dependencias pesadas (OpenCV, Pillow, pydicom) se importan de forma
perezosa dentro de las funciones que las necesitan. Así el núcleo de la lógica
permanece testeable con solo NumPy, inyectando dobles en las pruebas.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from .constants import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, IMG_SIZE

Resize = Callable[[np.ndarray, tuple[int, int]], np.ndarray]
ToGray = Callable[[np.ndarray], np.ndarray]
ClaheApply = Callable[[np.ndarray], np.ndarray]
ImRead = Callable[[str | Path], np.ndarray | None]
GrayToRgb = Callable[[np.ndarray], np.ndarray]


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normaliza un arreglo a ``uint8`` escalando el máximo a 255.

    Los valores negativos se recortan a cero. Si el arreglo está vacío o su
    máximo es cero, devuelve una imagen negra de la misma forma.
    """
    data = np.asarray(array, dtype=float)
    clipped = np.maximum(data, 0)
    peak = float(clipped.max()) if clipped.size else 0.0
    if peak == 0.0:
        return np.zeros(data.shape, dtype=np.uint8)
    return (clipped / peak * 255.0).astype(np.uint8)


def _default_resize(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(array, size)


def _default_to_gray(array: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)


def _default_clahe_apply(array: np.ndarray) -> np.ndarray:
    import cv2

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(array)


def _default_imread(path: str | Path) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    return cv2.imread(str(path))


def _default_gray_to_rgb(array: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)


def preprocess(
    array: np.ndarray,
    *,
    resize: Resize | None = None,
    to_gray: ToGray | None = None,
    clahe_apply: ClaheApply | None = None,
    size: tuple[int, int] = (IMG_SIZE, IMG_SIZE),
) -> np.ndarray:
    """Prepara una imagen para la CNN.

    Redimensiona a ``size``, convierte a escala de grises, aplica CLAHE,
    normaliza a ``[0, 1]`` y devuelve un tensor de forma ``(1, H, W, 1)``.
    """
    resize = resize or _default_resize
    to_gray = to_gray or _default_to_gray
    clahe_apply = clahe_apply or _default_clahe_apply

    out = resize(np.asarray(array), size)
    out = to_gray(out)
    out = clahe_apply(out)
    out = out / 255
    out = np.expand_dims(out, axis=-1)
    return np.expand_dims(out, axis=0)


def read_jpg_file(
    path: str | Path,
    *,
    imread: ImRead | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lee una imagen JPG/PNG.

    Returns:
        ``(normalizado, original)``: el arreglo ``uint8`` normalizado y el
        arreglo original tal como lo devuelve ``imread``.
    """
    if imread is None:
        imread = _default_imread

    img = imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    array = np.asarray(img)
    return normalize_to_uint8(array), array


def read_dicom_file(
    path: str | Path,
    *,
    reader: Callable[[str | Path], object] | None = None,
    to_rgb: GrayToRgb | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lee una imagen DICOM y la deja lista para mostrar y predecir.

    Returns:
        ``(rgb, gris_normalizado)``.
    """
    if reader is None:
        import pydicom

        reader = getattr(pydicom, "dcmread", None)
        if reader is None:
            reader = vars(pydicom)["read_file"]

    dataset = reader(path)
    pixel_array = np.asarray(cast(Any, dataset).pixel_array)
    gray = normalize_to_uint8(pixel_array)
    rgb = (to_rgb or _default_gray_to_rgb)(gray)
    return rgb, gray


def resize_for_display(image: np.ndarray, size: tuple[int, int] = (250, 250)) -> np.ndarray:
    """Redimensiona un arreglo de imagen usando Pillow (import perezoso)."""
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize(size))


__all__ = [
    "normalize_to_uint8",
    "preprocess",
    "read_dicom_file",
    "read_jpg_file",
    "resize_for_display",
]
