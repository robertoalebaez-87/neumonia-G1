"""Constantes compartidas por la aplicación de neumonía."""

from __future__ import annotations

from typing import Final

CLASS_NAMES: Final[dict[int, str]] = {
    0: "bacteriana",
    1: "normal",
    2: "viral",
}

NUM_CLASSES: Final[int] = len(CLASS_NAMES)
IMG_SIZE: Final[int] = 512
GRADCAM_FILTERS: Final[int] = 64
HEATMAP_ALPHA: Final[float] = 0.8
HEATMAP_LAYER: Final[str] = "conv10_thisone"
DEFAULT_MODEL_PATH: Final[str] = "WilhemNet86.h5"
CSV_DELIMITER: Final[str] = "-"
CSV_FILENAME: Final[str] = "historial.csv"
CLAHE_CLIP_LIMIT: Final[float] = 2.0
CLAHE_TILE_GRID: Final[tuple[int, int]] = (4, 4)

__all__ = [
    "CLAHE_CLIP_LIMIT",
    "CLAHE_TILE_GRID",
    "CLASS_NAMES",
    "CSV_DELIMITER",
    "CSV_FILENAME",
    "DEFAULT_MODEL_PATH",
    "GRADCAM_FILTERS",
    "HEATMAP_ALPHA",
    "HEATMAP_LAYER",
    "IMG_SIZE",
    "NUM_CLASSES",
]
