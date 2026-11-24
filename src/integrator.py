"""Orquestación preprocesar → predecir → explicar; interfaz pública exigida por la guía."""

from __future__ import annotations

import numpy as np

from grad_cam import grad_cam
from load_model import load_model
from neumonia.applogic import (
    build_csv_row,
    button_state,
    increment_report_id,
    report_paths,
    result_text,
)
from neumonia.constants import CSV_DELIMITER, CSV_FILENAME, DEFAULT_MODEL_PATH
from neumonia.demo import DEMO_NOTICE, run_demo, should_use_demo
from neumonia.images import read_dicom_file, read_jpg_file
from neumonia.predict import interpret_predictions, predict
from preprocess_img import preprocess


def analyze(array: np.ndarray, *, demo: bool = False) -> tuple[str, float, np.ndarray]:
    """Preprocesa, predice y explica una imagen; delega en el demo si se solicita."""
    if demo:
        return run_demo(array)

    def _preprocess(image: np.ndarray) -> np.ndarray:
        return preprocess(image)

    def _model_factory():
        return load_model()

    return predict(
        array,
        preprocess_fn=_preprocess,
        model_factory=_model_factory,
        grad_cam_fn=grad_cam,
    )


__all__ = [
    "CSV_DELIMITER",
    "CSV_FILENAME",
    "DEFAULT_MODEL_PATH",
    "DEMO_NOTICE",
    "analyze",
    "build_csv_row",
    "button_state",
    "increment_report_id",
    "interpret_predictions",
    "predict",
    "read_dicom_file",
    "read_jpg_file",
    "report_paths",
    "result_text",
    "run_demo",
    "should_use_demo",
]
