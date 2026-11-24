"""Generación del mapa de calor; interfaz pública exigida por la guía."""

from __future__ import annotations

from neumonia.gradcam import compute_gradcam
from neumonia.tf_backend import grad_cam

__all__ = ["compute_gradcam", "grad_cam"]
