"""Carga del modelo ``.h5``; interfaz pública exigida por la guía."""

from __future__ import annotations

from neumonia.model import clear_model_cache, load_model

__all__ = ["clear_model_cache", "load_model"]
