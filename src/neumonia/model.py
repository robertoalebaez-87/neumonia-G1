"""Carga del modelo de red neuronal convolucional."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .constants import DEFAULT_MODEL_PATH


@lru_cache(maxsize=1)
def _load_cached(path: str):
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.experimental.output_all_intermediates(True)
    keras = vars(tf)["keras"]
    return keras.models.load_model(path)


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """Carga el modelo ``.h5`` entrenado una sola vez usando TensorFlow/Keras.

    TensorFlow se importa de forma perezosa para no penalizar el arranque de la
    interfaz ni las pruebas unitarias.
    """
    return _load_cached(str(path))


def clear_model_cache() -> None:
    """Descarta el modelo en caché para permitir recargarlo."""
    _load_cached.cache_clear()


__all__ = ["clear_model_cache", "load_model"]
