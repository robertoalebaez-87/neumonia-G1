"""Traducción de índices de clase a etiquetas legibles."""

from __future__ import annotations

from .constants import CLASS_NAMES


def label_for_index(index: int) -> str:
    """Devuelve la etiqueta legible para un índice de clase.

    Args:
        index: Índice de clase devuelto por la CNN (0, 1 o 2).

    Returns:
        La etiqueta correspondiente: ``bacteriana``, ``normal`` o ``viral``.

    Raises:
        ValueError: si el índice no corresponde a ninguna clase conocida.
    """
    try:
        return CLASS_NAMES[int(index)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Índice de clase inválido: {index!r}") from exc


def is_valid_index(index: int) -> bool:
    """Indica si ``index`` es una clase conocida."""
    return index in CLASS_NAMES


def format_probability(proba: float) -> str:
    """Formatea una probabilidad (0-100) con dos decimales y el signo ``%``."""
    return f"{proba:.2f}%"


def describe_prediction(index: int) -> tuple[str, str]:
    """Devuelve ``(etiqueta, índice_normalizado)`` para un índice de clase."""
    return label_for_index(index), str(int(index))


def class_labels() -> list[str]:
    """Devuelve todas las etiquetas ordenadas por índice."""
    return [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]


__all__ = [
    "class_labels",
    "describe_prediction",
    "format_probability",
    "is_valid_index",
    "label_for_index",
]
