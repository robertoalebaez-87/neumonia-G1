"""Lógica de la interfaz independiente del framework de UI (testeable)."""

from __future__ import annotations

from .labels import format_probability


def build_csv_row(patient_id: str | int, label: str, proba: float) -> list[str]:
    """Construye la fila del historial CSV para un paciente."""
    return [str(patient_id), str(label), format_probability(proba)]


def report_paths(report_id: int | str) -> tuple[str, str]:
    """Devuelve ``(jpg, pdf)`` para un identificador de reporte."""
    return f"Reporte{int(report_id)}.jpg", f"Reporte{int(report_id)}.pdf"


def increment_report_id(current: int) -> int:
    """Avanza el contador de reportes."""
    return int(current) + 1


def button_state(has_image: object) -> str:
    """Estado del botón ``Predecir`` según haya o no imagen cargada."""
    return "enabled" if has_image else "disabled"


def validate_patient_id(value: str | None) -> bool:
    """Indica si la cédula del paciente es válida (no vacía)."""
    return bool(str(value or "").strip())


def result_text(label: object, proba: float) -> tuple[str, str]:
    """Devuelve el texto de resultado y probabilidad para la interfaz."""
    return str(label), format_probability(proba)


__all__ = [
    "build_csv_row",
    "button_state",
    "increment_report_id",
    "report_paths",
    "result_text",
    "validate_patient_id",
]
