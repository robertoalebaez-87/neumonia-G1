"""Pruebas de la lógica de interfaz independiente del framework de UI."""

from __future__ import annotations

import pytest

from neumonia.applogic import (
    build_csv_row,
    button_state,
    increment_report_id,
    report_paths,
    result_text,
    validate_patient_id,
)


def test_build_csv_row_tiene_tres_campos() -> None:
    assert len(build_csv_row("123", "normal", 95.5)) == 3


def test_build_csv_row_formatea_probabilidad() -> None:
    assert build_csv_row("123", "normal", 95.5)[2] == "95.50%"


def test_build_csv_row_conserva_la_etiqueta() -> None:
    assert build_csv_row("123", "viral", 10.0)[1] == "viral"


def test_build_csv_row_convierte_id_a_texto() -> None:
    assert build_csv_row(456, "normal", 10.0)[0] == "456"


def test_build_csv_row_acepta_id_vacio() -> None:
    assert build_csv_row("", "normal", 10.0)[0] == ""


def test_build_csv_row_conserva_orden() -> None:
    row = build_csv_row("id", "label", 1.0)
    assert row[0] == "id"
    assert row[1] == "label"
    assert row[2] == "1.00%"


def test_report_paths_devuelve_tupla_de_dos() -> None:
    assert len(report_paths(0)) == 2


def test_report_paths_jpg() -> None:
    assert report_paths(3)[0] == "Reporte3.jpg"


def test_report_paths_pdf() -> None:
    assert report_paths(3)[1] == "Reporte3.pdf"


def test_report_paths_convierte_a_entero() -> None:
    assert report_paths("7")[0] == "Reporte7.jpg"  # type: ignore[arg-type]


@pytest.mark.parametrize("report_id", [0, 1, 10, 999])
def test_report_paths_respeta_el_id(report_id: int) -> None:
    jpg, pdf = report_paths(report_id)
    assert jpg == f"Reporte{report_id}.jpg"
    assert pdf == f"Reporte{report_id}.pdf"


def test_increment_report_id_suma_uno() -> None:
    assert increment_report_id(0) == 1


def test_increment_report_id_avanza() -> None:
    assert increment_report_id(41) == 42


def test_increment_report_id_negativo() -> None:
    assert increment_report_id(-1) == 0


def test_increment_report_id_devuelve_entero() -> None:
    assert isinstance(increment_report_id(1), int)


@pytest.mark.parametrize("report_id", [0, 5, 100])
def test_increment_report_id_es_secuencial(report_id: int) -> None:
    assert increment_report_id(report_id) == report_id + 1


def test_button_state_con_imagen() -> None:
    assert button_state(True) == "enabled"


def test_button_state_sin_imagen() -> None:
    assert button_state(False) == "disabled"


def test_button_state_valor_verdadero() -> None:
    assert button_state(1) == "enabled"  # type: ignore[arg-type]


def test_button_state_valor_falso() -> None:
    assert button_state(0) == "disabled"  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["123", "abc", "0", " 7 "])
def test_validate_patient_id_acepta_texto(value: str) -> None:
    assert validate_patient_id(value) is True


@pytest.mark.parametrize("value", ["", "   ", None])
def test_validate_patient_id_rechaza_vacio(value: str | None) -> None:
    assert validate_patient_id(value) is False


def test_result_text_devuelve_tupla() -> None:
    assert len(result_text("normal", 88.0)) == 2


def test_result_text_formatea_probabilidad() -> None:
    assert result_text("normal", 88.0)[1] == "88.00%"


def test_result_text_conserva_etiqueta() -> None:
    assert result_text("bacteriana", 10.0)[0] == "bacteriana"


def test_result_text_convierte_etiqueta_a_texto() -> None:
    assert result_text(123, 10.0)[0] == "123"
