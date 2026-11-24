"""Pruebas de la traducción de clases y formato de probabilidad."""

from __future__ import annotations

from typing import cast

import pytest

from neumonia.labels import (
    class_labels,
    describe_prediction,
    format_probability,
    is_valid_index,
    label_for_index,
)


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, "bacteriana"), (1, "normal"), (2, "viral")],
)
def test_label_for_index_traduce_cada_clase(index: int, expected: str) -> None:
    assert label_for_index(index) == expected


@pytest.mark.parametrize("index", [3, -1, 99])
def test_label_for_index_rechaza_indices_fuera_de_rango(index: int) -> None:
    with pytest.raises(ValueError):
        label_for_index(index)


@pytest.mark.parametrize("index", ["x", None, [1]])
def test_label_for_index_rechaza_tipos_invalidos(index: object) -> None:
    with pytest.raises(ValueError):
        label_for_index(cast(int, index))


def test_label_for_index_acepta_string_numerico() -> None:
    assert label_for_index(cast(int, "2")) == "viral"


def test_label_for_index_acepta_float_entero() -> None:
    assert label_for_index(cast(int, 2.0)) == "viral"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_is_valid_index_acepta_clases_conocidas(index: int) -> None:
    assert is_valid_index(index) is True


@pytest.mark.parametrize("index", [3, -1])
def test_is_valid_index_rechaza_desconocidas(index: int) -> None:
    assert is_valid_index(index) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0.00%"),
        (12.345, "12.35%"),
        (12.344, "12.34%"),
        (100.0, "100.00%"),
        (99.999, "100.00%"),
        (-1.0, "-1.00%"),
    ],
)
def test_format_probability_dos_decimales(value: float, expected: str) -> None:
    assert format_probability(value) == expected


def test_class_labels_devuelve_orden_por_indice() -> None:
    assert class_labels() == ["bacteriana", "normal", "viral"]


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, ("bacteriana", "0")), (1, ("normal", "1")), (2, ("viral", "2"))],
)
def test_describe_prediction_devuelve_etiqueta_e_indice(
    index: int, expected: tuple[str, str]
) -> None:
    assert describe_prediction(index) == expected
