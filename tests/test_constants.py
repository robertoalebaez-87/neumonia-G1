"""Pruebas de las constantes del dominio."""

from __future__ import annotations

import pytest

from neumonia import constants


def test_class_names_tiene_tres_clases() -> None:
    assert len(constants.CLASS_NAMES) == 3


def test_class_names_usa_indices_cero_uno_dos() -> None:
    assert sorted(constants.CLASS_NAMES) == [0, 1, 2]


def test_num_classes_coincide_con_el_diccionario() -> None:
    assert len(constants.CLASS_NAMES) == constants.NUM_CLASSES


@pytest.mark.parametrize("label", ["bacteriana", "normal", "viral"])
def test_labels_esperados_estan_presentes(label: str) -> None:
    assert label in constants.CLASS_NAMES.values()


def test_img_size_es_512() -> None:
    assert constants.IMG_SIZE == 512


def test_gradcam_filters_es_64() -> None:
    assert constants.GRADCAM_FILTERS == 64


def test_heatmap_alpha_es_fraccion_valida() -> None:
    assert 0.0 <= constants.HEATMAP_ALPHA <= 1.0


def test_csv_delimiter_es_un_caracter() -> None:
    assert constants.CSV_DELIMITER == "-"


def test_csv_filename_tiene_extension_csv() -> None:
    assert constants.CSV_FILENAME.endswith(".csv")


def test_default_model_path_tiene_extension_h5() -> None:
    assert constants.DEFAULT_MODEL_PATH.endswith(".h5")


def test_heatmap_layer_no_esta_vacio() -> None:
    assert constants.HEATMAP_LAYER.strip()


def test_clahe_clip_limit_es_positivo() -> None:
    assert constants.CLAHE_CLIP_LIMIT > 0


def test_clahe_tile_grid_es_par_de_enteros() -> None:
    assert len(constants.CLAHE_TILE_GRID) == 2
    assert all(isinstance(value, int) for value in constants.CLAHE_TILE_GRID)


def test_labels_son_unicos() -> None:
    values = list(constants.CLASS_NAMES.values())
    assert len(values) == len(set(values))
