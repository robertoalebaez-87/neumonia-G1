"""Pruebas de integridad del paquete."""

from __future__ import annotations

import neumonia


def test_version_es_string() -> None:
    assert isinstance(neumonia.__version__, str)


def test_version_tiene_formato_semver() -> None:
    parts = neumonia.__version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts[:2])


def test_all_expone_version() -> None:
    assert "__version__" in neumonia.__all__


def test_import_constants() -> None:
    from neumonia import constants

    assert constants.NUM_CLASSES == 3


def test_import_labels() -> None:
    from neumonia.labels import label_for_index

    assert label_for_index(0) == "bacteriana"


def test_import_images() -> None:
    from neumonia.images import normalize_to_uint8

    assert callable(normalize_to_uint8)


def test_import_gradcam() -> None:
    from neumonia.gradcam import compute_heatmap

    assert callable(compute_heatmap)


def test_import_predict() -> None:
    from neumonia.predict import interpret_predictions

    assert callable(interpret_predictions)


def test_import_applogic() -> None:
    from neumonia.applogic import build_csv_row

    assert callable(build_csv_row)
