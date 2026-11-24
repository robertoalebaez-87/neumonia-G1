from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from ui import APP_TITLE, app

client = TestClient(app)


def _jpg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 130, 140)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _png_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (40, 90, 110)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_indice_entrega_interfaz_web() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert APP_TITLE in response.text
    assert "Analizar imagen" in response.text
    assert "Estudio cargado" in response.text
    assert "Mapa de calor" in response.text
    assert "Historial CSV" not in response.text
    assert 'fetch("/preview"' in response.text


def test_vista_previa_viene_del_servidor() -> None:
    response = client.post(
        "/preview",
        files={"file": ("radiografia.jpg", _jpg(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["preview"].startswith("data:image/png;base64,")


def test_analiza_imagen_en_modo_demo() -> None:
    response = client.post(
        "/analyze",
        files={"file": ("radiografia.jpg", _jpg(), "image/jpeg")},
        data={"patient_id": "P-1042", "demo": "true"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"]
    assert payload["heatmap"].startswith("data:image/png;base64,")
    assert payload["notice"]
    assert payload["query"]["patient_id"] == "P-1042"


def test_informe_pdf_adjunta_ambas_imagenes() -> None:
    response = client.post(
        "/report.pdf",
        json={
            "patient_id": "P-1",
            "label": "Normal",
            "probability": "91,2 %",
            "notice": "MODO DEMO — resultado ilustrativo",
            "preview": _png_url(),
            "heatmap": _png_url(),
        },
    )
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
    assert len(response.content) > 2000
