"""Interfaz web local: FastAPI + HTML embebido, sin dependencias de escritorio."""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from integrator import (
    DEMO_NOTICE,
    analyze,
    build_csv_row,
    read_dicom_file,
    read_jpg_file,
    should_use_demo,
)

APP_TITLE = "Estación de lectura de neumonía"

_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --ink: #0c2733;
  --paper: #f3f7f6;
  --portal: #0e7c86;
  --oxygen: #bfe3dc;
  --demo: #e5a93b;
  --line: rgba(12, 39, 51, .16);
  --muted: #58737c;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
}
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 11px 22px;
  border-bottom: 1px solid var(--line);
}
.brand { display: flex; align-items: center; gap: 10px; font-weight: 650; }
.brand-mark {
  width: 20px;
  height: 20px;
  border-radius: 4px;
  background: var(--portal);
  position: relative;
}
.brand-mark::before,
.brand-mark::after {
  content: "";
  position: absolute;
  background: var(--paper);
  border-radius: 1px;
}
.brand-mark::before { left: 8px; top: 4px; width: 4px; height: 12px; }
.brand-mark::after { left: 4px; top: 8px; width: 12px; height: 4px; }
.brand-sub {
  font-family: Georgia, serif;
  font-style: italic;
  font-weight: 400;
  color: var(--muted);
}
.topbar-note { font-size: .84rem; color: var(--muted); }
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(280px, .65fr);
  gap: 18px;
  max-width: 1600px;
  margin: 0 auto;
  padding: 16px 22px;
}
.viewer-head h1 {
  font-family: Georgia, serif;
  font-weight: 500;
  font-size: clamp(1.35rem, 2vw, 1.9rem);
  line-height: 1.15;
  margin: 0 0 5px;
  max-width: 34ch;
}
.viewer-head p { margin: 0; color: var(--muted); max-width: 78ch; line-height: 1.45; }
.compare {
  position: relative;
  height: clamp(250px, 46vh, 410px);
  margin-top: 12px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  overflow: hidden;
  border-radius: 3px;
  background: rgba(255, 255, 255, .12);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .08);
}
.compare.drag { box-shadow: inset 0 0 0 3px var(--oxygen); }
.frame {
  position: relative;
  height: 100%;
  min-height: 0;
  margin: 0;
  display: grid;
  place-items: center;
  overflow: hidden;
  background: #06171f;
}
.frame img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.frame figcaption {
  position: absolute;
  left: 14px;
  bottom: 12px;
  color: #c7ded9;
  font-size: .78rem;
}
.empty {
  position: absolute;
  inset: 0;
  z-index: 2;
  display: grid;
  place-items: center;
  padding: 24px;
  color: #c7ded9;
  text-align: center;
  background: rgba(6, 23, 31, .72);
  border: 1px dashed rgba(199, 222, 217, .45);
  border-radius: 2px;
}
#overlay { opacity: 0; transition: opacity .3s ease; }
#overlay.ready { opacity: 1; }
.viewer-foot { color: var(--muted); font-size: .78rem; margin: 7px 0 0; }
.rail { display: flex; flex-direction: column; gap: 12px; }
.panel {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 12px 12px 12px 3px;
  padding: 14px;
}
.panel h2 {
  font-family: Georgia, serif;
  font-size: 1.05rem;
  font-weight: 500;
  margin: 0 0 8px;
}
label { display: block; font-size: .8rem; margin: 9px 0 5px; }
input[type="text"] {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  font: inherit;
}
.file-btn {
  border: 1px dashed var(--portal);
  border-radius: 7px;
  padding: 9px;
  text-align: center;
  color: var(--portal);
  cursor: pointer;
}
.switch { display: flex; align-items: center; gap: 8px; margin: 11px 0; }
button {
  border: 0;
  border-radius: 7px;
  padding: 9px 12px;
  font: inherit;
  cursor: pointer;
}
.primary { width: 100%; background: var(--portal); color: #fff; }
.primary:disabled { background: #9cb7ba; cursor: not-allowed; }
.result { border-left: 4px solid var(--portal); }
.result-label { font-family: Georgia, serif; font-size: 1.25rem; margin: 0; }
.prob { color: var(--muted); font-size: .86rem; margin: 3px 0 8px; }
.bar { height: 6px; background: var(--oxygen); border-radius: 99px; overflow: hidden; }
.bar span { display: block; height: 100%; background: var(--portal); width: 0; }
.notice {
  background: rgba(229, 169, 59, .14);
  border-left: 3px solid var(--demo);
  padding: 8px;
  font-size: .76rem;
  line-height: 1.4;
}
.actions { margin-top: 10px; }
.actions button { width: 100%; background: transparent; border: 1px solid var(--line); }
.status { color: var(--muted); font-size: .8rem; margin: 8px 0 0; }
:focus-visible { outline: 3px solid var(--demo); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { #overlay { transition: none; } }
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; padding: 14px; gap: 14px; }
  .topbar { padding: 10px 14px; }
  .compare { grid-template-columns: 1fr; height: auto; }
  .frame { height: clamp(220px, 38vh, 320px); }
}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <span class="brand-mark" aria-hidden="true"></span>
    <span>Neumonía</span>
    <span class="brand-sub">estación de lectura</span>
  </div>
  <div class="topbar-note">Uso orientativo, no sustituye un diagnóstico médico</div>
</header>
<main class="layout">
  <section class="viewer" aria-labelledby="viewer-title">
    <div class="viewer-head">
      <h1 id="viewer-title">Lee la radiografía, no el valor aislado.</h1>
      <p>Carga un estudio y compara la imagen original con el mapa de calor.
      El resultado siempre se interpreta junto al contexto clínico.</p>
    </div>
    <div class="compare" id="lightbox">
      <figure class="frame">
        <img id="preview" alt="Radiografía cargada" hidden>
        <figcaption>Estudio cargado</figcaption>
      </figure>
      <figure class="frame">
        <img id="overlay" alt="Mapa de calor sobre la radiografía" hidden>
        <figcaption>Mapa de calor</figcaption>
      </figure>
      <div class="empty" id="empty">Suelta aquí una radiografía (.jpg, .png o .dcm)</div>
    </div>
    <p class="viewer-foot">El mapa señala zonas de interés para la revisión;
    no delimita hallazgos por sí solo.</p>
  </section>
  <aside class="rail">
    <section class="panel">
      <h2>Nueva lectura</h2>
      <label for="patient">Identificador del paciente</label>
      <input id="patient" type="text" placeholder="Ej. P-1042" autocomplete="off">
      <label class="file-btn" for="file">Seleccionar estudio</label>
      <input id="file" type="file" accept=".jpg,.jpeg,.png,.dcm,application/dicom" hidden>
      <label class="switch"><input id="demo" type="checkbox" checked> Modo demostración</label>
      <button class="primary" id="run" disabled>Analizar imagen</button>
      <p class="status" id="status">Esperando estudio.</p>
    </section>
    <section class="panel result" id="result" hidden>
      <p class="result-label" id="label"></p>
      <p class="prob" id="prob"></p>
      <div class="bar"><span id="bar"></span></div>
      <p class="notice" id="notice" hidden></p>
      <div class="actions">
        <button id="pdf" type="button">Informe PDF</button>
      </div>
    </section>
  </aside>
</main>
<script>
const fileInput = document.getElementById("file");
const preview = document.getElementById("preview");
const overlay = document.getElementById("overlay");
const empty = document.getElementById("empty");
const lightbox = document.getElementById("lightbox");
const run = document.getElementById("run");
const status = document.getElementById("status");
const resultPanel = document.getElementById("result");
let currentFile = null;
let last = null;

async function setFile(file) {
  currentFile = file;
  empty.hidden = true;
  overlay.hidden = true;
  overlay.classList.remove("ready");
  run.disabled = true;
  status.textContent = "Preparando vista previa.";
  const data = new FormData();
  data.append("file", file);
  try {
    const response = await fetch("/preview", { method: "POST", body: data });
    if (!response.ok) throw new Error("No se pudo mostrar la imagen");
    const payload = await response.json();
    preview.src = payload.preview;
    preview.hidden = false;
    status.textContent = "Estudio listo: " + file.name;
  } catch (error) {
    status.textContent = error.message;
    preview.hidden = true;
    empty.hidden = false;
    currentFile = null;
    return;
  }
  run.disabled = false;
}
fileInput.addEventListener("change", event => {
  if (event.target.files[0]) setFile(event.target.files[0]);
});
["dragenter", "dragover"].forEach(type => lightbox.addEventListener(type, event => {
  event.preventDefault();
  lightbox.classList.add("drag");
}));
["dragleave", "drop"].forEach(type => lightbox.addEventListener(type, event => {
  event.preventDefault();
  lightbox.classList.remove("drag");
}));
lightbox.addEventListener("drop", event => {
  const file = event.dataTransfer.files[0];
  if (file) setFile(file);
});
run.addEventListener("click", async () => {
  if (!currentFile) return;
  run.disabled = true;
  status.textContent = "Analizando imagen.";
  const data = new FormData();
  data.append("file", currentFile);
  data.append("patient_id", document.getElementById("patient").value);
  data.append("demo", document.getElementById("demo").checked ? "true" : "false");
  try {
    const response = await fetch("/analyze", { method: "POST", body: data });
    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.detail || "No se pudo analizar la imagen");
    }
    const payload = await response.json();
    overlay.src = payload.heatmap;
    overlay.hidden = false;
    requestAnimationFrame(() => overlay.classList.add("ready"));
    document.getElementById("label").textContent = payload.label;
    document.getElementById("prob").textContent = payload.probability;
    document.getElementById("bar").style.width = payload.percentage;
    const note = document.getElementById("notice");
    note.textContent = payload.notice;
    note.hidden = !payload.notice;
    resultPanel.hidden = false;
    last = {
      patient_id: payload.query.patient_id,
      label: payload.label,
      probability: payload.probability,
      notice: payload.notice,
      preview: document.getElementById("preview").src,
      heatmap: payload.heatmap
    };
    status.textContent = "Lectura terminada.";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    run.disabled = false;
  }
});
document.getElementById("pdf").addEventListener("click", async () => {
  if (!last) return;
  status.textContent = "Generando informe.";
  try {
    const response = await fetch("/report.pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(last)
    });
    if (!response.ok) throw new Error("No se pudo generar el informe");
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "informe.pdf";
    link.click();
    URL.revokeObjectURL(url);
    status.textContent = "Informe descargado.";
  } catch (error) {
    status.textContent = error.message;
  }
});
</script>
</body>
</html>
"""


def _encode_data_url(image: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


async def _read_upload(file: UploadFile) -> np.ndarray:
    raw = await file.read()
    suffix = Path(file.filename or "").suffix.lower()
    if not suffix:
        suffix = ".dcm" if "dicom" in (file.content_type or "") else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(raw)
        temporary_path = Path(temporary.name)
    try:
        rgb, _gray = (
            read_dicom_file(temporary_path) if suffix == ".dcm" else read_jpg_file(temporary_path)
        )
    finally:
        temporary_path.unlink(missing_ok=True)
    return rgb


def _blend_data_url(image: np.ndarray, heatmap: np.ndarray) -> str:
    from PIL import Image

    base = Image.fromarray(image).convert("RGB")
    overlay = Image.fromarray(heatmap).convert("RGB").resize(base.size)
    merged = Image.blend(base, overlay, 0.55)
    buffer = io.BytesIO()
    merged.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _decode_data_url(value: str) -> bytes | None:
    if not value or "," not in value:
        return None
    return base64.b64decode(value.split(",", 1)[1])


def _draw_fit_image(
    document: Any,
    png: bytes,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    from reportlab.lib.utils import ImageReader

    reader = ImageReader(io.BytesIO(png))
    source_width, source_height = reader.getSize()
    scale = min(width / source_width, height / source_height)
    drawn_width = source_width * scale
    drawn_height = source_height * scale
    document.drawImage(
        reader,
        x + (width - drawn_width) / 2,
        y + (height - drawn_height) / 2,
        width=drawn_width,
        height=drawn_height,
    )


def _pdf_report(
    patient_id: str,
    label: str,
    probability: str,
    notice: str,
    preview: str,
    heatmap: str,
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    document = canvas.Canvas(buffer, pagesize=A4)
    page_width, height = A4
    margin = 42.0
    document.setTitle("Informe de lectura")
    document.setFont("Helvetica-Bold", 18)
    document.drawString(margin, height - 52, "Informe de lectura")
    document.setFont("Helvetica", 10)
    document.drawString(margin, height - 78, f"Paciente: {patient_id}")
    document.drawString(margin, height - 96, f"Resultado: {label}")
    document.drawString(margin, height - 114, f"Probabilidad: {probability}")
    lines = [notice[index : index + 92] for index in range(0, len(notice), 92)]
    if lines:
        document.setFillColorRGB(0.65, 0.42, 0.08)
        for offset, line in enumerate(lines[:2]):
            document.drawString(margin, height - 140 - offset * 13, line)
        document.setFillColorRGB(0, 0, 0)
    image_width = (page_width - 3 * margin) / 2
    image_height = 250.0
    image_y = height - 180 - (max(1, len(lines[:2])) - 1) * 13 - image_height
    preview_png = _decode_data_url(preview)
    heatmap_png = _decode_data_url(heatmap)
    if preview_png:
        _draw_fit_image(document, preview_png, margin, image_y, image_width, image_height)
    if heatmap_png:
        _draw_fit_image(
            document,
            heatmap_png,
            2 * margin + image_width,
            image_y,
            image_width,
            image_height,
        )
    document.setFont("Helvetica", 9)
    document.drawCentredString(margin + image_width / 2, image_y - 14, "Estudio cargado")
    document.drawCentredString(2 * margin + 1.5 * image_width, image_y - 14, "Mapa de calor")
    document.setFillColorRGB(0.35, 0.45, 0.49)
    document.drawString(margin, 26, "Uso orientativo, no sustituye un diagnóstico médico")
    document.showPage()
    document.save()
    return buffer.getvalue()


class App:
    """Aplicación web local con el mismo nombre público de la guía."""

    def __init__(self) -> None:
        self.app: FastAPI = FastAPI()
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return _HTML.replace("__TITLE__", APP_TITLE)

        @self.app.post("/preview")
        async def preview_image(
            file: Annotated[UploadFile, File()],
        ) -> JSONResponse:
            rgb = await _read_upload(file)
            return JSONResponse({"preview": _encode_data_url(rgb)})

        @self.app.post("/analyze")
        async def analyze_image(
            file: Annotated[UploadFile, File()],
            patient_id: Annotated[str, Form()] = "",
            demo: Annotated[bool, Form()] = False,
        ) -> JSONResponse:
            rgb = await _read_upload(file)
            demo_mode = bool(demo or should_use_demo())
            label, score, heatmap = analyze(rgb, demo=demo_mode)
            row = build_csv_row(patient_id.strip() or "sin-id", label, score)
            notice = DEMO_NOTICE if demo_mode else ""
            percentage = f"{max(0.0, min(100.0, float(score))):.1f}%"
            return JSONResponse(
                {
                    "label": label,
                    "probability": row[2],
                    "percentage": percentage,
                    "heatmap": _blend_data_url(rgb, heatmap),
                    "notice": notice,
                    "query": {
                        "patient_id": row[0],
                        "label": row[1],
                        "probability": row[2],
                        "notice": notice,
                    },
                }
            )

        @self.app.post("/report.pdf")
        async def report_pdf(payload: dict[str, str]) -> Response:
            document = _pdf_report(
                payload.get("patient_id", ""),
                payload.get("label", ""),
                payload.get("probability", ""),
                payload.get("notice", ""),
                payload.get("preview", ""),
                payload.get("heatmap", ""),
            )
            return Response(
                document,
                media_type="application/pdf",
                headers={"Content-Disposition": 'attachment; filename="informe.pdf"'},
            )


app = App().app


def main() -> None:
    """Arranca el servidor local en ``http://127.0.0.1:8000``."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


__all__ = ["APP_TITLE", "App", "app", "main"]
