# AGENTS.md — Contexto para agentes LLM

Este repositorio sigue el **estándar de proyectos de Jan Polanco Velasco**. La
fuente única de verdad es [`CONSTITUTION.md`](CONSTITUTION.md). Léela antes de
cambiar código.

## Stack

- **Python 3.13** gestionado con **uv** (`pyproject.toml` + `uv.lock`).
- **src-layout**: el paquete vive en `src/neumonia/`.
- **Ruff** (lint + formato), **pytest**, **ty**, **pre-commit**.
- Dependencias de ML opcionales en el extra `ml` (TensorFlow, OpenCV, pydicom).

## Flujo de trabajo

```bash
uv sync --extra ml # o: make install
make check         # lint + format-check + typecheck + test
make run           # inicia la UI web en http://127.0.0.1:8000
```

- Tras editar los módulos sueltos de `src/*.py`, ejecuta
  `uv sync --extra ml --reinstall-package neumonia` para reflejarlos en el entorno.
- No uses `pip` ni `requirements.txt`; usa `uv add`.
- Commits en **Conventional Commits** (`feat:`, `fix:`, `test:`, `docs:`...).
- Actualiza `CONSTITUTION.md` en el mismo commit si cambias arquitectura o proceso.
- Nunca subas el modelo `*.h5`, datasets ni secretos.

## Módulos públicos de la guía

`src/` expone los seis nombres exigidos por la guía; reexportan `neumonia.*`:

| Archivo | API |
|---------|-----|
| `src/load_model.py` | `load_model`, `clear_model_cache` |
| `src/read_img.py` | `read_jpg_file`, `read_dicom_file` |
| `src/preprocess_img.py` | `preprocess`, `normalize_to_uint8` |
| `src/grad_cam.py` | `grad_cam`, `compute_gradcam` |
| `src/integrator.py` | `analyze`, `predict`, `interpret_predictions`, `run_demo` |
| `src/ui.py` | `App`, `main`, `APP_TITLE` (UI web FastAPI) |

Regla de dependencia: `ui.py` solo importa módulos internos de `integrator`;
nunca importa TensorFlow ni `load_model`. `integrator.analyze(...)` concentra
preprocesar → predecir → explicar.

## Mapa del código

| Módulo | Responsabilidad |
|--------|-----------------|
| `src/neumonia/constants.py` | Constantes del dominio (clases, tamaños, rutas). |
| `src/neumonia/labels.py` | Traducción de índices a etiquetas y formato. |
| `src/neumonia/images.py` | Lectura DICOM/JPG y preprocesamiento (import perezoso). |
| `src/neumonia/gradcam.py` | Matemática pura de Grad-CAM. |
| `src/neumonia/predict.py` | Interpretación y orquestación de la predicción. |
| `src/neumonia/applogic.py` | Lógica de interfaz independiente del framework de UI. |
| `src/neumonia/demo.py` | Modo demo ilustrativo sin modelo ni dependencias `ml`. |
| `src/neumonia/model.py` / `tf_backend.py` | Carga del modelo con caché y Grad-CAM con TensorFlow. |
| `src/neumonia/app.py` | Compatibilidad de la UI (reexporta `ui.App`). |
| `detector_neumonia.py` | Lanzador histórico. |

## Búsqueda de código (tgrep)

Usa `tgrep` en lugar de `grep` para búsquedas semánticas cuando esté disponible.
