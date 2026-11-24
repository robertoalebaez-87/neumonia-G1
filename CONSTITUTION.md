# CONSTITUTION.md — Herramienta de detección de neumonía (UAO-Neumonia)

> **Fuente única de verdad.** Ante cualquier duda sobre cómo se construye, prueba o
> publica este proyecto, este documento manda. Si el código y la constitución
> difieren, se corrige uno de los dos en el mismo commit.

## 1. Propósito

Ofrecer una herramienta de escritorio que apoye el diagnóstico médico de neumonía
a partir de radiografías de tórax (DICOM/JPG), clasificándolas en **bacteriana**,
**normal** o **viral** y explicando la decisión con un mapa de calor **Grad-CAM**.

## 2. No-propósito

No es un dispositivo médico certificado ni sustituye el criterio de un profesional.
No entrena modelos en producción ni almacena datos clínicos reales de pacientes.
El modo demo produce resultados ilustrativos y nunca debe interpretarse como diagnóstico.

## 3. Alcance

**Entra:** interfaz web local, preprocesamiento de imágenes, inferencia con un modelo
CNN preentrenado, modo demo sin modelo, Grad-CAM y reporte PDF con ambas imágenes.

**No entra:** entrenamiento del modelo, base de datos, autenticación, despliegue remoto.

## 4. Stack técnico

| Capa | Tecnología |
|------|------------|
| Lenguaje | Python 3.13 (fijado en `.python-version`) |
| Entorno y dependencias | `uv` (`pyproject.toml` + `uv.lock`); build con `hatchling` |
| Cálculo numérico | NumPy |
| Imágenes | Pillow; OpenCV y pydicom en el extra `ml` |
| Deep Learning | TensorFlow/Keras en el extra `ml` |
| Interfaz | FastAPI + Uvicorn + HTML/CSS/JS embebido; ReportLab para PDF |
| Calidad | Ruff (lint/formato), pytest, ty, pre-commit |
| CI/CD | GitHub Actions |

## 5. Estructura

```text
.
├── src/                   # API pública exigida por la guía
│   ├── load_model.py      # cargar el .h5 una sola vez
│   ├── read_img.py        # leer DICOM y JPEG
│   ├── preprocess_img.py  # resize 512×512, gris, CLAHE, normalizar
│   ├── grad_cam.py        # generar el mapa de calor
│   ├── integrator.py      # orquestación y fachada para ui
│   ├── ui.py              # UI web FastAPI + HTML embebido (solo importa integrator)
│   └── neumonia/          # implementación canónica (src-layout)
│       ├── constants.py   # constantes del dominio
│       ├── labels.py      # índices → etiquetas
│       ├── images.py      # lectura y preprocesamiento
│       ├── gradcam.py     # matemática pura de Grad-CAM
│       ├── predict.py     # interpretación y orquestación
│       ├── applogic.py    # lógica de UI independiente del framework
│       ├── demo.py        # modo demo ilustrativo sin dependencias ml
│       ├── model.py       # carga del modelo con caché (TF perezoso)
│       ├── tf_backend.py  # Grad-CAM con TensorFlow
│       └── app.py         # compatibilidad: reexporta ui.App
├── tests/                 # pruebas unitarias (pytest, patrón AAA)
├── tools/                 # utilidades del Makefile
├── docs/                  # documentación y diagramas
├── detector_neumonia.py   # lanzador histórico
└── main.py                # alias de arranque
```

Los módulos directos en `src/` son la API pública exigida por la guía. `ui.py`
contiene la interfaz y solo depende de `integrator`, que concentra las
dependencias de inferencia y reexporta el resto de la API. El paquete `neumonia`
conserva la lógica reutilizable y las pruebas; el `.h5` se carga una sola vez
mediante caché en `neumonia/model.py`.

## 6. Convenciones de código

- **Idioma:** identificadores y comentarios en español; términos técnicos en inglés
  cuando sean el nombre canónico (`Grad-CAM`, `DICOM`).
- **Estilo:** Ruff, `line-length = 100`, comillas dobles.
- **Tipado:** anotaciones en firmas públicas; paquete marcado como tipado (`py.typed`).
- **Import perezoso** de dependencias pesadas (`cv2`, `tensorflow`, `pydicom`,
  `PIL`, `tkcap`) dentro de las funciones que las usan, para que el núcleo y las
  pruebas no dependan de ellas.
- **Inyección de dependencias** en funciones de imagen/Grad-CAM para facilitar pruebas
  sin cargar OpenCV ni TensorFlow.
- Sin comentarios obvios; el código debe explicarse por sí mismo.

## 7. Flujo de trabajo

```bash
make install     # uv sync --extra ml
make dev         # + pre-commit install
make check       # lint + format-check + typecheck + test
make run         # UI web en http://127.0.0.1:8000
make help        # lista de objetivos
```

- Tras editar los módulos sueltos de `src/*.py`, ejecuta
  `uv sync --extra ml --reinstall-package neumonia` para reflejarlos en el entorno.
- Nada se sube con `make check` en rojo.
- Commits en **Conventional Commits** (`feat:`, `fix:`, `test:`, `docs:`, `chore:`...).
- `main` se modifica vía Pull Request.

## 8. Pruebas

- Ubicación: `tests/`, archivos `test_*.py`.
- Patrón **AAA** (Arrange · Act · Assert); una responsabilidad por prueba.
- Sin dependencias externas reales: se inyectan dobles y se usan datos sintéticos.
- **Objetivo del proyecto: 120+ pruebas unitarias** (actualmente 217).
- Cobertura medida con `make test-cov` (no es un umbral bloqueante todavía).

## 9. Datos y secretos

- El modelo `WilhemNet86.h5`, datasets e imágenes de pacientes **no se versionan**.
- No se incluyen credenciales ni datos personales en fixtures.

## 10. Historial de decisiones

| Fecha | Decisión | Motivo |
|-------|----------|--------|
| 2026-11-23 | Adoptar el estándar Jan Polanco Velasco | Reproducibilidad, calidad y gobernanza. |
| 2026-11-23 | Migrar de `requirements.txt` a `uv` + `pyproject.toml` | Una sola fuente de dependencias y lockfile determinista. |
| 2026-11-23 | Extraer la lógica a `src/neumonia` manteniendo el lanzador histórico | Permite 120+ pruebas sin cargar TF/OpenCV. |
| 2026-11-23 | Imports perezosos + inyección de dependencias | Probar el núcleo con NumPy y dobles. |
| 2026-11-23 | Dependencias de ML como extra `ml` | Arranque e CI rápidos y ligeros. |
| 2026-11-23 | Auditoría del estándar: 0 FALTA, 22 DIFIERE, 3 OK | Las diferencias son personalizaciones del proyecto; no se sobrescribieron. |
| 2026-11-23 | Modo demo con selección automática y casilla en la UI | Permite probar la aplicación sin el modelo `.h5` ni el extra `ml`. |
| 2026-11-23 | Módulos públicos de la guía en `src/` + caché del modelo | Cumplir los nombres exigidos sin duplicar lógica ni romper el `src-layout`. |
| 2026-11-23 | Backend de build `hatchling` | `uv_build` no empaqueta módulos sueltos junto al paquete `neumonia`. |
| 2026-11-23 | Regla de dependencia `ui.py → integrator` | La UI no importa TensorFlow ni `load_model`; se valida con un test AST. |
| 2026-11-23 | Compatibilidad con `pydicom` 3 | `read_file` desapareció; se usa `dcmread` con fallback a versiones antiguas. |
| 2026-11-23 | UI web local con FastAPI | Sustituye a Tkinter; HTML/CSS/JS embebido para mantener el requisito “solo `.py`” y la regla `ui.py → integrator`. |
| 2026-11-23 | Vista comparativa y PDF con imágenes | Dos paneles lado a lado; se retira la exportación CSV y el PDF adjunta estudio y mapa de calor. |

## 11. Notas y tareas pendientes

- [ ] Incluir el modelo entrenado (`WilhemNet86.h5`) fuera del repositorio; el modo demo permite probar la UI mientras tanto.
- [ ] Añadir capturas reales a `docs/` y al README.
- [ ] Generar diagrama de arquitectura con Archify.
- [x] Migrar la interfaz a una UI web local más mantenible.
- [x] Estandarizar estructura, Makefile, CI y 120+ pruebas unitarias.
