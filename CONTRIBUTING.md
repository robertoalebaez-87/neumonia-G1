# Contribución

Gracias por tu interés en mejorar esta herramienta. Sigue estos pasos.

## Entorno

```bash
uv sync
uv run pre-commit install
make check
```

## Ramas y commits

- Crea una rama descriptiva: `feat/mi-cambio`, `fix/mi-arreglo`.
- Usa **Conventional Commits**: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`,
  `style:`, `chore:`.
- Mensajes en imperativo y en un solo idioma.

## Estándares

- Ruff para lint y formato (`make lint`, `make format`).
- Tipado con `ty` (`make typecheck`).
- Pruebas con pytest, patrón AAA, sin dependencias externas reales.
- Si cambias arquitectura o proceso, actualiza `CONSTITUTION.md` en el mismo commit.

## Pull Request

1. Sincroniza con `main` y resuelve conflictos.
2. Verifica `make check` y `make pre-commit` en verde.
3. Describe el cambio, el motivo y cómo probarlo.
4. Completa la plantilla de PR.
