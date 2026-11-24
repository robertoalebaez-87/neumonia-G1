"""Compatibilidad: la interfaz web vive en el módulo público ``ui``."""

from __future__ import annotations

from ui import APP_TITLE, App, main

__all__ = ["APP_TITLE", "App", "main"]
