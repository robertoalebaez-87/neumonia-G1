"""Lectura de imágenes DICOM y JPEG; interfaz pública exigida por la guía."""

from __future__ import annotations

from neumonia.images import read_dicom_file, read_jpg_file

__all__ = ["read_dicom_file", "read_jpg_file"]
