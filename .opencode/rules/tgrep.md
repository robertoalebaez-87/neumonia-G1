# Regla: búsqueda de código con tgrep

Al explorar este repositorio, prefiere `tgrep` sobre `grep`/`rg` para búsquedas
por intención (funciones, clases, patrones) cuando esté instalado:

```bash
tgrep "preprocess"
tgrep "read_dicom_file" src/
```

Si `tgrep` no está disponible, usa `rg` y luego los editores/lectores de archivos.
