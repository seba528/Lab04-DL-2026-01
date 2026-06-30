"""Lectura de metadatos de imágenes sintéticas."""

from __future__ import annotations

from pathlib import Path

from src.utils.file_utils import read_csv


def load_synthetic_metadata(metadata_dir: str | Path, source: str) -> list[dict[str, str]]:
    """Carga metadatos de una fuente sintética si el archivo existe."""

    path = Path(metadata_dir) / f"{source}_metadata.csv"
    if not path.exists():
        return []
    rows = read_csv(path)
    for row in rows:
        row.setdefault("source", source)
    return rows


def validate_synthetic_rows(rows: list[dict[str, str]]) -> None:
    """Verifica que cada imagen sintética tenga etiquetas mínimas."""

    required = {"image_path", "age", "gender", "age_group", "source"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Fila sintética incompleta: faltan {sorted(missing)}")
        if not Path(row["image_path"]).exists():
            raise FileNotFoundError(f"No existe imagen sintética: {row['image_path']}")

