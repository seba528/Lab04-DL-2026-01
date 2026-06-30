"""Funciones pequeñas para crear carpetas y guardar archivos estructurados."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Crea una carpeta si no existe y devuelve su ruta."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Guarda un diccionario como JSON legible."""

    json_path = Path(path)
    ensure_dir(json_path.parent)
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    """Guarda filas tabulares usando la librería estándar."""

    csv_path = Path(path)
    ensure_dir(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Lee un CSV completo como lista de diccionarios."""

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))

