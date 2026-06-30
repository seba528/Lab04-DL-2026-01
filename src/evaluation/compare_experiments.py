"""Comparación tabular de experimentos."""

from __future__ import annotations

from pathlib import Path

from src.utils.file_utils import read_csv, write_csv


def compare_experiments(metrics_dir: str | Path, output_path: str | Path) -> list[dict]:
    """Consolida archivos `test_metrics.csv` en una tabla final."""

    rows: list[dict] = []
    for path in sorted(Path(metrics_dir).glob("*/test_metrics.csv")):
        experiment_rows = read_csv(path)
        if not experiment_rows:
            continue
        row = {"experiment": path.parent.name, **experiment_rows[0]}
        rows.append(row)
    if rows:
        write_csv(output_path, rows, list(rows[0]))
    return rows

