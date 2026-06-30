"""Construcción de conjuntos de entrenamiento según cada experimento."""

from __future__ import annotations

from pathlib import Path

from src.data.synthetic_dataset import load_synthetic_metadata
from src.utils.file_utils import read_csv


SYNTHETIC_FLAGS = {
    "cae": "use_cae",
    "vae": "use_vae",
    "gan": "use_gan",
}


def find_experiment(config: dict, name: str) -> dict:
    """Busca un experimento por nombre."""

    for experiment in config.get("experiments", []):
        if experiment["name"] == name:
            return experiment
    raise KeyError(f"Experimento no definido: {name}")


def build_training_rows(config: dict, experiment_name: str) -> list[dict[str, str]]:
    """Combina filas reales y sintéticas para entrenamiento."""

    experiment = find_experiment(config, experiment_name)
    metadata_dir = Path(config["paths"]["metadata_dir"])
    rows: list[dict[str, str]] = []

    if experiment.get("use_real", True):
        train_path = metadata_dir / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"No existe split de entrenamiento: {train_path}")
        rows.extend(read_csv(train_path))

    for source, flag in SYNTHETIC_FLAGS.items():
        if experiment.get(flag, False):
            rows.extend(load_synthetic_metadata(metadata_dir, source))

    if not rows:
        raise RuntimeError(f"El experimento {experiment_name} no tiene datos.")
    return rows


def build_eval_rows(config: dict, split: str) -> list[dict[str, str]]:
    """Carga validation o test usando solo imágenes reales."""

    if split not in {"validation", "test"}:
        raise ValueError("split debe ser validation o test.")
    path = Path(config["paths"]["metadata_dir"]) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe split {split}: {path}")
    return read_csv(path)

