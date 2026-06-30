"""Validaciones tempranas para evitar errores confusos en ejecución."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SUPPORTED_COLOR_SPACES = {"RGB", "GRAYSCALE", "HSV", "YCbCr"}
SUPPORTED_AGE_MODES = {"regression", "classification"}


def validate_config(config: dict[str, Any], *, require_dataset: bool = False) -> None:
    """Valida las claves principales de la configuración del laboratorio."""

    _validate_paths(config, require_dataset=require_dataset)
    _validate_split(config)
    _validate_preprocessing(config)
    _validate_age(config)
    _validate_experiments(config)


def _validate_paths(config: dict[str, Any], *, require_dataset: bool) -> None:
    paths = config.get("paths", {})
    original_dir = paths.get("original_utkface_dir")
    if not original_dir:
        raise ValueError("Debe definir paths.original_utkface_dir.")

    if require_dataset:
        dataset_path = Path(original_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"No existe paths.original_utkface_dir: {dataset_path}"
            )


def _validate_split(config: dict[str, Any]) -> None:
    split = config.get("split", {})
    total = float(split.get("train", 0)) + float(split.get("validation", 0)) + float(
        split.get("test", 0)
    )
    if abs(total - 1.0) > 1e-6:
        raise ValueError("split.train + split.validation + split.test debe ser 1.0.")
    for name in ("train", "validation", "test"):
        if float(split.get(name, 0)) <= 0:
            raise ValueError(f"split.{name} debe ser mayor que cero.")


def _validate_preprocessing(config: dict[str, Any]) -> None:
    preprocessing = config.get("preprocessing", {})
    image_size = preprocessing.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(int(value) <= 0 for value in image_size)
    ):
        raise ValueError("preprocessing.image_size debe tener dos enteros positivos.")

    color_space = preprocessing.get("color_space")
    if color_space not in SUPPORTED_COLOR_SPACES:
        raise ValueError(
            "preprocessing.color_space debe ser RGB, GRAYSCALE, HSV o YCbCr."
        )


def _validate_age(config: dict[str, Any]) -> None:
    age = config.get("age", {})
    if age.get("prediction_mode") not in SUPPORTED_AGE_MODES:
        raise ValueError("age.prediction_mode debe ser regression o classification.")

    groups = age.get("age_groups", [])
    if not groups:
        raise ValueError("Debe definir al menos un grupo de edad.")

    sorted_groups = sorted(groups, key=lambda item: int(item["min_age"]))
    previous_max = -1
    for group in sorted_groups:
        min_age = int(group["min_age"])
        max_age = int(group["max_age"])
        if min_age > max_age:
            raise ValueError(f"Grupo de edad inválido: {group['name']}")
        if min_age <= previous_max:
            raise ValueError("Los grupos de edad no deben solaparse.")
        previous_max = max_age


def _validate_experiments(config: dict[str, Any]) -> None:
    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("Debe definir al menos un experimento.")

    names = [experiment.get("name") for experiment in experiments]
    if len(names) != len(set(names)):
        raise ValueError("Los nombres de experimentos deben ser únicos.")

    for experiment in experiments:
        if experiment.get("use_gan") and not config.get("gan", {}).get(
            "synthetic_age_label_strategy"
        ):
            raise ValueError(
                "gan.synthetic_age_label_strategy es obligatorio al usar GAN."
            )

