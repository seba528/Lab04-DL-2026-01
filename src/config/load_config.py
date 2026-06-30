"""Funciones para cargar configuración YAML del laboratorio."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.config.validate_config import validate_config


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Lee un archivo YAML y devuelve un diccionario."""

    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"El archivo YAML debe contener un diccionario: {yaml_path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusiona diccionarios anidados sin modificar los originales."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(
    config_path: str | Path = "config/path.yaml",
    default_path: str | Path = "config/default.yaml",
    experiments_path: str | Path = "config/experiments.yaml",
    *,
    validate: bool = True,
    require_dataset: bool = False,
) -> dict[str, Any]:
    """Carga defaults, configuración principal y experimentos.

    Las rutas relativas se mantienen como texto para que los scripts decidan
    cómo resolverlas respecto de la raíz del proyecto.
    """

    config = read_yaml(default_path)
    config = deep_merge(config, read_yaml(config_path))
    config = deep_merge(config, read_yaml(experiments_path))

    if validate:
        validate_config(config, require_dataset=require_dataset)
    return config

