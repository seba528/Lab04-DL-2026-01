from pathlib import Path

import pytest
import yaml

from src.config import load_config
from src.config.validate_config import validate_config


def test_load_config_merges_defaults():
    config = load_config(validate=True)
    assert config["project"]["name"] == "LAB04-DL-2026-01"
    assert config["paths"]["original_utkface_dir"] == "data/raw/UTKFace"
    assert config["experiments"][0]["name"] == "E00_real_only"


def test_invalid_split_is_rejected():
    config = load_config(validate=False)
    config["split"]["test"] = 0.20
    with pytest.raises(ValueError):
        validate_config(config)


def test_missing_original_dataset_can_be_required(tmp_path: Path):
    config = load_config(validate=False)
    config["paths"]["original_utkface_dir"] = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        validate_config(config, require_dataset=True)

