from pathlib import Path

import pytest

from src.data.synthetic_dataset import validate_synthetic_rows


def test_synthetic_metadata_requires_existing_image(tmp_path: Path):
    row = {
        "image_path": str(tmp_path / "missing.png"),
        "age": "25",
        "gender": "0",
        "age_group": "young_adult",
        "source": "cae",
    }
    with pytest.raises(FileNotFoundError):
        validate_synthetic_rows([row])

