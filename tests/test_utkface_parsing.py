from pathlib import Path

import pytest

from src.config import load_config
from src.data.age_groups import build_age_groups
from src.data.utkface_dataset import parse_utkface_filename


def test_parse_valid_utkface_name():
    groups = build_age_groups(load_config())
    record = parse_utkface_filename(Path("25_1_2_20170116174525125.jpg"), groups)
    assert record.age == 25
    assert record.gender == 1
    assert record.race == 2
    assert record.age_group == "young_adult"


def test_parse_invalid_gender():
    groups = build_age_groups(load_config())
    with pytest.raises(ValueError):
        parse_utkface_filename("25_8_0_bad.jpg", groups)

