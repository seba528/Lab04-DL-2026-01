from src.config import load_config
from src.data.age_groups import build_age_groups
from src.data.utkface_dataset import parse_utkface_filename
from src.data.split_data import split_records


def test_split_records_is_reproducible():
    config = load_config()
    groups = build_age_groups(config)
    records = [
        parse_utkface_filename(f"{20 + index}_0_0_file{index}.jpg", groups)
        for index in range(10)
    ]
    first = split_records(records, config["split"], seed=42)
    second = split_records(records, config["split"], seed=42)
    assert [row.image_path for row in first["train"]] == [
        row.image_path for row in second["train"]
    ]

