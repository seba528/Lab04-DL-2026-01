"""Prepara metadatos y splits reproducibles de UTKFace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.age_groups import build_age_groups
from src.data.split_data import split_records
from src.data.utkface_dataset import discover_utkface_records
from src.utils.file_utils import ensure_dir, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepara UTKFace para LAB04.")
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--defaults", default="config/default.yaml")
    parser.add_argument("--experiments", default="config/experiments.yaml")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config, args.defaults, args.experiments, require_dataset=True)
    metadata_dir = ensure_dir(config["paths"]["metadata_dir"])
    reports_dir = ensure_dir(config["paths"]["reports_dir"])
    age_groups = build_age_groups(config)

    records, errors = discover_utkface_records(
        config["paths"]["original_utkface_dir"],
        age_groups,
        config["dataset"]["image_extensions"],
        max_images=int(config["dataset"].get("max_images", 0)),
    )
    fieldnames = ["image_path", "age", "gender", "age_group", "source", "race", "split"]
    write_csv(metadata_dir / "real_metadata.csv", [record.to_dict() for record in records], fieldnames)

    splits = split_records(records, config["split"], seed=int(config["project"]["seed"]))
    for split_name, split_records_ in splits.items():
        write_csv(
            metadata_dir / f"{split_name}.csv",
            [record.to_dict() for record in split_records_],
            fieldnames,
        )

    if errors:
        write_csv(metadata_dir / "invalid_images.csv", errors, ["image_path", "error"])

    summary = {
        "total_valid_images": len(records),
        "total_invalid_images": len(errors),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "original_utkface_dir": str(Path(config["paths"]["original_utkface_dir"])),
    }
    (reports_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
