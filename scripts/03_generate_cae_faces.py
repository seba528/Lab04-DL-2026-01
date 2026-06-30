"""Genera imágenes sintéticas usando CAE entrenado."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentations.cae_augmentation import generate_cae_images
from src.config import load_config
from src.data.preprocessing import build_base_transform
from src.data.utkface_dataset import UTKFaceDataset
from src.utils.file_utils import read_csv, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera imágenes con CAE.")
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--group", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from torch.utils.data import DataLoader

    from src.models.model_factory import build_cae

    config = load_config(args.config)
    rows = read_csv(Path(config["paths"]["metadata_dir"]) / "train.csv")
    groups = sorted({row["age_group"] for row in rows})
    if args.group:
        groups = [args.group]

    transform = build_base_transform(config)
    all_rows: list[dict] = []
    for group in groups:
        group_rows = [row for row in rows if row["age_group"] == group]
        loader = DataLoader(UTKFaceDataset(group_rows, transform=transform), batch_size=16)
        model = build_cae(config)
        checkpoint = Path(config["paths"]["models_dir"]) / "cae" / group / "best_model.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
        all_rows.extend(generate_cae_images(model, loader, config, group))

    if all_rows:
        write_csv(
            Path(config["paths"]["metadata_dir"]) / "cae_metadata.csv",
            all_rows,
            list(all_rows[0]),
        )
    print(f"Imágenes CAE registradas: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
