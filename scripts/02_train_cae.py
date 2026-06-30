"""Entrena autoencoders convolucionales por grupo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.preprocessing import build_base_transform
from src.data.utkface_dataset import UTKFaceDataset
from src.utils.device import resolve_device
from src.utils.file_utils import read_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena CAE por grupo de edad.")
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--group", default="", help="Grupo opcional de edad.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from torch.utils.data import DataLoader

    from src.models.model_factory import build_cae
    from src.training.train_cae import train_cae

    config = load_config(args.config)
    device = resolve_device(config["project"]["device"])
    rows = read_csv(Path(config["paths"]["metadata_dir"]) / "train.csv")
    groups = sorted({row["age_group"] for row in rows})
    if args.group:
        groups = [args.group]

    transform = build_base_transform(config)
    for group in groups:
        group_rows = [row for row in rows if row["age_group"] == group]
        loader = DataLoader(
            UTKFaceDataset(group_rows, transform=transform),
            batch_size=int(config["cae"]["batch_size"]),
            shuffle=True,
            num_workers=int(config["training"].get("num_workers", 0)),
        )
        model = build_cae(config)
        output_path = Path(config["paths"]["models_dir"]) / "cae" / group / "best_model.pt"
        train_cae(model, loader, config, device, output_path)
        print(f"CAE guardado: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
