"""Entrena DCGAN por grupo de edad."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena GAN por grupo.")
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--group", default="")
    args = parser.parse_args()

    from torch.utils.data import DataLoader

    from src.models.model_factory import build_gan
    from src.training.train_gan import train_gan

    config = load_config(args.config)
    device = resolve_device(config["project"]["device"])
    rows = read_csv(Path(config["paths"]["metadata_dir"]) / "train.csv")
    groups = sorted({row["age_group"] for row in rows})
    if args.group:
        groups = [args.group]

    transform = build_base_transform(config)
    for group in groups:
        loader = DataLoader(
            UTKFaceDataset([row for row in rows if row["age_group"] == group], transform=transform),
            batch_size=int(config["gan"]["batch_size"]),
            shuffle=True,
        )
        generator, discriminator = build_gan(config)
        output_dir = Path(config["paths"]["models_dir"]) / "gan" / group
        train_gan(generator, discriminator, loader, config, device, output_dir)
        print(f"GAN guardada: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
