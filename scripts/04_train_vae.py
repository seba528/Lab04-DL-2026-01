"""Entrena VAE por grupo de edad."""

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
    parser = argparse.ArgumentParser(description="Entrena VAE por grupo.")
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--group", default="")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from src.models.model_factory import build_vae
    from src.training.train_vae import train_vae

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
            batch_size=int(config["vae"]["batch_size"]),
            shuffle=True,
        )
        model = build_vae(config)
        output_path = Path(config["paths"]["models_dir"]) / "vae" / group / "best_model.pt"
        train_vae(model, loader, config, device, output_path)
        print(f"VAE guardado: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
