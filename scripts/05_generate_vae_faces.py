"""Genera imágenes sintéticas usando VAE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentations.vae_augmentation import generate_vae_images
from src.config import load_config
from src.data.age_groups import build_age_groups, group_midpoint
from src.utils.file_utils import write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera imágenes con VAE.")
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--gender", type=int, default=0)
    args = parser.parse_args()

    import torch

    from src.models.model_factory import build_vae

    config = load_config(args.config)
    groups = build_age_groups(config)
    rows: list[dict] = []
    for group in groups:
        model = build_vae(config)
        checkpoint = Path(config["paths"]["models_dir"]) / "vae" / group.name / "best_model.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
        rows.extend(
            generate_vae_images(
                model,
                config,
                group.name,
                age=group_midpoint(group.name, groups),
                gender=args.gender,
            )
        )

    if rows:
        write_csv(Path(config["paths"]["metadata_dir"]) / "vae_metadata.csv", rows, list(rows[0]))
    print(f"Imágenes VAE registradas: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
