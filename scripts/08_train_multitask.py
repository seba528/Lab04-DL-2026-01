"""Entrena un experimento multitarea definido en YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentations.combine_datasets import build_eval_rows, build_training_rows, find_experiment
from src.augmentations.traditional_transforms import build_train_transform
from src.config import load_config
from src.data.preprocessing import build_base_transform
from src.data.utkface_dataset import UTKFaceDataset
from src.utils.device import resolve_device
from src.utils.file_utils import ensure_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena modelo multitarea.")
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--experiment", default="E00_real_only")
    args = parser.parse_args()

    from torch.utils.data import DataLoader

    from src.models.model_factory import build_multitask_model
    from src.training.train_multitask import train_multitask_model

    config = load_config(args.config)
    experiment = find_experiment(config, args.experiment)
    device = resolve_device(config["project"]["device"])
    train_rows = build_training_rows(config, args.experiment)
    val_rows = build_eval_rows(config, "validation")

    train_transform = build_train_transform(
        config,
        enabled=bool(experiment.get("use_traditional_aug", False)),
    )
    eval_transform = build_base_transform(config)
    train_loader = DataLoader(
        UTKFaceDataset(train_rows, transform=train_transform),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        UTKFaceDataset(val_rows, transform=eval_transform),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    model = build_multitask_model(config)
    output_dir = ensure_dir(Path(config["paths"]["models_dir"]) / args.experiment)
    history = train_multitask_model(model, train_loader, val_loader, config, device, output_dir)

    report_dir = ensure_dir(Path(config["paths"]["reports_dir"]) / args.experiment)
    (report_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Experimento {args.experiment} entrenado con {len(history)} épocas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
