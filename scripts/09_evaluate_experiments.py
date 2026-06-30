"""Evalúa y compara experimentos multitarea."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentations.combine_datasets import build_eval_rows
from src.config import load_config
from src.data.preprocessing import build_base_transform
from src.data.utkface_dataset import UTKFaceDataset
from src.evaluation.compare_experiments import compare_experiments
from src.evaluation.evaluate_multitask import evaluate_model
from src.utils.device import resolve_device
from src.utils.file_utils import ensure_dir, write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa experimentos.")
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--experiment", action="append")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from src.models.model_factory import build_multitask_model

    config = load_config(args.config)
    device = resolve_device(config["project"]["device"])
    experiments = args.experiment or [item["name"] for item in config["experiments"]]
    test_rows = build_eval_rows(config, "test")
    transform = build_base_transform(config)
    loader = DataLoader(
        UTKFaceDataset(test_rows, transform=transform),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
    )

    for experiment in experiments:
        checkpoint = Path(config["paths"]["models_dir"]) / experiment / "best_model.pt"
        if not checkpoint.exists():
            print(f"Saltando {experiment}: no existe {checkpoint}")
            continue
        model = build_multitask_model(config).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
        result = evaluate_model(model, loader, device)
        metrics_dir = ensure_dir(Path(config["paths"]["metrics_dir"]) / experiment)
        write_json(metrics_dir / "test_metrics.json", result)
        write_csv(metrics_dir / "test_metrics.csv", [result["metrics"]], list(result["metrics"]))
        print(f"Evaluado {experiment}: {result['metrics']}")

    compare_experiments(
        config["paths"]["metrics_dir"],
        Path(config["paths"]["reports_dir"]) / "comparison_table.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
