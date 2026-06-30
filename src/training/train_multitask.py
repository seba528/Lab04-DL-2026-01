"""Bucle de entrenamiento del modelo multitarea."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.training.losses import MultiTaskLoss
from src.utils.file_utils import ensure_dir


def train_multitask_model(model, train_loader, val_loader, config: dict, device, output_dir):
    """Entrena, valida y guarda el mejor modelo por pérdida de validación."""

    output_dir = ensure_dir(output_dir)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = MultiTaskLoss(
        lambda_gender=float(config["losses"]["lambda_gender"]),
        lambda_age=float(config["losses"]["lambda_age"]),
        age_mode=config["age"]["prediction_mode"],
    )
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    patience = int(config["training"].get("patience", 8))
    stale_epochs = 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_losses = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_losses = _run_epoch(model, val_loader, criterion, None, device)
        row = {"epoch": epoch, **_prefix("train", train_losses), **_prefix("val", val_losses)}
        history.append(row)

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            stale_epochs = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config},
                output_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
            if config["training"].get("early_stopping", True) and stale_epochs >= patience:
                break

    _write_history(output_dir / "train_log.csv", history)
    return history


def _run_epoch(model, loader, criterion, optimizer, device) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total_loss": 0.0, "gender_loss": 0.0, "age_loss": 0.0}
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            images = batch["image"].to(device)
            gender_targets = batch["gender"].to(device)
            age_targets = batch["age"].to(device)
            if training:
                optimizer.zero_grad()
            gender_logits, age_output = model(images)
            losses = criterion(gender_logits, age_output, gender_targets, age_targets)
            if training:
                losses.total.backward()
                optimizer.step()
            batch_size = images.size(0)
            sample_count += batch_size
            totals["total_loss"] += losses.total.item() * batch_size
            totals["gender_loss"] += losses.gender.item() * batch_size
            totals["age_loss"] += losses.age.item() * batch_size

    if sample_count == 0:
        raise RuntimeError("El DataLoader no contiene muestras.")
    return {key: value / sample_count for key, value in totals.items()}


def _prefix(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _write_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

