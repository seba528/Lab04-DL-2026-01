"""Métricas de edad para regresión o clasificación."""

from __future__ import annotations

import math


def age_regression_metrics(targets: list[float], predictions: list[float]) -> dict[str, float]:
    """Calcula MAE y RMSE."""

    if not targets:
        raise ValueError("No hay edades para calcular métricas.")
    errors = [float(pred) - float(target) for target, pred in zip(targets, predictions)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return {"age_mae": mae, "age_rmse": rmse}


def age_error_by_group(
    targets: list[float],
    predictions: list[float],
    groups: list[str],
) -> dict[str, float]:
    """Calcula MAE por grupo de edad."""

    grouped: dict[str, list[float]] = {}
    for target, prediction, group in zip(targets, predictions, groups):
        grouped.setdefault(group, []).append(abs(float(prediction) - float(target)))
    return {
        f"age_mae_{group}": sum(errors) / len(errors)
        for group, errors in grouped.items()
        if errors
    }

