"""Evaluación de un modelo multitarea."""

from __future__ import annotations

from src.evaluation.metrics_age import age_regression_metrics
from src.evaluation.metrics_gender import confusion_matrix_binary, gender_metrics


def evaluate_model(model, loader, device) -> dict:
    """Ejecuta inferencia y devuelve métricas con valores crudos."""

    import torch

    model.eval()
    gender_targets: list[int] = []
    gender_predictions: list[int] = []
    age_targets: list[float] = []
    age_predictions: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gender_logits, age_output = model(images)
            gender_predictions.extend(gender_logits.argmax(dim=1).cpu().tolist())
            gender_targets.extend(batch["gender"].cpu().tolist())
            age_predictions.extend(age_output.detach().cpu().view(-1).tolist())
            age_targets.extend(batch["age"].cpu().tolist())

    metrics = {}
    metrics.update(gender_metrics(gender_targets, gender_predictions))
    metrics.update(age_regression_metrics(age_targets, age_predictions))
    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix_binary(gender_targets, gender_predictions),
        "gender_targets": gender_targets,
        "gender_predictions": gender_predictions,
        "age_targets": age_targets,
        "age_predictions": age_predictions,
    }

