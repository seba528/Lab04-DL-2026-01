"""Métricas de clasificación de género."""

from __future__ import annotations


def confusion_matrix_binary(targets: list[int], predictions: list[int]) -> list[list[int]]:
    matrix = [[0, 0], [0, 0]]
    for target, prediction in zip(targets, predictions):
        matrix[int(target)][int(prediction)] += 1
    return matrix


def gender_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    """Calcula accuracy, precision, recall y F1 ponderado."""

    matrix = confusion_matrix_binary(targets, predictions)
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("No hay muestras para calcular métricas.")
    accuracy = (matrix[0][0] + matrix[1][1]) / total
    per_class = []
    for klass in (0, 1):
        tp = matrix[klass][klass]
        fp = matrix[1 - klass][klass]
        fn = matrix[klass][1 - klass]
        support = sum(matrix[klass])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append((precision, recall, f1, support))
    weights = [item[3] / total for item in per_class]
    return {
        "gender_accuracy": accuracy,
        "gender_precision": sum(item[0] * weight for item, weight in zip(per_class, weights)),
        "gender_recall": sum(item[1] * weight for item, weight in zip(per_class, weights)),
        "gender_f1": sum(item[2] * weight for item, weight in zip(per_class, weights)),
    }

