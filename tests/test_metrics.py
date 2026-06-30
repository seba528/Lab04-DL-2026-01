from src.evaluation.metrics_age import age_regression_metrics
from src.evaluation.metrics_gender import gender_metrics


def test_gender_metrics_perfect_prediction():
    metrics = gender_metrics([0, 1, 1], [0, 1, 1])
    assert metrics["gender_accuracy"] == 1.0
    assert metrics["gender_f1"] == 1.0


def test_age_regression_metrics():
    metrics = age_regression_metrics([10, 20], [12, 18])
    assert metrics["age_mae"] == 2.0
    assert metrics["age_rmse"] == 2.0

