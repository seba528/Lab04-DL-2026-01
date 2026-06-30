import pytest

torch = pytest.importorskip("torch")

from src.config import load_config
from src.models.model_factory import build_multitask_model


def test_multitask_forward_pass():
    config = load_config()
    model = build_multitask_model(config)
    images = torch.randn(2, 3, 128, 128)
    gender_logits, age_output = model(images)
    assert gender_logits.shape == (2, 2)
    assert age_output.shape == (2,)

