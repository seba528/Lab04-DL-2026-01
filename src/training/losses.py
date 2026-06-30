"""Pérdidas para entrenamiento multitarea y generativo."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MultiTaskLossOutput:
    total: torch.Tensor
    gender: torch.Tensor
    age: torch.Tensor


class MultiTaskLoss(nn.Module):
    """Combina pérdida de género y pérdida de edad."""

    def __init__(
        self,
        *,
        lambda_gender: float = 1.0,
        lambda_age: float = 1.0,
        age_mode: str = "regression",
    ) -> None:
        super().__init__()
        self.lambda_gender = lambda_gender
        self.lambda_age = lambda_age
        self.age_mode = age_mode
        self.gender_loss = nn.CrossEntropyLoss()
        self.age_regression_loss = nn.SmoothL1Loss()
        self.age_classification_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        gender_logits: torch.Tensor,
        age_output: torch.Tensor,
        gender_targets: torch.Tensor,
        age_targets: torch.Tensor,
    ) -> MultiTaskLossOutput:
        gender_loss = self.gender_loss(gender_logits, gender_targets)
        if self.age_mode == "classification":
            age_loss = self.age_classification_loss(age_output, age_targets.long())
        else:
            age_loss = self.age_regression_loss(age_output, age_targets.float())
        total = self.lambda_gender * gender_loss + self.lambda_age * age_loss
        return MultiTaskLossOutput(total=total, gender=gender_loss, age=age_loss)


def vae_loss(reconstruction, images, mu, logvar, beta: float = 1.0):
    """Calcula reconstrucción más divergencia KL."""

    reconstruction_loss = F.mse_loss(reconstruction, images, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss

