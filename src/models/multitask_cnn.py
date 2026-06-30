"""CNN multitarea para género y edad."""

from __future__ import annotations

import torch
from torch import nn


class MultiTaskCNN(nn.Module):
    """Extrae rasgos compartidos y predice género y edad."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        gender_output_classes: int = 2,
        age_output_dim: int = 1,
        age_mode: str = "regression",
    ) -> None:
        super().__init__()
        self.age_mode = age_mode
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gender_head = nn.Linear(hidden_dim, gender_output_classes)
        self.age_head = nn.Linear(hidden_dim, age_output_dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.shared(self.features(images))
        gender_logits = self.gender_head(representation)
        age_output = self.age_head(representation)
        if self.age_mode == "regression":
            age_output = age_output.squeeze(1)
        return gender_logits, age_output

