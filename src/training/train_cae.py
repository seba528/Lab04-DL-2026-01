"""Entrenamiento de autoencoder convolucional."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def train_cae(model, loader, config: dict, device, output_path):
    """Entrena un CAE con pérdida MSE."""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["cae"]["learning_rate"]))
    for _ in range(int(config["cae"]["epochs"])):
        model.train()
        for batch in loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            reconstruction = model(images)
            loss = F.mse_loss(reconstruction, images)
            loss.backward()
            optimizer.step()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, output_path)

