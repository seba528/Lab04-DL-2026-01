"""Entrenamiento de VAE."""

from __future__ import annotations

import torch

from src.training.losses import vae_loss


def train_vae(model, loader, config: dict, device, output_path):
    """Entrena un VAE con reconstrucción y KL."""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["vae"]["learning_rate"]))
    beta = float(config["vae"].get("beta", 1.0))
    for _ in range(int(config["vae"]["epochs"])):
        model.train()
        for batch in loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(images)
            loss, _, _ = vae_loss(reconstruction, images, mu, logvar, beta)
            loss.backward()
            optimizer.step()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, output_path)

