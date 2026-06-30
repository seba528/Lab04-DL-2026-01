"""Entrenamiento básico de DCGAN."""

from __future__ import annotations

import torch
from torch import nn


def train_gan(generator, discriminator, loader, config: dict, device, output_dir):
    """Entrena un DCGAN pequeño para fines didácticos."""

    generator.to(device)
    discriminator.to(device)
    criterion = nn.BCEWithLogitsLoss()
    latent_dim = int(config["gan"]["latent_dim"])
    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(config["gan"]["learning_rate_generator"]),
        betas=(float(config["gan"].get("beta1", 0.5)), 0.999),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(config["gan"]["learning_rate_discriminator"]),
        betas=(float(config["gan"].get("beta1", 0.5)), 0.999),
    )

    for _ in range(int(config["gan"]["epochs"])):
        for batch in loader:
            real = batch["image"].to(device)
            batch_size = real.size(0)
            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            opt_d.zero_grad()
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake = generator(noise).detach()
            d_loss = criterion(discriminator(real), real_targets) + criterion(
                discriminator(fake), fake_targets
            )
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake = generator(noise)
            g_loss = criterion(discriminator(fake), real_targets)
            g_loss.backward()
            opt_g.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": generator.state_dict(), "config": config}, output_dir / "generator.pt")
    torch.save(
        {"model_state_dict": discriminator.state_dict(), "config": config},
        output_dir / "discriminator.pt",
    )

