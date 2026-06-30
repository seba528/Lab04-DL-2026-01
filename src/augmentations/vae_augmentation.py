"""Generación de imágenes sintéticas con VAE."""

from __future__ import annotations

from pathlib import Path


def generate_vae_images(model, config: dict, group_name: str, age: float, gender: int) -> list[dict]:
    """Muestrea imágenes desde el espacio latente del VAE."""

    import torch
    from torchvision.utils import save_image

    output_dir = Path(config["paths"]["generated_dir"]) / "vae" / group_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    count = int(config["vae"]["samples_per_group"])
    latent_dim = int(config["vae"]["latent_dim"])
    strategy = config["vae"].get("synthetic_age_label_strategy", "group_midpoint")

    model.eval()
    with torch.no_grad():
        z = torch.randn(count, latent_dim)
        generated = model.decode(z)
        for index, image in enumerate(generated):
            path = output_dir / f"vae_{index:06d}.png"
            save_image((image.clamp(-1, 1) + 1) / 2, path)
            rows.append(
                {
                    "image_path": str(path),
                    "age": age,
                    "gender": gender,
                    "age_group": group_name,
                    "source": "vae",
                    "generator_type": "vae",
                    "generator_group": group_name,
                    "synthetic_age_label_strategy": strategy,
                    "original_image_path": "",
                }
            )
    return rows

