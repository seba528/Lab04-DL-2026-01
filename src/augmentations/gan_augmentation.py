"""Generación de imágenes sintéticas con GAN."""

from __future__ import annotations

from pathlib import Path


def generate_gan_images(generator, config: dict, group_name: str, age: float, gender: int) -> list[dict]:
    """Genera muestras desde un generador DCGAN."""

    import torch
    from torchvision.utils import save_image

    output_dir = Path(config["paths"]["generated_dir"]) / "gan" / group_name
    output_dir.mkdir(parents=True, exist_ok=True)
    count = int(config["gan"]["samples_per_group"])
    latent_dim = int(config["gan"]["latent_dim"])
    strategy = config["gan"].get("synthetic_age_label_strategy", "group_midpoint")
    rows: list[dict] = []

    generator.eval()
    with torch.no_grad():
        z = torch.randn(count, latent_dim, 1, 1)
        generated = generator(z)
        for index, image in enumerate(generated):
            path = output_dir / f"gan_{index:06d}.png"
            save_image((image.clamp(-1, 1) + 1) / 2, path)
            rows.append(
                {
                    "image_path": str(path),
                    "age": age,
                    "gender": gender,
                    "age_group": group_name,
                    "source": "gan",
                    "generator_type": "gan",
                    "generator_group": group_name,
                    "synthetic_age_label_strategy": strategy,
                    "original_image_path": "",
                }
            )
    return rows

