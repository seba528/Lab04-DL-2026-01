"""Generación de imágenes sintéticas con un autoencoder convolucional."""

from __future__ import annotations

from pathlib import Path


def generate_cae_images(model, loader, config: dict, group_name: str) -> list[dict]:
    """Genera reconstrucciones con pequeñas perturbaciones latentes.

    La función espera un modelo con método `reconstruct_with_noise`. El script de
    generación decide cuántas muestras guardar y qué metadatos asociar.
    """

    import torch
    from torchvision.utils import save_image

    model.eval()
    output_dir = Path(config["paths"]["generated_dir"]) / "cae" / group_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    samples_per_group = int(config["cae"]["samples_per_group"])
    noise_std = float(config["cae"].get("noise_std", 0.0))
    written = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            generated = model.reconstruct_with_noise(images, noise_std=noise_std)
            for index, image in enumerate(generated):
                if written >= samples_per_group:
                    return rows
                path = output_dir / f"cae_{written:06d}.png"
                save_image((image.clamp(-1, 1) + 1) / 2, path)
                rows.append(
                    {
                        "image_path": str(path),
                        "age": float(batch["age"][index]),
                        "gender": int(batch["gender"][index]),
                        "age_group": batch["age_group"][index],
                        "source": "cae",
                        "generator_type": "cae",
                        "generator_group": group_name,
                        "synthetic_age_label_strategy": "inherit_from_original",
                        "original_image_path": batch["image_path"][index],
                    }
                )
                written += 1
    return rows

