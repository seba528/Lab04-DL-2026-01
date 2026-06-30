"""Preprocesamiento determinista de imágenes."""

from __future__ import annotations


def build_base_transform(config: dict):
    """Crea una transformación determinista con torchvision."""

    try:
        from torchvision import transforms
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "torchvision es requerido para construir transformaciones."
        ) from error

    preprocessing = config["preprocessing"]
    image_size = tuple(preprocessing["image_size"])
    color_space = preprocessing["color_space"]

    steps = [
        transforms.Lambda(lambda image: convert_color_space(image, color_space)),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
    if preprocessing.get("normalize", True):
        channels = 1 if color_space == "GRAYSCALE" else 3
        steps.append(transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels))
    return transforms.Compose(steps)


def convert_color_space(image, color_space: str):
    """Convierte una imagen PIL al espacio de color configurado."""

    if color_space == "RGB":
        return image.convert("RGB")
    if color_space == "GRAYSCALE":
        return image.convert("L")
    if color_space == "HSV":
        return image.convert("HSV")
    if color_space == "YCbCr":
        return image.convert("YCbCr")
    raise ValueError(f"Espacio de color no soportado: {color_space}")

