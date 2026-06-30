"""Aumento tradicional configurable con torchvision."""

from __future__ import annotations

from src.data.preprocessing import convert_color_space


def build_train_transform(config: dict, *, enabled: bool = True):
    """Crea la transformación de entrenamiento."""

    try:
        from torchvision import transforms
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("torchvision es requerido para augmentation.") from error

    preprocessing = config["preprocessing"]
    aug = config.get("traditional_augmentation", {})
    image_size = tuple(preprocessing["image_size"])
    color_space = preprocessing["color_space"]

    steps = [transforms.Lambda(lambda image: convert_color_space(image, color_space))]
    if enabled and aug.get("enabled", False):
        if aug.get("random_horizontal_flip", False):
            steps.append(transforms.RandomHorizontalFlip())
        if aug.get("random_rotation_degrees", 0):
            steps.append(transforms.RandomRotation(aug["random_rotation_degrees"]))
        if aug.get("random_affine", False):
            steps.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)))
        if aug.get("color_jitter", False) and color_space != "GRAYSCALE":
            steps.append(
                transforms.ColorJitter(
                    brightness=aug.get("brightness", 0.0),
                    contrast=aug.get("contrast", 0.0),
                    saturation=aug.get("saturation", 0.0),
                )
            )
        if aug.get("gaussian_blur", False):
            steps.append(transforms.GaussianBlur(kernel_size=3))

    steps.extend([transforms.Resize(image_size), transforms.ToTensor()])
    if preprocessing.get("normalize", True):
        channels = 1 if color_space == "GRAYSCALE" else 3
        steps.append(transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels))
    return transforms.Compose(steps)

