"""Selección de dispositivo de entrenamiento."""

from __future__ import annotations


def resolve_device(device_name: str = "auto"):
    """Devuelve un `torch.device` según la configuración solicitada."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyTorch no está instalado. Instale las dependencias antes de entrenar."
        ) from error

    requested = device_name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device debe ser auto, cpu, cuda o mps.")
    return torch.device(requested)

