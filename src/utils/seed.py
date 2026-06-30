"""Control de semillas para reproducibilidad."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Configura semillas de Python, NumPy y PyTorch si están disponibles."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass

