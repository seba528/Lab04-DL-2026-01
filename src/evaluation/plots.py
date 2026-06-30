"""Gráficos de resultados usando matplotlib."""

from __future__ import annotations

from pathlib import Path


def plot_loss_curves(history: list[dict], output_path: str | Path) -> None:
    """Guarda curvas de pérdida si matplotlib está disponible."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    if not history:
        return
    epochs = [row["epoch"] for row in history]
    plt.figure()
    plt.plot(epochs, [row["train_total_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_total_loss"] for row in history], label="validation")
    plt.xlabel("Época")
    plt.ylabel("Pérdida total")
    plt.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

