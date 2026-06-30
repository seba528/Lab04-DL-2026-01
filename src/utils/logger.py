"""Registro simple para scripts educativos."""

from __future__ import annotations

from datetime import datetime


def log(message: str) -> None:
    """Imprime un mensaje con marca temporal corta."""

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

