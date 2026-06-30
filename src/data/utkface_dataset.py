"""Parser y dataset PyTorch para UTKFace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from src.data.age_groups import AgeGroup, assign_age_group

try:
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError:
    TorchDataset = object


@dataclass(frozen=True)
class UTKFaceRecord:
    """Una imagen real de UTKFace y sus etiquetas extraídas del nombre."""

    image_path: str
    age: float
    gender: int
    age_group: str
    source: str = "real"
    race: int | None = None
    split: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def parse_utkface_filename(path: str | Path, age_groups: list[AgeGroup]) -> UTKFaceRecord:
    """Extrae edad y género desde nombres como `25_1_2_201701.jpg`."""

    image_path = Path(path)
    parts = image_path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Nombre UTKFace inválido: {image_path.name}. "
            "Se esperaba edad_genero_raza_fecha.jpg."
        )

    try:
        age = float(parts[0])
        gender = int(parts[1])
        race = int(parts[2]) if len(parts) >= 3 else None
    except ValueError as error:
        raise ValueError(f"Etiquetas inválidas en {image_path.name}.") from error

    if age < 0:
        raise ValueError(f"La edad no puede ser negativa: {image_path.name}.")
    if gender not in {0, 1}:
        raise ValueError(f"Género fuera de rango en {image_path.name}: {gender}.")

    group = assign_age_group(age, age_groups)
    return UTKFaceRecord(
        image_path=str(image_path),
        age=age,
        gender=gender,
        race=race,
        age_group=group.name,
    )


def discover_utkface_records(
    dataset_dir: str | Path,
    age_groups: list[AgeGroup],
    extensions: list[str],
    *,
    max_images: int = 0,
) -> tuple[list[UTKFaceRecord], list[dict[str, str]]]:
    """Recorre la carpeta original y separa registros válidos de errores."""

    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio UTKFace: {root}")

    allowed = {extension.lower() for extension in extensions}
    records: list[UTKFaceRecord] = []
    errors: list[dict[str, str]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        try:
            _validate_image(path)
            records.append(parse_utkface_filename(path, age_groups))
        except Exception as error:  # noqa: BLE001 - se registra y continúa.
            errors.append({"image_path": str(path), "error": str(error)})
        if max_images and len(records) >= max_images:
            break

    if not records:
        raise RuntimeError(f"No se encontraron imágenes válidas en {root}.")
    return records, errors


def _validate_image(path: Path) -> None:
    """Abre la imagen para detectar archivos corruptos de forma temprana."""

    with Image.open(path) as image:
        image.verify()


class UTKFaceDataset(TorchDataset):
    """Dataset PyTorch que devuelve imagen, género, edad y metadatos básicos."""

    def __init__(self, records: list[UTKFaceRecord | dict], transform: Callable | None = None):
        if TorchDataset is object:
            raise ModuleNotFoundError("PyTorch es requerido para usar UTKFaceDataset.")
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records[index]
        row = record.to_dict() if isinstance(record, UTKFaceRecord) else record
        with Image.open(row["image_path"]) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            raise TypeError("La transformación debe convertir la imagen a torch.Tensor.")

        gender = torch.tensor(int(row["gender"]), dtype=torch.long)
        age = torch.tensor(float(row["age"]), dtype=torch.float32)
        return {
            "image": image,
            "gender": gender,
            "age": age,
            "age_group": row["age_group"],
            "source": row.get("source", "real"),
            "image_path": row["image_path"],
        }
