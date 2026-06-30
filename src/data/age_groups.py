"""Definición y asignación de grupos de edad."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgeGroup:
    """Representa un intervalo cerrado de edades."""

    name: str
    min_age: int
    max_age: int

    @property
    def midpoint(self) -> float:
        return (self.min_age + self.max_age) / 2.0


def build_age_groups(config: dict) -> list[AgeGroup]:
    """Construye grupos desde la sección `age.age_groups`."""

    groups = [
        AgeGroup(
            name=item["name"],
            min_age=int(item["min_age"]),
            max_age=int(item["max_age"]),
        )
        for item in config["age"]["age_groups"]
    ]
    return sorted(groups, key=lambda group: group.min_age)


def assign_age_group(age: float | int, groups: list[AgeGroup]) -> AgeGroup:
    """Devuelve el grupo que contiene la edad indicada."""

    for group in groups:
        if group.min_age <= float(age) <= group.max_age:
            return group
    raise ValueError(f"La edad {age} no pertenece a ningún grupo configurado.")


def group_midpoint(name: str, groups: list[AgeGroup]) -> float:
    """Obtiene el punto medio de un grupo por nombre."""

    for group in groups:
        if group.name == name:
            return group.midpoint
    raise KeyError(f"Grupo de edad desconocido: {name}")

