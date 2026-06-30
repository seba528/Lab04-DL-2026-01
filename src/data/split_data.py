"""Particiones reproducibles con estratificación opcional."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from src.data.utkface_dataset import UTKFaceRecord


def split_records(
    records: Iterable[UTKFaceRecord],
    split_config: dict,
    *,
    seed: int,
) -> dict[str, list[UTKFaceRecord]]:
    """Divide registros en train, validation y test."""

    records_list = list(records)
    stratify_by = split_config.get("stratify_by", [])
    grouped = _group_records(records_list, stratify_by)

    output = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)

    for group_records in grouped.values():
        shuffled = list(group_records)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * float(split_config["train"])))
        n_validation = int(round(n * float(split_config["validation"])))

        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_validation = min(max(n_validation, 1), n - n_train - 1)
        else:
            n_train = max(1, n - 1)
            n_validation = 0

        slices = {
            "train": shuffled[:n_train],
            "validation": shuffled[n_train : n_train + n_validation],
            "test": shuffled[n_train + n_validation :],
        }
        for split_name, split_rows in slices.items():
            output[split_name].extend(
                replace(record, split=split_name) for record in split_rows
            )

    return output


def _group_records(
    records: list[UTKFaceRecord],
    stratify_by: list[str],
) -> dict[tuple, list[UTKFaceRecord]]:
    if not stratify_by:
        return {("all",): records}

    grouped: dict[tuple, list[UTKFaceRecord]] = defaultdict(list)
    for record in records:
        row = record.to_dict()
        key = tuple(row[field] for field in stratify_by)
        grouped[key].append(record)
    return grouped

