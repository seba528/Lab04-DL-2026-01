"""Orquestador principal del laboratorio LAB04-DL-2026-01."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    """Construye la interfaz de línea de comandos del laboratorio."""

    parser = argparse.ArgumentParser(
        description=(
            "Orquesta la preparación, entrenamiento y evaluación del laboratorio "
            "LAB04-DL-2026-01."
        )
    )
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--defaults", default="config/default.yaml")
    parser.add_argument("--experiments", default="config/experiments.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra los comandos sin ejecutarlos.",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="Lista los experimentos definidos en YAML.")
    subparsers.add_parser("prepare", help="Prepara metadatos y splits de UTKFace.")
    subparsers.add_parser("minimal", help="Ejecuta el flujo mínimo del laboratorio.")
    subparsers.add_parser("extended", help="Ejecuta la extensión VAE/GAN del laboratorio.")
    subparsers.add_parser("all", help="Ejecuta flujo mínimo y extendido.")

    train_parser = subparsers.add_parser("train", help="Entrena uno o más experimentos.")
    train_parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Nombre de experimento. Se puede repetir.",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evalúa experimentos entrenados.")
    evaluate_parser.add_argument(
        "--experiment",
        action="append",
        help="Nombre de experimento. Si se omite, intenta evaluar todos.",
    )

    subparsers.add_parser("cae", help="Entrena CAE y genera imágenes CAE.")
    subparsers.add_parser("vae", help="Entrena VAE y genera imágenes VAE.")
    subparsers.add_parser("gan", help="Entrena GAN y genera imágenes GAN.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada principal."""

    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "list"

    if command == "list":
        return list_experiments(args)
    if command == "prepare":
        return run_prepare(args)
    if command == "train":
        return run_train(args, args.experiment)
    if command == "evaluate":
        return run_evaluate(args, args.experiment)
    if command == "cae":
        return run_cae(args)
    if command == "vae":
        return run_vae(args)
    if command == "gan":
        return run_gan(args)
    if command == "minimal":
        return run_minimal(args)
    if command == "extended":
        return run_extended(args)
    if command == "all":
        first = run_minimal(args)
        return first if first else run_extended(args)

    parser.error(f"Comando no reconocido: {command}")
    return 2


def list_experiments(args: argparse.Namespace) -> int:
    """Muestra el catálogo de experimentos sin entrenar."""

    config = load_config(
        args.config,
        args.defaults,
        args.experiments,
        validate=True,
        require_dataset=False,
    )
    print("Experimentos definidos")
    print("-" * 96)
    for experiment in config["experiments"]:
        sources = []
        if experiment.get("use_real"):
            sources.append("real")
        if experiment.get("use_traditional_aug"):
            sources.append("traditional")
        if experiment.get("use_cae"):
            sources.append("cae")
        if experiment.get("use_vae"):
            sources.append("vae")
        if experiment.get("use_gan"):
            sources.append("gan")
        print(f"{experiment['name']:<24} {' + '.join(sources)}")
    return 0


def run_prepare(args: argparse.Namespace) -> int:
    return run_script(
        "scripts/01_prepare_dataset.py",
        [
            "--config",
            args.config,
            "--defaults",
            args.defaults,
            "--experiments",
            args.experiments,
        ],
        dry_run=args.dry_run,
    )


def run_train(args: argparse.Namespace, experiments: list[str]) -> int:
    for experiment in experiments:
        code = run_script(
            "scripts/08_train_multitask.py",
            ["--config", args.config, "--experiment", experiment],
            dry_run=args.dry_run,
        )
        if code:
            return code
    return 0


def run_evaluate(args: argparse.Namespace, experiments: list[str] | None) -> int:
    command_args = ["--config", args.config]
    for experiment in experiments or []:
        command_args.extend(["--experiment", experiment])
    return run_script("scripts/09_evaluate_experiments.py", command_args, dry_run=args.dry_run)


def run_cae(args: argparse.Namespace) -> int:
    first = run_script("scripts/02_train_cae.py", ["--config", args.config], dry_run=args.dry_run)
    if first:
        return first
    return run_script("scripts/03_generate_cae_faces.py", ["--config", args.config], dry_run=args.dry_run)


def run_vae(args: argparse.Namespace) -> int:
    first = run_script("scripts/04_train_vae.py", ["--config", args.config], dry_run=args.dry_run)
    if first:
        return first
    return run_script("scripts/05_generate_vae_faces.py", ["--config", args.config], dry_run=args.dry_run)


def run_gan(args: argparse.Namespace) -> int:
    first = run_script("scripts/06_train_gan.py", ["--config", args.config], dry_run=args.dry_run)
    if first:
        return first
    return run_script("scripts/07_generate_gan_faces.py", ["--config", args.config], dry_run=args.dry_run)


def run_minimal(args: argparse.Namespace) -> int:
    """Ejecuta datos reales, augmentation tradicional y CAE."""

    steps = [
        lambda: run_prepare(args),
        lambda: run_train(args, ["E00_real_only", "E01_real_traditional"]),
        lambda: run_cae(args),
        lambda: run_train(args, ["E02_real_cae"]),
        lambda: run_evaluate(args, ["E00_real_only", "E01_real_traditional", "E02_real_cae"]),
    ]
    return run_steps(steps)


def run_extended(args: argparse.Namespace) -> int:
    """Ejecuta la extensión con VAE, GAN y experimentos combinados."""

    steps = [
        lambda: run_vae(args),
        lambda: run_gan(args),
        lambda: run_train(args, ["E03_real_vae", "E04_real_gan", "E05_all"]),
        lambda: run_evaluate(args, ["E03_real_vae", "E04_real_gan", "E05_all"]),
    ]
    return run_steps(steps)


def run_steps(steps) -> int:
    for step in steps:
        code = step()
        if code:
            return code
    return 0


def run_script(script: str, args: list[str], *, dry_run: bool = False) -> int:
    """Ejecuta un script numerado usando el mismo intérprete de Python."""

    command = [sys.executable, str(PROJECT_ROOT / script), *args]
    printable = " ".join(command)
    if dry_run:
        print(printable)
        return 0
    print(f"\n$ {printable}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

