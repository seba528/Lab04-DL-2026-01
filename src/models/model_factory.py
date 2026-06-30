"""Fábrica de modelos desde YAML."""

from __future__ import annotations

from src.models.convolutional_autoencoder import ConvolutionalAutoencoder
from src.models.gan import DCGANDiscriminator, DCGANGenerator
from src.models.multitask_cnn import MultiTaskCNN
from src.models.variational_autoencoder import VariationalAutoencoder


def input_channels(config: dict) -> int:
    return 1 if config["preprocessing"]["color_space"] == "GRAYSCALE" else 3


def build_multitask_model(config: dict) -> MultiTaskCNN:
    model_config = config["multitask_model"]
    return MultiTaskCNN(
        in_channels=input_channels(config),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        gender_output_classes=int(model_config["gender_output_classes"]),
        age_output_dim=int(model_config["age_output_dim"]),
        age_mode=config["age"]["prediction_mode"],
    )


def build_cae(config: dict) -> ConvolutionalAutoencoder:
    return ConvolutionalAutoencoder(
        in_channels=input_channels(config),
        latent_dim=int(config["cae"]["latent_dim"]),
    )


def build_vae(config: dict) -> VariationalAutoencoder:
    return VariationalAutoencoder(
        in_channels=input_channels(config),
        latent_dim=int(config["vae"]["latent_dim"]),
    )


def build_gan(config: dict) -> tuple[DCGANGenerator, DCGANDiscriminator]:
    channels = input_channels(config)
    generator = DCGANGenerator(int(config["gan"]["latent_dim"]), channels)
    discriminator = DCGANDiscriminator(channels)
    return generator, discriminator

