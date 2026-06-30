# LAB04-DL-2026-01

Reconocimiento multitarea de género y edad con aumento de datos tradicional, autoencoders, VAE y GAN usando UTKFace.

## Objetivo

Este laboratorio extiende la lógica del LAB03 para comparar un modelo CNN multitarea entrenado con imágenes reales y distintas fuentes de aumento de datos. La tarea de género se aborda como clasificación y la tarea de edad como regresión por defecto.

## Datos originales

Las imágenes originales de UTKFace se cargan desde la ruta configurable:

```yaml
paths:
  original_utkface_dir: data/raw/UTKFace
```

Esa carpeta es la única fuente de imágenes originales y debe tratarse como entrada de solo lectura. Si el dataset está en otra ubicación, cambie solo `paths.original_utkface_dir` en `config/path.yaml`.

## Imágenes generadas

Las imágenes sintéticas se guardan localmente dentro del proyecto:

```text
data/generated/cae/
data/generated/vae/
data/generated/gan/
```

Sus metadatos se guardan en `data/metadata/` con la columna `source`, para distinguir imágenes `real`, `cae`, `vae` y `gan`.

## Instalación

Con Conda:

```bash
conda env create -f environment.yml
conda activate lab04-dl-2026-01
```

Con `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Orquestador principal

El archivo `main.py` es el punto de entrada principal del laboratorio. Internamente llama a los scripts numerados para mantener el flujo transparente.

```bash
python3 main.py list
python3 main.py prepare
python3 main.py train --experiment E00_real_only
python3 main.py evaluate --experiment E00_real_only
```

Para revisar el flujo sin ejecutarlo:

```bash
python3 main.py --dry-run minimal
python3 main.py --dry-run extended
```

## Flujo mínimo

```bash
python3 main.py minimal
```

El flujo mínimo ejecuta, en orden, los comandos equivalentes a:

```bash
python3 scripts/01_prepare_dataset.py --config config/path.yaml
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E00_real_only
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E01_real_traditional
python3 scripts/02_train_cae.py --config config/path.yaml
python3 scripts/03_generate_cae_faces.py --config config/path.yaml
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E02_real_cae
python3 scripts/09_evaluate_experiments.py --config config/path.yaml
```

## Flujo extendido

```bash
python3 main.py extended
```

El flujo extendido ejecuta, en orden, los comandos equivalentes a:

```bash
python3 scripts/04_train_vae.py --config config/path.yaml
python3 scripts/05_generate_vae_faces.py --config config/path.yaml
python3 scripts/06_train_gan.py --config config/path.yaml
python3 scripts/07_generate_gan_faces.py --config config/path.yaml
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E03_real_vae
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E04_real_gan
python3 scripts/08_train_multitask.py --config config/path.yaml --experiment E05_all
python3 scripts/09_evaluate_experiments.py --config config/path.yaml
```

## Experimentos

Los experimentos están definidos en `config/experiments.yaml`:

- `E00_real_only`: imágenes reales.
- `E01_real_traditional`: imágenes reales con aumento tradicional.
- `E02_real_cae`: imágenes reales más imágenes CAE.
- `E03_real_vae`: imágenes reales más imágenes VAE.
- `E04_real_gan`: imágenes reales más imágenes GAN.
- `E05_all`: combinación completa.

Validación y prueba usan solo imágenes reales. Las imágenes generadas se mezclan únicamente con el conjunto de entrenamiento.

## Métricas

El laboratorio reporta métricas separadas por tarea:

- Género: accuracy, precision, recall y F1.
- Edad: MAE y RMSE.
- Global: pérdida total, pérdida de género y pérdida de edad.

## Material docente

El archivo `presentation/Laboratorio_04_Guia_Uso_Codigo.tex` contiene una presentación Beamer didáctica para guiar el uso del laboratorio en servidor, explicar los YAML y revisar los comandos principales.
