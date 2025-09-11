# LATTE

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Latent-space, anchor-guided test generation and evaluation for image classifiers.
> **Single-model** and **Multi-model** testing on MNIST, CIFAR-10, ImageNet, FashionMNIST, and SVHN.

---

## Table of Contents

* [Overview](#overview)
* [Requirements](#requirements)
* [Quickstart](#quickstart)
* [Datasets](#datasets)
* [Run: Single-Model](#run-single-model)
* [Run: Multi-Model (Differential)](#run-multi-model-differential)
* [Configuration](#configuration)
* [Expected Outputs](#expected-outputs)

---

## Overview

**LATTE** provides a clean, modular pipeline to:

* train baseline DNN classifiers,
* train **VQ-VAE** generators used for latent-space test case synthesis,
* evaluate models under **single-model oracles** and **multi-model (differential) oracles**, and
* log results, checkpoints, and visualizations automatically.

**Supported settings**

* **Single-model**: MNIST, CIFAR-10, ImageNet
* **Multi-model**: MNIST (LeNet-4 vs LeNet-5), FashionMNIST (Custom-1.6B vs Custom-3.3B), SVHN (All-CNN-A vs All-CNN-B)


---

## Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Quickstart

1. **Clone**

```bash
git clone <repository_url>
cd <project_name>
```

2. **Prepare folders**

```bash
mkdir -p data output/log
```

3. **Run (defaults in `main.py`)**

```bash
python main.py
```

Logs/checkpoints go to `output/…` (paths are auto-created).

---

## Datasets

All datasets are auto-downloaded to `./data` when first used:

* **MNIST** (28×28 gray), **FashionMNIST** (28×28 gray)
* **CIFAR-10** (32×32 RGB), **SVHN** (32×32 RGB)
* **ImageNet** (supply your own path or loader; see comments in `prepare_data_*` and `Model.py`)

---

## Run: Single-Model

Supported datasets: **MNIST**, **CIFAR-10**, **ImageNet**.

In `main.py`, choose a model name (e.g., `cifar10_resnet8_encoder_decoder`, `imagenet_resnet50_encoder_decoder`, etc.) and run:

```bash
python main.py
```

This will:

* train the selected **classifier**,
* train the paired **VQ-VAE** generator,
* evaluate with single-model oracles,
* save logs & checkpoints under `output/<model_tag>/…`.

---

## Run: Multi-Model (Differential)

Supported pairs:

| Dataset          | Model A     | Model B     | VQ-VAE (shared)             |
| ---------------- | ----------- | ----------- | --------------------------- |
| **MNIST**        | LeNet-4     | LeNet-5     | `Lenet5_VQVAE_mnist`        |
| **FashionMNIST** | Custom-1.6B | Custom-3.3B | `CUSTOM_VQVAE_fashionmnist` |
| **SVHN**         | All-CNN-A   | All-CNN-B   | `ALLCNN_VQVAE_svhn`       |

In `main.py`, set:

```python
DATASET = "mnist"          # or "fashionmnist" / "svhn"
```

Then run:

```bash
python main.py
```

What happens:

* trains **Model A** and **Model B** for the chosen dataset,
* trains the dataset-appropriate **VQ-VAE**,
* runs a **differential test** reporting the *disagreement rate* (where the two models predict different labels).

---

## Configuration

Key knobs in `main.py`:

```python
# Dataset selector for multi-model
DATASET = "mnist"  # "mnist" | "fashionmnist" | "svhn"

# Batching / epochs
train_batch_size = 64
test_batch_size  = 256
pre_train_epoch = 10
vqvae_train_epoch = 30

# Optim / loss weights
learning_rate = 1e-3
dec_loss_weight_ = 2.0   # recon loss weight (VQ-VAE)
vq_loss_weight_  = 1.0   # VQ loss weight (VQ-VAE)

# Optional adversarial/generator phase
USE_ADVERSARIAL_PHASE = False
```

Model wiring for each dataset lives in a helper like:

```python
model_a, model_b, vqvae, save_path = build_models_for(DATASET, num_classes)
```

Feel free to swap in your own backbones or adjust capacities (e.g., width multipliers in `Custom_Model_*`).

---

## Expected Outputs

* **Logs**: `output/log/…` and per-experiment subdirs under `output/<tag>/…`
* **Classifier checkpoints**: `output/<tag>/modelA/dnn_model.pth`, `output/<tag>/modelB/dnn_model.pth`
* **VQ-VAE checkpoint**: `output/<tag>/vqvae/vqvae_model.pth`
* **Metrics**:

  * Single-model: NoF, DoF, FID, FSR, GE
  * Multi-model: NoF, FC, CS, FID, GE

---

