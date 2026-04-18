# LATTE

## Install

```bash
conda create -n latte python=3.10 -y
conda activate latte
pip install -r requirements.txt
pip install torchvision
pip install -e .
```

## Datasets

MNIST, FashionMNIST, SVHN, CIFAR-10 download automatically to `./data` on first use.

ImageNet must be provided manually in `./imagenet/`:

```
imagenet/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

Download from https://image-net.org/ after registering and arrange `val/` into class folders with the official devkit.

## Run

Every experiment needs a trained VQ-VAE, at least one trained classifier, then a LATTE run.

### MNIST single-model (LeNet-5)

```bash
python train_vqvae.py --config configs/mnist_vqvae.yaml
python train_classifier.py --config configs/mnist_lenet5_single.yaml --target a
python run_latte.py --config configs/mnist_lenet5_single.yaml
python evaluate_results.py --config configs/mnist_lenet5_single.yaml \
  --failures results/mnist_lenet5_single/failures_single.pt
```

Swap `lenet5` for `lenet4` with `configs/mnist_lenet4_single.yaml`.

### CIFAR-10 single-model

```bash
python train_vqvae.py --config configs/cifar10_vqvae.yaml
python train_classifier.py --config configs/cifar10_vgg16_single.yaml --target a
python run_latte.py --config configs/cifar10_vgg16_single.yaml
python evaluate_results.py --config configs/cifar10_vgg16_single.yaml \
  --failures results/cifar10_vgg16_single/failures_single.pt
```

Swap `vgg16` for `resnet18` with `configs/cifar10_resnet18_single.yaml`.

### ImageNet single-model

```bash
python train_vqvae.py --config configs/imagenet_vqvae.yaml
python train_classifier.py --config configs/imagenet_vgg19_single.yaml --target a
python run_latte.py --config configs/imagenet_vgg19_single.yaml
python evaluate_results.py --config configs/imagenet_vgg19_single.yaml \
  --failures results/imagenet_vgg19_single/failures_single.pt
```

Swap `vgg19` for `resnet50` with `configs/imagenet_resnet50_single.yaml`.

### Multi-model (differential) testing

```bash
python train_vqvae.py --config configs/mnist_vqvae.yaml
python train_classifier.py --config configs/mnist_multi.yaml --target a
python train_classifier.py --config configs/mnist_multi.yaml --target b
python run_latte.py --config configs/mnist_multi.yaml
python evaluate_results.py --config configs/mnist_multi.yaml \
  --failures results/mnist_multi/failures_multi.pt
```

Replace `mnist` with `fashionmnist` or `svhn` using the matching config. The model pairs are:

| Dataset       | Model A     | Model B     |
|---------------|-------------|-------------|
| mnist         | LeNet-4     | LeNet-5     |
| fashionmnist  | Custom-1.6B | Custom-3.3B |
| svhn          | All-CNN-A   | All-CNN-B   |

## Semantic drift (DINOv2)

`evaluate_results.py` computes semantic drift via DINOv2 on a sample of failures. It downloads weights via `torch.hub`; no extra install is required beyond the default `pip install -e .`.
