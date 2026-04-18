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

Each experiment: train the VQ-VAE once, train the classifier(s), run LATTE, evaluate.

### MNIST single-model

```bash
python train_vqvae.py --config configs/mnist_vqvae.yaml
python train_classifier.py --config configs/mnist_lenet5_single.yaml --target a
python run_latte.py --config configs/mnist_lenet5_single.yaml
python evaluate_results.py --config configs/mnist_lenet5_single.yaml \
  --failures results/mnist_lenet5_single/failures_single.pt
```

Replace `lenet5` with `lenet4` via `configs/mnist_lenet4_single.yaml`.

### CIFAR-10 single-model

```bash
python train_vqvae.py --config configs/cifar10_vqvae.yaml
python train_classifier.py --config configs/cifar10_vgg16_single.yaml --target a
python run_latte.py --config configs/cifar10_vgg16_single.yaml
python evaluate_results.py --config configs/cifar10_vgg16_single.yaml \
  --failures results/cifar10_vgg16_single/failures_single.pt
```

Replace `vgg16` with `resnet18` via `configs/cifar10_resnet18_single.yaml`.

### ImageNet single-model

```bash
python train_vqvae.py --config configs/imagenet_vqvae.yaml
python train_classifier.py --config configs/imagenet_vgg19_single.yaml --target a
python run_latte.py --config configs/imagenet_vgg19_single.yaml
python evaluate_results.py --config configs/imagenet_vgg19_single.yaml \
  --failures results/imagenet_vgg19_single/failures_single.pt
```

Replace `vgg19` with `resnet50` via `configs/imagenet_resnet50_single.yaml`.

### Multi-model

```bash
python train_vqvae.py --config configs/mnist_vqvae.yaml
python train_classifier.py --config configs/mnist_multi.yaml --target a
python train_classifier.py --config configs/mnist_multi.yaml --target b
python run_latte.py --config configs/mnist_multi.yaml
python evaluate_results.py --config configs/mnist_multi.yaml \
  --failures results/mnist_multi/failures_multi.pt
```

Replace `mnist` with `fashionmnist` (use `configs/fashionmnist_multi.yaml` and `configs/fashionmnist_vqvae.yaml`) or `svhn` (use `configs/svhn_multi.yaml` and `configs/svhn_vqvae.yaml`).
