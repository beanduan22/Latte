from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Subset


_DATASET_META: Dict[str, Dict] = {
    'mnist':        {'num_classes': 10,  'channels': 1, 'size': 28},
    'fashionmnist': {'num_classes': 10,  'channels': 1, 'size': 28},
    'svhn':         {'num_classes': 10,  'channels': 3, 'size': 32},
    'cifar10':      {'num_classes': 10,  'channels': 3, 'size': 32},
    'imagenet':     {'num_classes': 1000,'channels': 3, 'size': 224},
}


def dataset_meta(name: str) -> Dict:
    return _DATASET_META[name.lower()]


def _tv():
    from torchvision import datasets, transforms
    return datasets, transforms


def build_transforms(name: str, normalization: str, image_size: Optional[int]):
    _, transforms = _tv()
    name = name.lower()
    meta = dataset_meta(name)
    size = image_size or meta['size']

    if normalization == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalization == 'half':
        mean = [0.5] * meta['channels']
        std = [0.5] * meta['channels']
    else:
        mean = None
        std = None

    def compose(tf_list):
        if mean is not None:
            tf_list = tf_list + [transforms.Normalize(mean=mean, std=std)]
        return transforms.Compose(tf_list)

    base_train = [transforms.Resize((size, size)), transforms.ToTensor()]
    base_test = [transforms.Resize((size, size)), transforms.ToTensor()]

    if name == 'imagenet':
        train_tf = compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_tf = compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
    else:
        if name in {'cifar10', 'svhn'}:
            base_train = [transforms.Resize((size, size)), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        train_tf = compose(base_train)
        test_tf = compose(base_test)

    return train_tf, test_tf


def build_datasets(name: str, root: str, normalization: str = 'half', image_size: Optional[int] = None):
    datasets, _ = _tv()
    name = name.lower()
    train_tf, test_tf = build_transforms(name, normalization=normalization, image_size=image_size)
    if name == 'mnist':
        train_ds = datasets.MNIST(root=root, train=True, download=True, transform=train_tf)
        test_ds = datasets.MNIST(root=root, train=False, download=True, transform=test_tf)
    elif name == 'fashionmnist':
        train_ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=train_tf)
        test_ds = datasets.FashionMNIST(root=root, train=False, download=True, transform=test_tf)
    elif name == 'svhn':
        train_ds = datasets.SVHN(root=root, split='train', download=True, transform=train_tf)
        test_ds = datasets.SVHN(root=root, split='test', download=True, transform=test_tf)
    elif name == 'cifar10':
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
    elif name == 'imagenet':
        train_ds = datasets.ImageFolder(root=f'{root}/train', transform=train_tf)
        test_ds = datasets.ImageFolder(root=f'{root}/val', transform=test_tf)
    else:
        raise ValueError(f'Unsupported dataset: {name}')
    return train_ds, test_ds


def build_loaders(name: str, root: str, batch_size: int, num_workers: int = 4,
                  normalization: str = 'half', image_size: Optional[int] = None):
    train_ds, test_ds = build_datasets(name, root, normalization=normalization, image_size=image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def _label_of(dataset, idx: int) -> int:
    x, y = dataset[idx]
    return int(y)


def group_indices_by_class(dataset, num_classes: int, limit: Optional[int] = None) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    n = len(dataset)
    if limit is not None:
        n = min(n, limit)
    for i in range(n):
        c = _label_of(dataset, i)
        if 0 <= c < num_classes:
            buckets[c].append(i)
    return buckets


@torch.no_grad()
def select_correctly_classified_seeds(model, dataset, device, num_seeds: int,
                                      per_class_cap: Optional[int] = None) -> List[int]:
    model.eval()
    indices: List[int] = []
    counts: Dict[int, int] = {}
    for i in range(len(dataset)):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        pred = int(model(x).argmax(dim=1).item())
        if pred == int(y):
            if per_class_cap is not None:
                c = int(y)
                if counts.get(c, 0) >= per_class_cap:
                    continue
                counts[c] = counts.get(c, 0) + 1
            indices.append(i)
        if len(indices) >= num_seeds:
            break
    return indices


@torch.no_grad()
def select_agreement_seeds(model_a, model_b, dataset, device, num_seeds: int) -> List[int]:
    model_a.eval()
    model_b.eval()
    indices: List[int] = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        pa = int(model_a(x).argmax(dim=1).item())
        pb = int(model_b(x).argmax(dim=1).item())
        if pa == pb == int(y):
            indices.append(i)
        if len(indices) >= num_seeds:
            break
    return indices
