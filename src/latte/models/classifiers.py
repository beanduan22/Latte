from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet4(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=False),
            nn.Linear(120, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 4),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(inplace=False),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CustomFashion(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 64):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=False),
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=False),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=False),
            nn.Conv2d(c3, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class AllCNN(nn.Module):
    def __init__(self, num_classes: int = 10, variant: str = 'A'):
        super().__init__()
        c = {'A': 96, 'B': 128}[variant.upper()]
        self.features = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(c, c * 2, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(c * 2, c * 2, 3, stride=2, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(c * 2, c * 2, 1), nn.ReLU(inplace=False),
            nn.Conv2d(c * 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return F.adaptive_avg_pool2d(h, 1).flatten(1)


def build_classifier(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == 'lenet4':
        return LeNet4(num_classes=num_classes)
    if name == 'lenet5':
        return LeNet5(num_classes=num_classes)
    if name == 'custom1':
        return CustomFashion(num_classes=num_classes, width=48)
    if name == 'custom2':
        return CustomFashion(num_classes=num_classes, width=72)
    if name == 'allcnna':
        return AllCNN(num_classes=num_classes, variant='A')
    if name == 'allcnnb':
        return AllCNN(num_classes=num_classes, variant='B')

    from torchvision import models
    if name == 'vgg16':
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.vgg16(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == 'vgg19':
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.vgg19(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f'Unsupported classifier: {name}')
