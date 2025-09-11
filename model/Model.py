import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
from torch.nn import modules
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
class Resnet50_Model_imagenet(nn.Module):
    def __init__(self, num_classes=50):
        super(Resnet50_Model_imagenet, self).__init__()
        # Load pre-trained ResNet50 model
        base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Modify the fully connected layer to have 50 output classes
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

from torchvision.models import VGG19_Weights

class Vgg19_Model_imagenet(nn.Module):
    def __init__(self, num_classes=50):
        super(Vgg19_Model_imagenet, self).__init__()

        base_model = models.vgg19(weights=VGG19_Weights.DEFAULT)

        self.features = base_model.features

        self.classifier = nn.Sequential(
            *list(base_model.classifier.children())[:-1],
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  
        x = self.classifier(x)
        return x


import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

class Vgg16_Model_cifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(Vgg16_Model_cifar10, self).__init__()

        base_model = models.vgg16(pretrained=True)

        base_model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),  
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Resnet18_Model_cifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet18_Model_cifar10, self).__init__()

        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()  
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, 10)

    def add_noise(self, x, noise_level=0.05):

        noise = noise_level * torch.randn_like(x)

        noisy_x = x + noise
        return noisy_x

    def forward(self, x):
        # x = self.add_noise(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Lenet5_Model_mnist(nn.Module):
    def __init__(self):
        super(Lenet5_Model_mnist, self).__init__()
        # First convolutional layer with 1 input channel (grayscale image) and 20 output channels
        self.conv1 = nn.Conv2d(1, 6, 5)  

        self.avgpool1 = nn.AvgPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.avgpool2 = nn.AvgPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, 4)

        self.fc1 = nn.Linear(120, 84)

        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入层
        x = F.relu(self.conv1(x))  
        x = self.avgpool1(x)  
        x = F.relu(self.conv2(x))  
        x = self.avgpool2(x)  
        x = F.relu(self.conv3(x))  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))  
        class_logits = self.fc2(x)  
        return class_logits


class Lenet4_Model_mnist(nn.Module):
    def __init__(self):
        super(Lenet4_Model_mnist, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)

        self.avgpool1 = nn.AvgPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.avgpool2 = nn.AvgPool2d(2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 84)

        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.avgpool1(x)  
        x = F.relu(self.conv2(x))  
        x = self.avgpool2(x)  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x)) 
        class_logits = self.fc2(x)  
        return class_logits



class AllCNNA_Model_cifar10(nn.Module):
    """All-CNN-A for CIFAR-10 (3x32x32)"""
    def __init__(self, num_classes=10, dropout_p=0.5):
        super(AllCNNA_Model_cifar10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,  96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,  96, kernel_size=3, stride=2, padding=1, bias=False),  # downsample

            nn.Dropout(p=dropout_p),

            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192, kernel_size=3, stride=2, padding=1, bias=False),  # downsample

            nn.Dropout(p=dropout_p),

            nn.Conv2d(192,192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)          # (B, C=num_classes, H, W)
        x = self.pool(x)              # (B, C, 1, 1)
        x = torch.flatten(x, 1)       # (B, C)
        return x


class AllCNNB_Model_cifar10(nn.Module):
    """All-CNN-B for CIFAR-10; a slightly wider variant."""
    def __init__(self, num_classes=10, dropout_p=0.5):
        super(AllCNNB_Model_cifar10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # downsample

            nn.Dropout(p=dropout_p),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # downsample

            nn.Dropout(p=dropout_p),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class Custom_Model_1_6B(nn.Module):
    """
    Scalable custom CNN backbone. Increase width_mult and num_blocks to grow params.
    The class name is a label; actual params depend on config.
    """
    def __init__(self, num_classes=1000, width_mult=3.0, num_blocks=(4, 6, 8, 6)):
        super(Custom_Model_1_6B, self).__init__()
        c1 = int(64  * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)

        layers = []
        # stem
        layers += [_ConvBlock(3, c1, k=3, s=1, p=1)]
        # stage 1
        layers += [_ConvBlock(c1, c1, k=3, s=2, p=1)]
        for _ in range(num_blocks[0]-1):
            layers += [_ConvBlock(c1, c1)]
        # stage 2
        layers += [_ConvBlock(c1, c2, k=3, s=2, p=1)]
        for _ in range(num_blocks[1]-1):
            layers += [_ConvBlock(c2, c2)]
        # stage 3
        layers += [_ConvBlock(c2, c3, k=3, s=2, p=1)]
        for _ in range(num_blocks[2]-1):
            layers += [_ConvBlock(c3, c3)]
        # stage 4
        layers += [_ConvBlock(c3, c4, k=3, s=2, p=1)]
        for _ in range(num_blocks[3]-1):
            layers += [_ConvBlock(c4, c4)]

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Custom_Model_3_3B(nn.Module):
    """
    A wider/deeper variant of the custom backbone.
    Increase width_mult / blocks further to grow parameter count.
    """
    def __init__(self, num_classes=1000, width_mult=4.0, num_blocks=(6, 8, 12, 8)):
        super(Custom_Model_3_3B, self).__init__()
        c1 = int(96   * width_mult)
        c2 = int(192  * width_mult)
        c3 = int(384  * width_mult)
        c4 = int(768  * width_mult)

        layers = []
        # stem
        layers += [_ConvBlock(3, c1, k=3, s=1, p=1)]
        # stage 1
        layers += [_ConvBlock(c1, c1, k=3, s=2, p=1)]
        for _ in range(num_blocks[0]-1):
            layers += [_ConvBlock(c1, c1)]
        # stage 2
        layers += [_ConvBlock(c1, c2, k=3, s=2, p=1)]
        for _ in range(num_blocks[1]-1):
            layers += [_ConvBlock(c2, c2)]
        # stage 3
        layers += [_ConvBlock(c2, c3, k=3, s=2, p=1)]
        for _ in range(num_blocks[2]-1):
            layers += [_ConvBlock(c3, c3)]
        # stage 4
        layers += [_ConvBlock(c3, c4, k=3, s=2, p=1)]
        for _ in range(num_blocks[3]-1):
            layers += [_ConvBlock(c4, c4)]

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
