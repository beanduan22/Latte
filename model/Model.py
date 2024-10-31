import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
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
        # 加载预训练的 VGG19 模型
        base_model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        # 使用 VGG19 的特征部分（卷积层）
        self.features = base_model.features
        # 修改分类器，使其输出 50 个类别
        # 复制原始分类器，替换最后一层
        self.classifier = nn.Sequential(
            *list(base_model.classifier.children())[:-1],
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平张量
        x = self.classifier(x)
        return x


import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

class Vgg16_Model_cifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(Vgg16_Model_cifar10, self).__init__()
        # 加载预训练的 VGG16 模型
        base_model = models.vgg16(pretrained=True)
        # CIFAR-10 图片大小为 32x32，VGG16 预期输入为 224x224，需要进行调整

        # 修改第一层的卷积层，保持输入通道数为 3，输出通道数为 64，但调整核大小和步幅
        base_model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # 移除最后的全连接层，替换为适合 CIFAR-10 的分类器
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化到 1x1
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),  # 输入特征维度根据池化输出调整
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
        # 使用最新的推荐方式加载预训练权重
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # 修改第一个卷积层
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()  # 移除最大池化层
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features, 10)

    def add_noise(self, x, noise_level=0.05):
        # 生成和输入相同形状的随机噪声
        noise = noise_level * torch.randn_like(x)
        # 将噪声添加到原始输入
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
        self.conv1 = nn.Conv2d(1, 6, 5)  # 输入通道数为1（灰度图），输出通道数为6，卷积核大小为5
        # S2: 平均池化层
        self.avgpool1 = nn.AvgPool2d(2, stride=2)
        # C3: 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # S4: 平均池化层
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        # C5: 卷积层
        self.conv3 = nn.Conv2d(16, 120, 4)
        # F6: 全连接层
        self.fc1 = nn.Linear(120, 84)
        # 输出层
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入层
        x = F.relu(self.conv1(x))  # 输出形状：(batch_size, 6, 28, 28)
        x = self.avgpool1(x)  # 输出形状：(batch_size, 6, 14, 14)
        x = F.relu(self.conv2(x))  # 输出形状：(batch_size, 16, 10, 10)
        x = self.avgpool2(x)  # 输出形状：(batch_size, 16, 5, 5)
        x = F.relu(self.conv3(x))  # 输出形状：(batch_size, 120, 1, 1)
        x = x.view(x.size(0), -1)  # 展平，输出形状：(batch_size, 120)
        x = F.relu(self.fc1(x))  # 输出形状：(batch_size, 84)
        class_logits = self.fc2(x)  # 输出层
        return class_logits


class Lenet4_Model_mnist(nn.Module):
    def __init__(self):
        super(Lenet4_Model_mnist, self).__init__()
        # 第一层卷积层，输入通道数为1（灰度图），输出通道数为6，卷积核大小为5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # S2: 平均池化层
        self.avgpool1 = nn.AvgPool2d(2, stride=2)
        # C3: 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # S4: 平均池化层
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        # F5: 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 84)
        # 输出层
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 输出形状: (batch_size, 6, 28, 28)
        x = self.avgpool1(x)  # 输出形状: (batch_size, 6, 14, 14)
        x = F.relu(self.conv2(x))  # 输出形状: (batch_size, 16, 10, 10)
        x = self.avgpool2(x)  # 输出形状: (batch_size, 16, 5, 5)
        x = x.view(x.size(0), -1)  # 展平，输出形状: (batch_size, 16*5*5)
        x = F.relu(self.fc1(x))  # 输出形状: (batch_size, 84)
        class_logits = self.fc2(x)  # 输出层
        return class_logits
