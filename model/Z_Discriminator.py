import torch.nn as nn

class LatentDiscriminator(nn.Module):
    def __init__(self):
        super(LatentDiscriminator, self).__init__()
        # 将输入大小改为 8192，以适应展平后的 (64, 512, 4, 4) 输入
        self.model = nn.Sequential(
            nn.Linear(128 * 28 * 28, 128),  # 8192 -> 256
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),          # 256 -> 128
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),            # 128 -> 1
            nn.Sigmoid()                  # 最后使用 Sigmoid 激活函数
        )

    def forward(self, x):
        # 展平输入 (batch_size, 512, 4, 4) -> (batch_size, 8192)
        x = x.view(x.size(0), -1)
        return self.model(x)
