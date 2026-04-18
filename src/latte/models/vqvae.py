from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_downsamples: int, num_res_blocks: int, res_hidden: int):
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(num_downsamples):
            layers += [nn.Conv2d(c, hidden, 4, stride=2, padding=1), nn.ReLU(inplace=False)]
            c = hidden
        layers += [nn.Conv2d(hidden, hidden, 3, padding=1)]
        for _ in range(num_res_blocks):
            layers += [ResidualBlock(hidden, res_hidden)]
        layers += [nn.ReLU(inplace=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, hidden: int, num_upsamples: int, num_res_blocks: int, res_hidden: int):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_res_blocks):
            layers += [ResidualBlock(hidden, res_hidden)]
        layers += [nn.ReLU(inplace=False)]
        for _ in range(num_upsamples - 1):
            layers += [nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1), nn.ReLU(inplace=False)]
        layers += [nn.ConvTranspose2d(hidden, out_channels, 4, stride=2, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment = commitment
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)
        d = (flat.pow(2).sum(dim=1, keepdim=True)
             - 2 * flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(dim=1))
        indices = d.argmin(dim=1)
        quantized = self.embedding(indices).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        loss = codebook_loss + self.commitment * commitment_loss

        quantized_st = z + (quantized - z).detach()
        return quantized_st, loss, indices.view(B, H, W)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        q, _, _ = self.forward(z)
        return q


class VQVAE(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, num_downsamples: int = 2,
                 num_res_blocks: int = 2, res_hidden: int = 32,
                 codebook_size: int = 512, commitment: float = 0.25):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden, num_downsamples, num_res_blocks, res_hidden)
        self.pre_quant = nn.Conv2d(hidden, hidden, 1)
        self.quantizer = VectorQuantizer(codebook_size, hidden, commitment=commitment)
        self.decoder = Decoder(in_channels, hidden, num_downsamples, num_res_blocks, res_hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.pre_quant(h)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        z_q, vq_loss, _ = self.quantizer(z)
        x_rec = self.decoder(z_q)
        return x_rec, vq_loss


def build_vqvae(dataset: str) -> VQVAE:
    name = dataset.lower()
    if name in {'mnist', 'fashionmnist'}:
        return VQVAE(in_channels=1, hidden=64, num_downsamples=2, num_res_blocks=2, res_hidden=32,
                     codebook_size=256)
    if name in {'svhn', 'cifar10'}:
        return VQVAE(in_channels=3, hidden=128, num_downsamples=2, num_res_blocks=2, res_hidden=32,
                     codebook_size=512)
    if name == 'imagenet':
        return VQVAE(in_channels=3, hidden=256, num_downsamples=3, num_res_blocks=3, res_hidden=64,
                     codebook_size=1024)
    raise ValueError(f'Unsupported dataset for VQ-VAE: {dataset}')
