
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple

__all__ = [
    'VQVAE',
    'Encoder',
    'Decoder',
    'ResBlock',
    'SubSampleBlock',
    'SubsampleTransposeBlock',
    'VectorQuantizer'
]

class ResBlock(nn.Module):
    def __init__(self, in_channel:int, hid_channel:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channel,
                hid_channel,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                hid_channel,
                in_channel,
                kernel_size=1
            ),
        )

    def forward(self, input:Tensor) -> Tensor:
        out = self.conv(input)
        out += input
        return out

class SubSampleBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, num_layers:int):
        """
        Downsamples the input image by a factor of 2^num_layers
        """
        super().__init__()
        layers = []
        current_channel = in_channel
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(
                    current_channel,
                    out_channel,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.ReLU()
            ])
            current_channel = out_channel
        self.blocks = nn.Sequential(*layers)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)

class SubsampleTransposeBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, num_layers:int):
        """
        Upsamples the input image by a factor of 2^num_layers
        """
        super().__init__()
        layers = []
        current_channel = in_channel
        for _ in range(num_layers):
            layers.extend([
                nn.ConvTranspose2d(
                    current_channel,
                    out_channel,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.ReLU()
            ])
            current_channel = out_channel
        self.blocks = nn.Sequential(*layers)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)

class Encoder(nn.Module):
    def __init__(
            self,
            in_channel:int,
            hid_channel:int,
            n_res_block:int,
            n_res_channel:int,
            num_downsamples:int
        ):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, hid_channel, kernel_size=4, stride=2, padding=1)]
        for _ in range(num_downsamples - 1):
            blocks.append(SubSampleBlock(hid_channel, hid_channel, num_layers=1))
        for _ in range(n_res_block):
            blocks.append(ResBlock(hid_channel, n_res_channel))
        blocks.append(nn.ReLU())
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)

class Decoder(nn.Module):
    def __init__(
        self,
        hid_channel:int,
        out_channel:int,
        n_res_block:int,
        n_res_channel:int,
        num_upsamples:int
    ):
        super().__init__()
        blocks = []
        for _ in range(n_res_block):
            blocks.append(ResBlock(hid_channel, n_res_channel))
        blocks.append(nn.ReLU())
        for _ in range(num_upsamples - 1):
            blocks.append(SubsampleTransposeBlock(hid_channel, hid_channel, num_layers=1))
        blocks.append(nn.ConvTranspose2d(hid_channel, out_channel, kernel_size=4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input:Tensor) -> Tensor:
        return self.blocks(input)


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int, beta: float = 0.25):
        """
        VectorQuantizer with loss and perplexity calculation.

        Args:
            n_embed (int): Number of embedding vectors.
            embed_dim (int): Dimension of each embedding vector.
            beta (float): Commitment loss scaling factor.
        """
        super().__init__()
        self.n_embed = n_embed#256
        self.embed_dim = embed_dim#8
        self.beta = beta  # 权重用于控制嵌入损失的比例
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1 / n_embed, 1 / n_embed)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, float, Tensor]:
        """
        Forward pass through the quantizer. It returns quantized output, indices of embeddings,
        loss, and perplexity.

        Args:
            z (Tensor): Input feature map from encoder. Shape: (B, C, H, W)

        Returns:
            z_q (Tensor): Quantized tensor of same shape as z.
            loss (Tensor): Quantization loss (commitment loss + embedding loss).
            perplexity (float): Perplexity of the quantized representations.
            indices (Tensor): Indices of closest embeddings. Shape: (B*H*W)
        """
        # 1. Reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embed_dim)  # (B*H*W, C)

        # 2. Compute distances from z to embedding vectors (z - e)^2 = z^2 + e^2 - 2 * e * z
        distances = (
                torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
                torch.sum(self.embedding.weight ** 2, dim=1) -
                2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )  # (B*H*W, n_embed)

        # 3. Find closest embeddings (minimum distance)
        encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W)
        encodings = torch.zeros(encoding_indices.size(0), self.n_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)  # One-hot encoding

        # 4. Quantize the latent vector (z -> e)
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)  # (B, H, W, C)

        # 5. Compute the loss (commitment loss + embedding loss)
        commitment_loss = F.mse_loss(z_q.detach(), z)
        embedding_loss = F.mse_loss(z_q, z.detach())
        loss = embedding_loss + self.beta * commitment_loss

        # 6. Preserve gradients through straight-through estimator
        z_q = z + (z_q - z).detach()

        # 7. Compute perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Reshape z_q back to original (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, perplexity, encoding_indices


class Resnet18_VQVAE_cifar10(nn.Module):
    def __init__(
        self,
        in_channel: int = 3,
        hid_channel: int = 128,
        n_res_block: int = 20,#12
        n_res_channel: int = 64,
        embed_dim: int = 8,#16
        n_embed: int = 128,
        num_downsamples: int = 3,
        beta: float = 0.25  # 增加 beta 参数用于量化器
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channel,
            hid_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )
        self.enc_to_quant = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = VectorQuantizer(
            n_embed=n_embed,
            embed_dim=embed_dim,
            beta=beta  # 将 beta 传递给 VectorQuantizer
        )
        self.quant_to_dec = nn.Conv2d(embed_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        # Encoder
        z_e = self.encoder(input)  # (B, hid_channel, H', W')
        #z_e = self.enc_to_quant(z_e)  # (B, embed_dim, H', W')

        # Vector Quantization
        z_q, loss, perplexity, indices = self.quantize(z_e)  # (B, embed_dim, H', W')

        # Decoder
        #z_q = self.quant_to_dec(z_q)  # (B, hid_channel, H', W')
        x_recon = self.decoder(z_q)  # (B, in_channel, H, W)

        return x_recon, z_e, loss, perplexity, indices

class Vgg16_VQVAE_cifar10(nn.Module):
    def __init__(
        self,
        in_channel: int = 3,
        hid_channel: int = 128,
        n_res_block: int = 20,#12
        n_res_channel: int = 64,
        embed_dim: int = 8,#16
        n_embed: int = 128,
        num_downsamples: int = 3,
        beta: float = 0.25  # 增加 beta 参数用于量化器
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channel,
            hid_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )
        self.enc_to_quant = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = VectorQuantizer(
            n_embed=n_embed,
            embed_dim=embed_dim,
            beta=beta  # 将 beta 传递给 VectorQuantizer
        )
        self.quant_to_dec = nn.Conv2d(embed_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        # Encoder
        z_e = self.encoder(input)  # (B, hid_channel, H', W')
        #z_e = self.enc_to_quant(z_e)  # (B, embed_dim, H', W')

        # Vector Quantization
        z_q, loss, perplexity, indices = self.quantize(z_e)  # (B, embed_dim, H', W')

        # Decoder
        #z_q = self.quant_to_dec(z_q)  # (B, hid_channel, H', W')
        x_recon = self.decoder(z_q)  # (B, in_channel, H, W)

        return x_recon, z_e, loss, perplexity, indices

class Vgg19_VQVAE_imagenet(nn.Module):
    def __init__(
        self,
        in_channel: int = 3,
        hid_channel: int = 128 ,#32 64 128 #128
        n_res_block: int = 8,# 12 6 8 #16
        n_res_channel: int = 256,#128 256 256 #128
        embed_dim: int = 16,
        n_embed: int = 256,#128 256 256 #128
        num_downsamples: int = 3,#3 3 3 #2
        beta: float = 0.25  # 增加 beta 参数用于量化器
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channel,
            hid_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )
        self.enc_to_quant = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = VectorQuantizer(
            n_embed=n_embed,
            embed_dim=embed_dim,
            beta=beta  # 将 beta 传递给 VectorQuantizer
        )
        self.quant_to_dec = nn.Conv2d(embed_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        # Encoder
        z_e = self.encoder(input)  # (B, hid_channel, H', W')
        #z_e = self.enc_to_quant(z_e)  # (B, embed_dim, H', W')

        # Vector Quantization
        z_q, loss, perplexity, indices = self.quantize(z_e)  # (B, embed_dim, H', W')

        # Decoder
        #z_q = self.quant_to_dec(z_q)  # (B, hid_channel, H', W')
        x_recon = self.decoder(z_q)  # (B, in_channel, H, W)

        return x_recon, z_e, loss, perplexity, indices

class ALLCNN_VQVAE_fashionmnist(nn.Module):
    def __init__(
        self,
        in_channel: int = 1,          # FashionMNIST is grayscale
        hid_channel: int = 96,
        n_res_block: int = 2,
        n_res_channel: int = 32,
        embed_dim: int = 8,
        n_embed: int = 96,
        num_downsamples: int = 2,
        beta: float = 0.25
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channel,
            hid_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )
        self.enc_to_quant = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = VectorQuantizer(
            n_embed=n_embed,
            embed_dim=embed_dim,
            beta=beta
        )
        self.quant_to_dec = nn.Conv2d(embed_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, float, Tensor]:
        # Encoder
        z_e = self.encoder(input)
        z_e_proj = self.enc_to_quant(z_e)

        # Vector Quantization
        z_q, loss, perplexity, indices = self.quantize(z_e_proj)

        # Decoder
        z_q_up = self.quant_to_dec(z_q)
        x_recon = self.decoder(z_q_up)

        return x_recon, z_e, loss, perplexity, indices


class CUSTOM_VQVAE_svhn(nn.Module):
    def __init__(
        self,
        in_channel: int = 3,          # SVHN is RGB
        hid_channel: int = 128,
        n_res_block: int = 12,
        n_res_channel: int = 64,
        embed_dim: int = 16,
        n_embed: int = 256,
        num_downsamples: int = 3,
        beta: float = 0.25
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channel,
            hid_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )
        self.enc_to_quant = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = VectorQuantizer(
            n_embed=n_embed,
            embed_dim=embed_dim,
            beta=beta
        )
        self.quant_to_dec = nn.Conv2d(embed_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, float, Tensor]:
        # Encoder
        z_e = self.encoder(input)
        z_e_proj = self.enc_to_quant(z_e)

        # Vector Quantization
        z_q, loss, perplexity, indices = self.quantize(z_e_proj)

        # Decoder
        z_q_up = self.quant_to_dec(z_q)
        x_recon = self.decoder(z_q_up)

        return x_recon, z_e, loss, perplexity, indices

class Resnet50_VQVAE_imagenet(nn.Module):
    def __init__(
        self,
        in_channel: int = 3,
        hid_channel: int = 128 ,#32 64 128 #128
        n_res_block: int = 8,# 12 6 8 #16
        n_res_channel: int = 256,#128 256 256 #128
        embed_dim: int = 16,
        n_embed: int = 256,#128 256 256 #128
        num_downsamples: int = 3,#3 3 3 #2
        beta: float = 0.25  # 增加 beta 参数用于量化器
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channel,
            hid_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )
        self.enc_to_quant = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = VectorQuantizer(
            n_embed=n_embed,
            embed_dim=embed_dim,
            beta=beta  # 将 beta 传递给 VectorQuantizer
        )
        self.quant_to_dec = nn.Conv2d(embed_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        # Encoder
        z_e = self.encoder(input)  # (B, hid_channel, H', W')
        #z_e = self.enc_to_quant(z_e)  # (B, embed_dim, H', W')

        # Vector Quantization
        z_q, loss, perplexity, indices = self.quantize(z_e)  # (B, embed_dim, H', W')

        # Decoder
        #z_q = self.quant_to_dec(z_q)  # (B, hid_channel, H', W')
        x_recon = self.decoder(z_q)  # (B, in_channel, H, W)

        return x_recon, z_e, loss, perplexity, indices


class Lenet5_VQVAE_mnist(nn.Module):
    def __init__(
        self,
        in_channel: int = 1,
        hid_channel: int = 96,
        n_res_block: int = 2,
        n_res_channel: int = 32,
        embed_dim: int = 8,
        n_embed: int = 96,
        num_downsamples: int = 2,
        beta: float = 0.25  # 增加 beta 参数用于量化器
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channel,
            hid_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )
        self.enc_to_quant = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = VectorQuantizer(
            n_embed=n_embed,
            embed_dim=embed_dim,
            beta=beta  # 将 beta 传递给 VectorQuantizer
        )
        self.quant_to_dec = nn.Conv2d(embed_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        # Encoder
        z_e = self.encoder(input)  # (B, hid_channel, H', W')
        #z_e = self.enc_to_quant(z_e)  # (B, embed_dim, H', W')

        # Vector Quantization
        z_q, loss, perplexity, indices = self.quantize(z_e)  # (B, embed_dim, H', W')

        # Decoder
        #z_q = self.quant_to_dec(z_q)  # (B, hid_channel, H', W')
        x_recon = self.decoder(z_q)  # (B, in_channel, H, W)

        return x_recon, z_e, loss, perplexity, indices



class Lenet4_VQVAE_mnist(nn.Module):
    def __init__(
        self,
        in_channel: int = 1,
        hid_channel: int = 96,
        n_res_block: int = 2,
        n_res_channel: int = 32,
        embed_dim: int = 8,
        n_embed: int = 96,
        num_downsamples: int = 2,
        beta: float = 0.25  # 增加 beta 参数用于量化器
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channel,
            hid_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )
        self.enc_to_quant = nn.Conv2d(hid_channel, embed_dim, 1)
        self.quantize = VectorQuantizer(
            n_embed=n_embed,
            embed_dim=embed_dim,
            beta=beta  # 将 beta 传递给 VectorQuantizer
        )
        self.quant_to_dec = nn.Conv2d(embed_dim, hid_channel, 1)
        self.decoder = Decoder(
            hid_channel,
            in_channel,
            n_res_block,
            n_res_channel,
            num_downsamples
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        # Encoder
        z_e = self.encoder(input)  # (B, hid_channel, H', W')
        #z_e = self.enc_to_quant(z_e)  # (B, embed_dim, H', W')

        # Vector Quantization
        z_q, loss, perplexity, indices = self.quantize(z_e)  # (B, embed_dim, H', W')

        # Decoder
        #z_q = self.quant_to_dec(z_q)  # (B, hid_channel, H', W')
        x_recon = self.decoder(z_q)  # (B, in_channel, H, W)


        return x_recon, z_e, loss, perplexity, indices
