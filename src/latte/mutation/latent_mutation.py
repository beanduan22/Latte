from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class MutationConfig:
    exploration_degree: int
    num_steps: int


def mutate(z_seed: torch.Tensor, z_anchor: torch.Tensor, cfg: MutationConfig) -> torch.Tensor:
    if cfg.num_steps <= 0:
        raise ValueError('num_steps must be positive')
    t = max(0, min(cfg.exploration_degree, cfg.num_steps)) / float(cfg.num_steps)
    return z_seed + t * (z_anchor - z_seed)


def decode_mutation(vqvae, z_mut: torch.Tensor) -> torch.Tensor:
    z_q = vqvae.quantize(z_mut)
    return vqvae.decode(z_q)
