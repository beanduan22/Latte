from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F


def _ensure_rgb_224(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = x.clamp(-1.0, 1.0) * 0.5 + 0.5
    return (x - mean) / std


class DINOv2Encoder:
    def __init__(self, device: torch.device, variant: str = 'dinov2_vits14'):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', variant)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_rgb_224(x).to(self.device)
        return self.model(x)


def compute_semantic_drift(pairs: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device,
                           variant: str = 'dinov2_vits14') -> float:
    if not pairs:
        return float('nan')
    encoder = DINOv2Encoder(device, variant=variant)
    drifts = []
    for x_seed, x_test in pairs:
        z1 = F.normalize(encoder.embed(x_seed), dim=-1)
        z2 = F.normalize(encoder.embed(x_test), dim=-1)
        cos = (z1 * z2).sum(dim=-1)
        drifts.append(float((1.0 - cos).mean().item()))
    return float(sum(drifts) / len(drifts))
