from __future__ import annotations
import torch


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
