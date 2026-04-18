from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from tqdm import tqdm


def evaluate_vqvae(vqvae, loader, device) -> Dict[str, float]:
    vqvae.eval()
    recon_sum, vq_sum, total = 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_rec, vq_loss = vqvae(x)
            recon = F.mse_loss(x_rec, x, reduction='sum').item() / x.numel()
            recon_sum += recon * x.size(0)
            vq_sum += float(vq_loss.item()) * x.size(0)
            total += x.size(0)
    return {'recon_loss': recon_sum / max(1, total), 'vq_loss': vq_sum / max(1, total)}


def train_vqvae(vqvae, train_loader, test_loader, device, epochs: int, lr: float,
                recon_weight: float = 1.0) -> list:
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        vqvae.train()
        pbar = tqdm(train_loader, desc=f'VQ-VAE {epoch + 1}/{epochs}')
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            x_rec, vq_loss = vqvae(x)
            recon = F.mse_loss(x_rec, x)
            loss = recon_weight * recon + vq_loss
            loss.backward()
            optimizer.step()
            pbar.set_postfix(recon=float(recon.item()), vq=float(vq_loss.item()))
        metrics = evaluate_vqvae(vqvae, test_loader, device)
        metrics['epoch'] = epoch + 1
        history.append(metrics)
    return history
