from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from tqdm import tqdm


def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return {'loss': loss_sum / max(1, total), 'acc': correct / max(1, total)}


def train_classifier(model, train_loader, test_loader, device, epochs: int, lr: float) -> list:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Classifier {epoch + 1}/{epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))
        metrics = evaluate(model, test_loader, device)
        metrics['epoch'] = epoch + 1
        history.append(metrics)
    return history
