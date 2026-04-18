from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import torch

from latte.utils.config import load_config
from latte.utils.seed import set_seed
from latte.utils.device import get_device
from latte.utils.io import ensure_dir, save_json
from latte.data.datasets import build_loaders
from latte.models.vqvae import build_vqvae
from latte.train.vqvae_trainer import train_vqvae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = get_device()

    train_loader, test_loader = build_loaders(
        cfg['dataset']['name'],
        cfg['dataset']['root'],
        cfg['vqvae']['batch_size'],
        cfg['vqvae'].get('num_workers', 4),
        normalization=cfg['dataset'].get('normalization', 'half'),
        image_size=cfg['dataset'].get('image_size'),
    )
    vqvae = build_vqvae(cfg['dataset']['name']).to(device)

    history = train_vqvae(
        vqvae, train_loader, test_loader, device,
        epochs=cfg['vqvae']['epochs'],
        lr=cfg['vqvae']['lr'],
        recon_weight=cfg['vqvae'].get('recon_weight', 1.0),
    )

    out_dir = ensure_dir(cfg['output_dir'])
    torch.save({'model': vqvae.state_dict(), 'dataset': cfg['dataset']['name']},
               out_dir / 'vqvae.pt')
    save_json(history, out_dir / 'vqvae_history.json')
    print(history[-1])


if __name__ == '__main__':
    main()
