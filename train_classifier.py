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
from latte.data.datasets import build_loaders, dataset_meta
from latte.models.classifiers import build_classifier
from latte.train.classifier_trainer import train_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--target', default='a', choices=['a', 'b'])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = get_device()

    train_loader, test_loader = build_loaders(
        cfg['dataset']['name'],
        cfg['dataset']['root'],
        cfg['train']['batch_size'],
        cfg['train'].get('num_workers', 4),
        normalization=cfg['dataset'].get('normalization', 'half'),
        image_size=cfg['dataset'].get('image_size'),
    )
    meta = dataset_meta(cfg['dataset']['name'])

    model_key = 'model_a' if args.target == 'a' else 'model_b'
    model_cfg = cfg[model_key]
    model = build_classifier(model_cfg['name'], meta['num_classes'],
                             pretrained=model_cfg.get('pretrained', False)).to(device)

    history = train_classifier(
        model, train_loader, test_loader, device,
        epochs=cfg['train']['epochs'],
        lr=cfg['train']['lr'],
    )

    out_dir = ensure_dir(cfg['output_dir'])
    torch.save({'model': model.state_dict(), 'name': model_cfg['name']},
               out_dir / f'classifier_{args.target}.pt')
    save_json(history, out_dir / f'classifier_{args.target}_history.json')
    print(history[-1])


if __name__ == '__main__':
    main()
