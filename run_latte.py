from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import torch

from latte.utils.config import load_config
from latte.utils.seed import set_seed
from latte.utils.device import get_device
from latte.utils.io import ensure_dir, save_json, save_torch
from latte.data.datasets import (
    build_datasets, dataset_meta, group_indices_by_class,
    select_correctly_classified_seeds, select_agreement_seeds,
)
from latte.models.classifiers import build_classifier
from latte.models.vqvae import build_vqvae
from latte.testing.latte import LatteTester, LatteConfig
from latte.metrics.failure import compute_metrics


def _load_classifier(cfg, meta, device, target: str):
    key = 'model_a' if target == 'a' else 'model_b'
    mcfg = cfg[key]
    model = build_classifier(mcfg['name'], meta['num_classes'],
                             pretrained=mcfg.get('pretrained', False)).to(device)
    ckpt = torch.load(cfg['classifier_ckpt'][target], map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = get_device()

    ds_cfg = cfg['dataset']
    meta = dataset_meta(ds_cfg['name'])

    train_ds, _ = build_datasets(
        ds_cfg['name'], ds_cfg['root'],
        normalization=ds_cfg.get('normalization', 'half'),
        image_size=ds_cfg.get('image_size'),
    )

    model_a = _load_classifier(cfg, meta, device, 'a')
    model_b = None
    if cfg['latte']['oracle'] == 'multi':
        model_b = _load_classifier(cfg, meta, device, 'b')

    vqvae = build_vqvae(ds_cfg['name']).to(device)
    vq_ckpt = torch.load(cfg['vqvae_ckpt'], map_location=device)
    vqvae.load_state_dict(vq_ckpt['model'])
    vqvae.eval()

    lcfg = cfg['latte']
    if lcfg['oracle'] == 'single':
        seed_indices = select_correctly_classified_seeds(
            model_a, train_ds, device,
            num_seeds=lcfg['num_seeds'],
            per_class_cap=lcfg.get('per_class_cap'),
        )
    else:
        seed_indices = select_agreement_seeds(
            model_a, model_b, train_ds, device, num_seeds=lcfg['num_seeds'],
        )

    buckets = group_indices_by_class(
        train_ds, meta['num_classes'],
        limit=lcfg.get('anchor_pool_limit'),
    )

    tester_cfg = LatteConfig(
        num_seeds=lcfg['num_seeds'],
        pairs_per_seed=lcfg['pairs_per_seed'],
        exploration_degree=lcfg['exploration_degree'],
        num_steps=lcfg['num_steps'],
        oracle=lcfg['oracle'],
        per_class_cap=lcfg.get('per_class_cap'),
        anchor_seed=lcfg.get('anchor_seed', cfg['seed']),
        store_samples=lcfg.get('store_samples', True),
    )
    tester = LatteTester(vqvae, device, tester_cfg)
    result = tester.run(train_ds, seed_indices, buckets, model_a, model_b)

    metrics = compute_metrics(result)
    out_dir = ensure_dir(cfg['output_dir'])
    save_torch(result, out_dir / f"failures_{lcfg['oracle']}.pt")
    save_json(metrics, out_dir / f"metrics_{lcfg['oracle']}.json")
    print(metrics)


if __name__ == '__main__':
    main()
