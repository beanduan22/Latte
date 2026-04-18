from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import random
import torch
from torch.utils.data import Dataset

from latte.models.classifiers import build_classifier
from latte.models.vqvae import build_vqvae
from latte.mutation.latent_mutation import MutationConfig, mutate, decode_mutation
from latte.testing.latte import LatteTester, LatteConfig
from latte.metrics.failure import compute_metrics


class SyntheticDataset(Dataset):
    def __init__(self, n: int, channels: int, size: int, num_classes: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, channels, size, size, generator=g) * 2 - 1
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], int(self.y[idx].item())


def test_mutation_is_interpolation():
    z_s = torch.zeros(1, 4, 4, 4)
    z_a = torch.ones(1, 4, 4, 4)
    cfg = MutationConfig(exploration_degree=5, num_steps=10)
    z_m = mutate(z_s, z_a, cfg)
    assert torch.allclose(z_m, torch.full_like(z_s, 0.5))


def test_end_to_end_single_and_multi():
    device = torch.device('cpu')
    ds = SyntheticDataset(n=40, channels=1, size=28, num_classes=10, seed=1)

    model_a = build_classifier('lenet5', 10, pretrained=False).to(device).eval()
    model_b = build_classifier('lenet4', 10, pretrained=False).to(device).eval()
    vqvae = build_vqvae('mnist').to(device).eval()

    x0 = ds[0][0].unsqueeze(0)
    x_rec, _ = vqvae(x0)
    assert x_rec.shape == x0.shape

    buckets = {c: [] for c in range(10)}
    for i in range(len(ds)):
        buckets[int(ds[i][1])].append(i)

    for oracle in ('single', 'multi'):
        cfg = LatteConfig(
            num_seeds=3,
            pairs_per_seed=2,
            exploration_degree=3,
            num_steps=10,
            oracle=oracle,
            store_samples=True,
        )
        seed_indices = [i for i in range(len(ds)) if len(buckets[int(ds[i][1])]) > 0][:3]
        tester = LatteTester(vqvae, device, cfg)
        result = tester.run(ds, seed_indices, buckets, model_a, model_b)
        metrics = compute_metrics(result)
        assert 'failure_count' in metrics
        assert metrics['oracle'] == oracle


def main():
    random.seed(0)
    torch.manual_seed(0)
    test_mutation_is_interpolation()
    test_end_to_end_single_and_multi()
    print('SMOKE_TEST_OK')


if __name__ == '__main__':
    main()
