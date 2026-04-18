from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random
import time
import torch

from latte.mutation.latent_mutation import MutationConfig, mutate, decode_mutation


@dataclass
class LatteConfig:
    num_seeds: int
    pairs_per_seed: int
    exploration_degree: int
    num_steps: int
    oracle: str
    per_class_cap: Optional[int] = None
    anchor_seed: int = 0
    store_samples: bool = True


class AnchorPool:
    def __init__(self, buckets_by_class: Dict[int, List[int]], rng: random.Random):
        self.buckets = {c: idxs[:] for c, idxs in buckets_by_class.items() if idxs}
        self.rng = rng

    def sample(self, exclude_class: int) -> int:
        classes = [c for c in self.buckets.keys() if c != exclude_class]
        if not classes:
            raise RuntimeError('No anchor classes available.')
        c = self.rng.choice(classes)
        return self.rng.choice(self.buckets[c])


class LatteTester:
    def __init__(self, vqvae: torch.nn.Module, device: torch.device, cfg: LatteConfig):
        self.vqvae = vqvae
        self.device = device
        self.cfg = cfg

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vqvae.encode(x.to(self.device))

    @torch.no_grad()
    def run(self, dataset, seed_indices: List[int], buckets_by_class: Dict[int, List[int]],
            model_a: torch.nn.Module, model_b: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
        self.vqvae.eval()
        model_a.eval()
        if model_b is not None:
            model_b.eval()

        rng = random.Random(self.cfg.anchor_seed)
        anchors = AnchorPool(buckets_by_class, rng)
        mut_cfg = MutationConfig(exploration_degree=self.cfg.exploration_degree,
                                 num_steps=self.cfg.num_steps)

        seed_results = []
        total_start = time.perf_counter()

        for seed_idx in seed_indices:
            x_seed, y_seed = dataset[seed_idx]
            x_seed_t = x_seed.unsqueeze(0).to(self.device)
            y_seed = int(y_seed)
            z_seed = self._encode(x_seed_t)
            pred_a_seed = int(model_a(x_seed_t).argmax(dim=1).item())
            pred_b_seed = int(model_b(x_seed_t).argmax(dim=1).item()) if model_b is not None else None

            per_seed_failures: List[Dict[str, Any]] = []
            per_seed_diverse_classes = set()
            per_seed_confusion_pairs = set()
            seed_start = time.perf_counter()

            for _ in range(self.cfg.pairs_per_seed):
                a_idx = anchors.sample(y_seed)
                x_anchor, _ = dataset[a_idx]
                z_anchor = self._encode(x_anchor.unsqueeze(0).to(self.device))
                z_mut = mutate(z_seed, z_anchor, mut_cfg)
                x_rec = decode_mutation(self.vqvae, z_mut)

                pred_a = int(model_a(x_rec).argmax(dim=1).item())
                pred_b = int(model_b(x_rec).argmax(dim=1).item()) if model_b is not None else None

                is_failure = False
                if self.cfg.oracle == 'single':
                    if pred_a != pred_a_seed:
                        is_failure = True
                        per_seed_diverse_classes.add(pred_a)
                elif self.cfg.oracle == 'multi':
                    if pred_b is None:
                        raise RuntimeError('Multi-model oracle requires model_b.')
                    if pred_a != pred_b:
                        is_failure = True
                        pair = tuple(sorted((pred_a, pred_b)))
                        per_seed_confusion_pairs.add(pair)
                else:
                    raise ValueError(f'Unsupported oracle: {self.cfg.oracle}')

                if is_failure:
                    entry: Dict[str, Any] = {
                        'seed_idx': int(seed_idx),
                        'anchor_idx': int(a_idx),
                        'pred_a': pred_a,
                        'pred_b': pred_b,
                        'og_a': pred_a_seed,
                        'og_b': pred_b_seed,
                    }
                    if self.cfg.store_samples:
                        entry['x'] = x_rec.detach().cpu()
                        entry['x_seed'] = x_seed.detach().cpu()
                    per_seed_failures.append(entry)

            seed_elapsed = time.perf_counter() - seed_start
            seed_results.append({
                'seed_idx': int(seed_idx),
                'seed_class': y_seed,
                'og_a': pred_a_seed,
                'og_b': pred_b_seed,
                'failures': per_seed_failures,
                'diverse_classes': sorted(per_seed_diverse_classes),
                'confusion_pairs': sorted(per_seed_confusion_pairs),
                'time_sec': float(seed_elapsed),
            })

        total_elapsed = time.perf_counter() - total_start
        return {
            'seed_results': seed_results,
            'total_time_sec': float(total_elapsed),
            'oracle': self.cfg.oracle,
        }
