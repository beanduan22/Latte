from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import torch

from latte.utils.config import load_config
from latte.utils.device import get_device
from latte.utils.io import save_json
from latte.metrics.failure import compute_metrics
from latte.metrics.semantic_drift import compute_semantic_drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--failures', required=True)
    parser.add_argument('--drift_samples', type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    run_result = torch.load(args.failures, map_location='cpu')
    metrics = compute_metrics(run_result)

    pairs = []
    for s in run_result['seed_results']:
        for f in s['failures']:
            if 'x' in f and 'x_seed' in f:
                pairs.append((f['x_seed'], f['x']))
                if len(pairs) >= args.drift_samples:
                    break
        if len(pairs) >= args.drift_samples:
            break

    try:
        metrics['semantic_drift'] = compute_semantic_drift(pairs, device)
    except Exception as e:
        metrics['semantic_drift'] = f'Unavailable: {e}'

    out_path = Path(cfg['output_dir']) / f"evaluation_{run_result.get('oracle', 'unknown')}.json"
    save_json(metrics, out_path)
    print(metrics)


if __name__ == '__main__':
    main()
