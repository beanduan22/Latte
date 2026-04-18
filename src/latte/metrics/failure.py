from __future__ import annotations
from typing import Any, Dict, List


def compute_metrics(run_result: Dict[str, Any]) -> Dict[str, Any]:
    seeds = run_result['seed_results']
    total_seeds = len(seeds)
    oracle = run_result.get('oracle', 'single')

    total_failures = sum(len(s['failures']) for s in seeds)
    seed_covered = sum(1 for s in seeds if len(s['failures']) > 0)

    metrics: Dict[str, Any] = {
        'oracle': oracle,
        'failure_count': int(total_failures),
        'seed_coverage': float(seed_covered / max(1, total_seeds)),
        'testing_time_sec': float(run_result.get('total_time_sec', 0.0)),
    }

    if oracle == 'single':
        per_seed_dof = [len(s['diverse_classes']) for s in seeds]
        metrics['failure_diversity'] = float(sum(per_seed_dof) / max(1, total_seeds))
    elif oracle == 'multi':
        pair_set = set()
        for s in seeds:
            for p in s['confusion_pairs']:
                pair_set.add(tuple(p))
        metrics['confusion_pair_diversity'] = int(len(pair_set))

    return metrics
