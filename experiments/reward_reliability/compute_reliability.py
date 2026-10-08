#!/usr/bin/env python3
"""Compute the Appendix A.4 absolute-agreement ICC and human correlations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


PROMPT_DIMENSIONS = ('situation', 'strategy', 'outcome')
HUMAN_DIMENSIONS = ('visual', 'emotion', 'situation', 'strategy', 'outcome')


def icc_absolute(matrix):
    """Two-way random-effects absolute agreement: ICC(2,1) and ICC(2,k)."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError('ICC needs at least two candidates and two prompt variants')
    if not np.isfinite(values).all():
        raise ValueError('ICC input contains non-finite scores')
    n, k = values.shape
    grand = values.mean()
    row_means, column_means = values.mean(1), values.mean(0)
    ms_rows = k * np.square(row_means - grand).sum() / (n - 1)
    ms_columns = n * np.square(column_means - grand).sum() / (k - 1)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    single_denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    average_denominator = ms_rows + (ms_columns - ms_error) / n
    return ((ms_rows - ms_error) / single_denominator,
            (ms_rows - ms_error) / average_denominator)


def load_records(path):
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        prompt = item.get('prompt_scores', {})
        reward = item.get('reward_scores', {})
        for dimension in PROMPT_DIMENSIONS:
            scores = prompt.get(dimension)
            if not isinstance(scores, list) or len(scores) != 3:
                raise ValueError(f'Line {line_number}: {dimension} needs three prompt scores')
        required_rewards = set(HUMAN_DIMENSIONS) | {'gated_reward'}
        if not required_rewards <= reward.keys():
            raise ValueError(f'Line {line_number}: incomplete reward_scores')
        records.append(item)
    if not records:
        raise ValueError('No reliability records found')
    return records


def mean_human(item, key):
    values = item.get('human_scores', {}).get(key)
    if values is None:
        return None
    if not isinstance(values, list) or not values or not np.isfinite(values).all():
        raise ValueError(f'Invalid human scores for {key}')
    return float(np.mean(values))


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=root / 'outputs/reward_reliability/reliability.json')
    parser.add_argument('--expected_candidates', type=int, default=1829)
    parser.add_argument('--expected_human_subset', type=int, default=150)
    args = parser.parse_args()
    records = load_records(args.input)
    if args.expected_candidates and len(records) != args.expected_candidates:
        raise ValueError(f'Expected {args.expected_candidates} candidates; found {len(records)}')
    result = {'candidate_count': len(records), 'prompt_reliability': {}, 'human_alignment': {}}
    for dimension in PROMPT_DIMENSIONS:
        matrix = [x['prompt_scores'][dimension] for x in records]
        single, average = icc_absolute(matrix)
        result['prompt_reliability'][dimension] = {
            'ICC(2,1)': float(single), 'ICC(2,3)': float(average)}
    correlation_specs = [(x, x) for x in HUMAN_DIMENSIONS] + [('gated_reward', 'overall_quality')]
    human_counts = set()
    for reward_key, human_key in correlation_specs:
        pairs = [(float(x['reward_scores'][reward_key]), mean_human(x, human_key)) for x in records]
        pairs = [(reward, human) for reward, human in pairs if human is not None]
        human_counts.add(len(pairs))
        if len(pairs) < 3:
            raise ValueError(f'Insufficient human annotations for {reward_key}')
        rho, p = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
        result['human_alignment'][reward_key] = {
            'human_dimension': human_key, 'n': len(pairs),
            'spearman_rho': float(rho), 'p_value': float(p)}
    if args.expected_human_subset and human_counts != {args.expected_human_subset}:
        raise ValueError(f'Expected the same {args.expected_human_subset}-candidate human subset; '
                         f'found counts {sorted(human_counts)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
