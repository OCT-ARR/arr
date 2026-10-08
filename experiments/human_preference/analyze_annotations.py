#!/usr/bin/env python3
"""Aggregate blind preferences and compute pre-adjudication and human–judge agreement."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scipy.stats import binomtest


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def unique_index(rows, field):
    indexed = {}
    for row in rows:
        key = int(row[field])
        if key in indexed:
            raise ValueError(f'Duplicate {field}: {key}')
        indexed[key] = row
    return indexed


def label(value):
    value = str(value).strip().upper()
    if value not in {'A', 'B', 'TIE'}:
        raise ValueError(f'Expected A/B/TIE, received {value!r}')
    return value


def mapped_label(value, key):
    value = label(value)
    return 'TIE' if value == 'TIE' else key[f'candidate_{value.lower()}_model']


def fleiss_kappa(rows, categories):
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError('Fleiss kappa requires three independent labels per item')
    n = len(rows)
    counts = [Counter(row) for row in rows]
    observed = sum(sum(v * (v - 1) for v in row.values()) / 6 for row in counts) / n
    expected = sum((sum(row[c] for row in counts) / (3 * n)) ** 2 for c in categories)
    return None if abs(1 - expected) < 1e-12 else (observed - expected) / (1 - expected)


def cohen_agreement(human, judge, categories):
    if not human or len(human) != len(judge):
        raise ValueError('Agreement requires equally sized nonempty label lists')
    if (set(human) | set(judge)) - set(categories):
        raise ValueError('Unknown agreement category')
    n = len(human)
    matches = sum(a == b for a, b in zip(human, judge))
    h, j = Counter(human), Counter(judge)
    expected = sum(h[c] * j[c] for c in categories) / (n * n)
    observed = matches / n
    return {'n': n, 'exact_matches': matches, 'exact_agreement': observed,
            'chance_agreement': expected,
            'cohens_kappa': None if abs(1 - expected) < 1e-12 else (observed - expected) / (1 - expected)}


def summarize(keys, annotations, judge_records=None, left_name='faces_main'):
    key_by_item = unique_index(keys, 'item')
    unique_index(keys, 'sample_id')
    annotations_by_item = unique_index(annotations, 'item')
    if not keys or key_by_item.keys() != annotations_by_item.keys():
        raise ValueError('Annotations must cover every questionnaire item exactly once')
    opponents = {row[f'candidate_{side}_model'] for row in keys for side in ('a', 'b')} - {left_name}
    if len(opponents) != 1:
        raise ValueError('Expected a single left/right model pair')
    right_name = opponents.pop()
    categories = (left_name, right_name, 'TIE')
    final, independent, adjudications = [], [], 0
    for item, key in sorted(key_by_item.items()):
        row = annotations_by_item[item]
        if int(row['sample_id']) != int(key['sample_id']):
            raise ValueError(f'Item {item}: sample ID differs from private key')
        if {key['candidate_a_model'], key['candidate_b_model']} != {left_name, right_name}:
            raise ValueError(f'Item {item}: inconsistent model pair')
        raw = [label(row.get(f'annotator_{i}')) for i in (1, 2, 3)]
        independent.append([mapped_label(v, key) for v in raw])
        votes = Counter(raw)
        winner, count = votes.most_common(1)[0]
        if count == 1:
            winner = label(row.get('adjudicated_label'))
            adjudications += 1
        final.append({'sample_id': int(key['sample_id']), 'winner_name': mapped_label(winner, key)})
    counts = Counter(row['winner_name'] for row in final)
    n = len(final)
    decisive = counts[left_name] + counts[right_name]
    result = {'n': n, 'left_name': left_name, 'right_name': right_name,
              'counts': {c: counts[c] for c in categories},
              'rates_including_ties': {c: counts[c] / n for c in categories},
              'adjudicated_items': adjudications,
              'fleiss_kappa_before_adjudication': fleiss_kappa(independent, categories),
              'decisive_count': decisive,
              'left_share_excluding_ties': counts[left_name] / decisive if decisive else None,
              'two_sided_sign_test_p': float(binomtest(counts[left_name], decisive, 0.5).pvalue) if decisive else 1.0}
    if judge_records is not None:
        judge_by_id = unique_index(judge_records, 'sample_id')
        if judge_by_id.keys() != {row['sample_id'] for row in final}:
            raise ValueError('Judge records must match all human sample IDs exactly')
        human = [row['winner_name'] for row in final]
        judged = [judge_by_id[row['sample_id']]['winner_name'] for row in final]
        result['human_judge_agreement'] = cohen_agreement(human, judged, categories)
    return result, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private_key', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--judge_records', type=Path)
    parser.add_argument('--left_name', default='faces_main')
    parser.add_argument('--expected_samples', type=int, help='Infer paper split by default; 0 allows a pilot.')
    parser.add_argument('--output_dir', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'outputs/human_preference/analysis')
    args = parser.parse_args()
    keys = json.loads(args.private_key.read_text(encoding='utf-8'))
    expected = args.expected_samples
    if expected is None:
        datasets = {row.get('dataset') for row in keys}
        if len(datasets) != 1 or next(iter(datasets)) not in {'mesc', 'meld'}:
            raise ValueError('Specify --expected_samples for a custom questionnaire')
        expected = 450 if datasets == {'mesc'} else 300
    if expected < 0 or (expected and len(keys) != expected):
        raise ValueError(f'Expected {expected} questionnaire items; found {len(keys)}')
    result, final = summarize(keys, read_jsonl(args.annotations),
                              read_jsonl(args.judge_records) if args.judge_records else None,
                              args.left_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'final_preferences.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in final))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
