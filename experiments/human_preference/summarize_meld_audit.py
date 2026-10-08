#!/usr/bin/env python3
"""Summarize three-annotator MELD-ESC300 validity audits; no paper labels are embedded."""
import argparse
import json
from pathlib import Path


CRITERIA = ('emotional_need', 'relevant_response', 'supportive_function')


def summarize(rows, expected_samples=300):
    if not rows or (expected_samples and len(rows) != expected_samples):
        raise ValueError(f'Expected {expected_samples} nonempty audit rows; found {len(rows)}')
    seen, resolved = set(), []
    for row in rows:
        sample_id = int(row['sample_id'])
        if sample_id in seen:
            raise ValueError(f'Duplicate sample_id: {sample_id}')
        seen.add(sample_id)
        if row.get('subset') not in {'strict', 'relaxed'}:
            raise ValueError('subset must be strict or relaxed')
        item = {'subset': row['subset']}
        for criterion in CRITERIA:
            votes = row.get(criterion)
            if not isinstance(votes, list) or len(votes) != 3 or any(type(v) is not bool for v in votes):
                raise ValueError(f'{sample_id}/{criterion}: expected three boolean judgments')
            item[criterion] = sum(votes) >= 2
        item['overall_validity'] = all(item[c] for c in CRITERIA)
        resolved.append(item)
    result = {}
    for subset in ('strict', 'relaxed', 'overall'):
        items = resolved if subset == 'overall' else [r for r in resolved if r['subset'] == subset]
        n = len(items)
        result[subset] = {'n': n}
        for criterion in (*CRITERIA, 'overall_validity'):
            count = sum(row[criterion] for row in items)
            result[subset][criterion] = {'count': count, 'rate': count / n if n else None}
    if expected_samples == 300 and (result['strict']['n'], result['relaxed']['n']) != (208, 92):
        raise ValueError('The paper audit requires 208 strict and 92 relaxed samples')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--expected_samples', type=int, default=300, help='0 allows a pilot.')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'outputs/human_preference/meld_audit.json')
    args = parser.parse_args()
    if args.expected_samples < 0:
        raise ValueError('--expected_samples must be nonnegative')
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    result = summarize(rows, args.expected_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
