#!/usr/bin/env python3
"""Prepare anonymized A/B/TIE questionnaires and empty annotation templates for full splits."""
from __future__ import annotations
import argparse
import gzip
import json
import random
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = {'mesc': ('MESC-FT', 'experiments/llm_blind_judge/outputs/generations'),
        'meld': ('MELD-ESC300', 'experiments/generalization/outputs_emotyda/generations')}

def read_rows(path):
    path = path if path.exists() else path.with_suffix(path.suffix + '.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def response(row, max_words):
    text = str(row.get('response') or row.get('final_response') or row.get('raw_output') or '')
    match = re.search(r'<response>(.*?)</response>', text, flags=re.S|re.I)
    text = match.group(1) if match else re.sub(r'<think>.*?</think>', '', text, flags=re.S|re.I)
    return ' '.join(text.strip().split()[:max_words])

def index_rows(path):
    return {int(row.get('source_index', row.get('sample_id'))): row for row in read_rows(path) if row.get('status', 'ok') == 'ok'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(DATA), default='mesc')
    parser.add_argument('--left_predictions', type=Path)
    parser.add_argument('--right_predictions', type=Path)
    parser.add_argument('--left_name', default='faces_main')
    parser.add_argument('--right_name', default='grpo_baseline')
    parser.add_argument('--limit', type=int, default=0, help='0 uses all 450 MESC or 300 MELD examples.')
    parser.add_argument('--max_words', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_root', type=Path, default=ROOT / 'outputs/human_preference')
    args = parser.parse_args()
    if args.limit < 0 or args.max_words <= 0:
        raise ValueError('Invalid sample limit or word cap')
    folder, gen = DATA[args.dataset]
    samples = read_rows(ROOT / 'data' / folder / 'test.jsonl')
    left = index_rows(args.left_predictions or ROOT / gen / 'faces_main.jsonl')
    right = index_rows(args.right_predictions or ROOT / gen / 'grpo_baseline.jsonl')
    rng = random.Random(args.seed)
    ids = list(range(len(samples)))
    rng.shuffle(ids)
    if args.limit:
        ids = ids[:args.limit]
    placements = [i % 2 == 0 for i in range(len(ids))]
    rng.shuffle(placements)
    lines = ['# Blind human preference annotation', '',
             f'Compare Candidate A and B using the same dialogue and visual evidence. Consider empathy, multimodal appropriateness, strategy suitability, and helpfulness. Select A, B, or TIE. Reasoning is hidden and each response is capped at {args.max_words} words.', '']
    keys, empty = [], []
    for number, (sample_id, left_is_a) in enumerate(zip(ids, placements), 1):
        user = next(turn for turn in samples[sample_id]['conversations'] if turn['from'] == 'user')
        la, rb = response(left[sample_id], args.max_words), response(right[sample_id], args.max_words)
        ca, cb = (la, rb) if left_is_a else (rb, la)
        lines += [f'## Item {number}', '', str(user['value']).replace('<image>', '').strip(), '']
        for image in user.get('images', []):
            path = Path(image); path = path if path.is_absolute() else ROOT / path
            lines += [f'![Visual evidence]({path})', '']
        lines += [f'Candidate A: {ca}', '', f'Candidate B: {cb}', '', 'Preference: A / B / TIE', '']
        keys.append({'item': number, 'sample_id': sample_id, 'dataset': args.dataset,
                     'candidate_a_model': args.left_name if left_is_a else args.right_name,
                     'candidate_b_model': args.right_name if left_is_a else args.left_name})
        empty.append({'item': number, 'sample_id': sample_id, 'annotator_1': None, 'annotator_2': None,
                      'annotator_3': None, 'adjudicated_label': None})
    out = args.output_root / args.dataset / f'{args.left_name}_vs_{args.right_name}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'questionnaire.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'private_key.json').write_text(json.dumps(keys, indent=2), encoding='utf-8')
    with (out / 'annotations_template.jsonl').open('w') as handle:
        for row in empty:
            handle.write(json.dumps(row) + '\n')
    print(f'Prepared {len(keys)} blinded items in {out}')

if __name__ == '__main__':
    main()
