#!/usr/bin/env python3
"""Convert local candidate records to API-metric inputs using the same dataset-derived IDs."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/api_baselines'))
from api_baselines.data import load_dataset
DATASETS = {
    'mesc': ('MESC-FT', 'experiments/llm_blind_judge/outputs/generations'),
    'meld_esc300': ('MELD-ESC300', 'experiments/generalization/outputs_emotyda/generations'),
    'stickerconv': ('StickerConv', 'experiments/generalization/outputs/stickerconv/generations'),
    'avamerg': ('AvaMERG', 'experiments/generalization/outputs/avamerg/generations'),
}
MODEL_MAP = {'faces_main': 'faces', 'grpo_baseline': 'baseline'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=['stickerconv', 'avamerg'])
    parser.add_argument('--output_root', type=Path, default=ROOT / 'experiments/api_baselines/outputs/raw')
    args = parser.parse_args()
    for name in args.datasets:
        folder, generated = DATASETS[name]
        samples = load_dataset(name, ROOT / 'data' / folder / 'test.jsonl')
        for local_name, exported_name in MODEL_MAP.items():
            src = ROOT / generated / f'{local_name}.jsonl'
            rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
            by_index = {int(row['sample_id']): row for row in rows}
            missing = [sample.source_index for sample in samples if sample.source_index not in by_index]
            if missing:
                raise ValueError(f'{src}: missing {len(missing)} aligned predictions')
            out = args.output_root / name / f'{exported_name}.jsonl'
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open('w', encoding='utf-8') as handle:
                for sample in samples:
                    row = by_index[sample.source_index]
                    handle.write(json.dumps({'sample_id': sample.sample_id, 'source_index': sample.source_index,
                                             'final_response': row.get('response') or row.get('final_response') or row.get('raw_output') or '',
                                             'status': 'ok'}, ensure_ascii=False) + '\n')
            print(f'{name}/{exported_name}: {len(samples)} records -> {out}')

if __name__ == '__main__':
    main()
