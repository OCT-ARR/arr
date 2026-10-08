#!/usr/bin/env python3
"""Evaluate cap-15 responses with the shared three-prompt, 1-5 pointwise judge."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.llm_blind_judge import pairwise_controlled_eval as controlled

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(controlled.DATASETS), default='mesc')
    parser.add_argument('--predictions', type=Path, help='Override predictions for a credit variant or API opponent.')
    parser.add_argument('--model_name', default='faces_main')
    parser.add_argument('--sample_limit', type=int, default=0)
    parser.add_argument('--max_words', type=int, default=15)
    parser.add_argument('--judge_model', type=Path, default=ROOT / 'models/judge-32b')
    parser.add_argument('--judge_device', default='cuda:0')
    parser.add_argument('--output_root', type=Path, default=ROOT / 'outputs/controlled_pointwise')
    args = parser.parse_args()
    if args.sample_limit < 0 or args.max_words <= 0:
        raise ValueError('Invalid sample limit or word cap')
    base = controlled.load_base_module()
    ds = controlled.DATASETS[args.dataset]
    src = args.predictions or (ds.left_generation_file if args.model_name == 'faces_main' else ds.right_generation_file)
    samples = base.load_test_samples(ds.test_file, args.sample_limit)
    predictions = controlled.load_predictions(base, src, args.model_name, samples)
    visible, _, protocol = controlled.build_control_predictions('length_controlled', predictions, predictions, args.max_words, 'cap')
    out_dir = args.output_root / args.dataset / f'cap{args.max_words}'
    judge = base.StickerConvJudge(str(args.judge_model), args.judge_device)
    try:
        base.run_pointwise(judge, samples, visible, out_dir / f'{args.model_name}.jsonl', out_dir / f'{args.model_name}_summary.json')
        controlled.write_json(out_dir / f'{args.model_name}_protocol.json', protocol)
    finally:
        judge.close()

if __name__ == '__main__':
    main()
