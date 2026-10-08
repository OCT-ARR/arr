#!/usr/bin/env python3
"""Generate deterministic MESC test responses for one merged optimization baseline."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['dpo', 'simpo', 'rloo', 'ppo'], required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sample_limit', type=int, default=0, help='0 requires all 450 MESC test examples')
    parser.add_argument('--output_dir', type=Path,
                        default=ROOT / 'outputs/optimization_baselines/generations')
    args = parser.parse_args()
    if args.sample_limit < 0:
        parser.error('--sample_limit must be nonnegative')
    os.chdir(ROOT)
    from experiments.llm_blind_judge import stickerconv_eval as base
    model = args.model or ROOT / 'checkpoints/optimization_baselines' / f'{args.method}_merged'
    if not model.is_dir():
        raise FileNotFoundError(f'Merged model not found: {model}')
    test_file = ROOT / 'data/MESC-FT/test.jsonl'
    samples = base.load_test_samples(test_file, args.sample_limit)
    if args.sample_limit == 0 and len(samples) != 450:
        raise ValueError(f'Paper protocol requires 450 MESC examples; found {len(samples)}')
    base.RUN_GENERATION = True
    base.FORCE_REGENERATE = True
    base.PARALLEL_CANDIDATE_GENERATION = False
    base.CANDIDATE_DEVICES = {args.method: args.device}
    base.GEN_DO_SAMPLE = False
    base.GEN_MAX_NEW_TOKENS = 256
    base.generate_all_predictions({args.method: model}, samples, args.output_dir)


if __name__ == '__main__':
    main()
