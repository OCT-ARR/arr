#!/usr/bin/env python3
"""Generate FACES/FIRES external candidates with the shared greedy-decoding implementation."""
from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mesc', 'meld', 'stickerconv', 'avamerg'], default='stickerconv')
    parser.add_argument('--sample_limit', type=int, default=0, help='0 uses the complete 450-example MESC or 300-example external split.')
    parser.add_argument('--faces_model', type=Path, default=ROOT / 'checkpoints/faces_main')
    parser.add_argument('--baseline_model', type=Path, default=ROOT / 'checkpoints/grpo_baseline')
    parser.add_argument('--faces_device', default='cuda:0')
    parser.add_argument('--baseline_device', default='cuda:1')
    parser.add_argument('--faces_name', default='faces_main')
    parser.add_argument('--output_dir', type=Path, help='Override the generation directory for a credit variant.')
    args = parser.parse_args()
    if args.sample_limit < 0:
        raise ValueError('--sample_limit must be >= 0')
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from experiments.llm_blind_judge import stickerconv_eval as base
    from experiments.llm_blind_judge.pairwise_controlled_eval import DATASETS
    ds = DATASETS[args.dataset]
    samples = base.load_test_samples(ds.test_file, args.sample_limit)
    if not samples:
        raise ValueError(f'No valid samples in {ds.test_file}')
    expected = 450 if args.dataset == 'mesc' else 300
    if args.sample_limit == 0 and len(samples) != expected:
        raise ValueError(f'The {args.dataset} protocol requires {expected} samples; found {len(samples)}')
    for path in [args.faces_model, args.baseline_model]:
        if not path.is_dir():
            raise FileNotFoundError(path)
    base.CANDIDATE_DEVICES = {args.faces_name: args.faces_device, 'grpo_baseline': args.baseline_device}
    base.GEN_MAX_NEW_TOKENS = 256
    base.GEN_DO_SAMPLE = False
    base.RUN_GENERATION = True
    # Spawned workers get their device through the task arguments.
    base.generate_all_predictions({args.faces_name: args.faces_model, 'grpo_baseline': args.baseline_model},
                                  samples, args.output_dir or ds.left_generation_file.parent)

if __name__ == '__main__':
    main()
