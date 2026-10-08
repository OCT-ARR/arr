#!/usr/bin/env python3
"""Compare aligned and unaligned credit with the five-dimensional validation diagnostic."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--faces_model', type=Path, default=ROOT / 'checkpoints/faces_main')
    parser.add_argument('--unaligned_model', type=Path, default=ROOT / 'experiments/ablation/a3_w_o_fine_grained_credit/merged_model')
    parser.add_argument('--validation_file', type=Path, default=ROOT / 'data/MESC-FT/val.jsonl')
    parser.add_argument('--judge_model', type=Path, default=ROOT / 'models/judge-32b')
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda_devices', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1'))
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'outputs/span_alignment')
    args = parser.parse_args()
    if args.samples <= 0 or args.samples % 2:
        raise ValueError('--samples must be positive and even for two disjoint evaluation rounds')
    for path in [args.faces_model, args.unaligned_model, args.validation_file, args.judge_model]:
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ['AUTO_TEST_VISIBLE_CUDA_DEVICES'] = args.cuda_devices
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from src.grpo.faces_grpo import evaluate_faces_checkpoints as evaluator
    import torch
    original_loader = evaluator.build_model_load_attempts
    def load_attempts(config, device):
        quant = getattr(config, 'quantization_config', None)
        if isinstance(quant, dict) and quant.get('quant_method', '').lower() == 'awq' and device.startswith('cuda'):
            common = dict(config=config, trust_remote_code=True, low_cpu_mem_usage=True, device_map={'': device})
            return [dict(common, dtype=torch.float16), dict(common, torch_dtype=torch.float16)]
        return original_loader(config, device)
    def discover():
        return [evaluator.ModelSpec(name='faces_main', path=str(args.faces_model)),
                evaluator.ModelSpec(name='without_span_alignment', path=str(args.unaligned_model))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.TEST_FILE = args.validation_file
    evaluator.JUDGE_MODEL_PATH = args.judge_model
    evaluator.ROUND_SIZE = args.samples // 2
    evaluator.RANDOM_SEED = args.seed
    evaluator.SCORE_FILE = args.output_dir / 'score.txt'
    evaluator.SCORE_EVERY_DIMENSION_FILE = args.output_dir / 'score_every_dimension.txt'
    evaluator.discover_candidate_models = discover
    evaluator.build_model_load_attempts = load_attempts
    evaluator.main()

if __name__ == '__main__':
    main()
