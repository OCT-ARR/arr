#!/usr/bin/env python3
"""Create fixed, reward-ranked training pairs from the common SFT checkpoint."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/optimization_baselines.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/optimization_baselines/preferences.jsonl')
    parser.add_argument('--limit', type=int, default=0, help='0 uses every training prompt')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    config_path, output = args.config.resolve(), args.output.resolve()
    os.chdir(ROOT)
    from experiments.optimization_baselines import common as C
    from src.grpo.faces_grpo import train_faces_grpo as T
    config = C.load_config(config_path)
    cfg = C.train_config(config)
    if int(config.get('preference_resample_rounds', 0)) <= 0:
        raise ValueError('preference_resample_rounds must be positive')
    T.set_seed(cfg.seed)
    if Path(cfg.train_file).resolve() == Path(cfg.val_file).resolve():
        raise ValueError('Training and validation files must differ')
    if output.exists() or output.with_suffix('.manifest.json').exists():
        raise FileExistsError('Use a new output path to avoid mixing preference datasets')
    dataset = T.JsonlConversationDataset(cfg.train_file)
    if not len(dataset):
        raise ValueError('Empty training dataset')
    model, processor = T.build_policy_model_and_processor(cfg)
    model.requires_grad_(False)
    engine = C.reward_engine(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    total, kept = 0, 0
    with output.open('x') as handle:
        for index in range(min(args.limit or len(dataset), len(dataset))):
            sample = dataset[index]
            inputs, images = C.prompt_inputs(processor, sample, cfg, config['max_sequence_length'])
            tokens, scores = [], []
            for _ in range(config['preference_resample_rounds']):
                new_tokens = C.generate(model, processor, inputs, cfg, config['max_sequence_length'])
                texts = [processor.tokenizer.decode(ids, skip_special_tokens=True) for ids in new_tokens]
                rewards = engine.score_completions(sample, texts, images)
                new_scores = [float(r.total) for r in rewards]
                if len(new_scores) != len(new_tokens) or not all(math.isfinite(x) for x in new_scores):
                    raise ValueError(f'Invalid rewards at training sample {index}')
                tokens.extend(new_tokens)
                scores.extend(new_scores)
                lo = min(range(len(scores)), key=scores.__getitem__)
                hi = max(range(len(scores)), key=scores.__getitem__)
                if scores[hi] - scores[lo] > config['preference_min_gap'] and tokens[hi] != tokens[lo]:
                    break
            total += 1
            if scores[hi] - scores[lo] <= config['preference_min_gap'] or tokens[hi] == tokens[lo]:
                raise ValueError(
                    f'No strict preference for training prompt {index} after '
                    f"{config['preference_resample_rounds']} sampling rounds")
            record = dict(source_index=index, sample_sha256=C.sample_hash(sample),
                          chosen_ids=tokens[hi], rejected_ids=tokens[lo])
            handle.write(json.dumps(record) + '\n')
            kept += 1
            print(f'processed={total} pairs={kept}', flush=True)
    manifest = dict(config=config, config_sha256=C.file_hash(config_path),
                    train_sha256=C.file_hash(cfg.train_file),
                    preferences_sha256=C.file_hash(output), prompts=total, pairs=kept,
                    source='SFT-generated completions ranked by FACES gated reward')
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not kept:
        raise ValueError('No strict preference pairs were produced')


if __name__ == '__main__':
    main()
