#!/usr/bin/env python3
"""Train DPO or SimPO from fixed reward-ranked multimodal preference pairs."""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_and_verify(preferences, manifest_path, config_path, dataset, train_file, common):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('config_sha256') != common.file_hash(config_path):
        raise ValueError('Baseline config does not match the preference manifest')
    if manifest['train_sha256'] != common.file_hash(train_file):
        raise ValueError('Training split does not match the preference manifest')
    if manifest['preferences_sha256'] != common.file_hash(preferences):
        raise ValueError('Preference file does not match its manifest')
    records = []
    for line_number, line in enumerate(preferences.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        required = {'source_index', 'sample_sha256', 'chosen_ids', 'rejected_ids'}
        if not required <= item.keys():
            raise ValueError(f'Missing fields on preference line {line_number}')
        index = item['source_index']
        if not isinstance(index, int) or not 0 <= index < len(dataset):
            raise ValueError(f'Invalid source_index on line {line_number}')
        if common.sample_hash(dataset[index]) != item['sample_sha256']:
            raise ValueError(f'Source sample changed for preference line {line_number}')
        for key in ('chosen_ids', 'rejected_ids'):
            if not item[key] or not all(isinstance(x, int) and x >= 0 for x in item[key]):
                raise ValueError(f'Invalid {key} on preference line {line_number}')
        records.append(item)
    if not records:
        raise ValueError('Preference dataset is empty')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['dpo', 'simpo'], required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/optimization_baselines.json')
    parser.add_argument('--preferences', type=Path, default=ROOT / 'outputs/optimization_baselines/preferences.jsonl')
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--validate_only', action='store_true')
    args = parser.parse_args()
    args.config, args.preferences = args.config.resolve(), args.preferences.resolve()
    manifest_path = args.preferences.with_suffix('.manifest.json')
    if not args.config.is_file() or not args.preferences.is_file() or not manifest_path.is_file():
        raise FileNotFoundError('Config, preferences, or preference manifest is missing')
    os.chdir(ROOT)
    from experiments.optimization_baselines import common as C
    from experiments.optimization_baselines.objectives import dpo_loss, simpo_loss
    from src.grpo.faces_grpo import train_faces_grpo as T
    config = C.load_config(args.config, args.method)
    cfg = C.train_config(config)
    dataset = T.JsonlConversationDataset(cfg.train_file)
    records = read_and_verify(args.preferences, manifest_path, args.config,
                              dataset, cfg.train_file, C)
    if args.validate_only:
        print(json.dumps({'method': args.method, 'validated_pairs': len(records)}))
        return

    import torch
    from transformers import get_cosine_schedule_with_warmup
    T.set_seed(cfg.seed)
    base, processor = T.build_policy_model_and_processor(cfg)
    model = T.create_new_lora_model(base, cfg)
    C.disable_dropout(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    batches = math.ceil(len(records) / cfg.batch_size)
    total_steps = max(1, batches * cfg.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps)
    method_cfg = config['methods'][args.method]
    global_step = 0
    for epoch in range(cfg.epochs):
        order = list(range(len(records)))
        random.Random(cfg.seed + epoch).shuffle(order)
        for offset in range(0, len(order), cfg.batch_size):
            batch = order[offset:offset + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for record_index in batch:
                record = records[record_index]
                sample = dataset[record['source_index']]
                inputs, _ = C.prompt_inputs(processor, sample, cfg, config['max_sequence_length'])
                chosen = C.sequence(inputs, record['chosen_ids'], cfg, config['max_sequence_length'])
                rejected = C.sequence(inputs, record['rejected_ids'], cfg, config['max_sequence_length'])
                model.train()
                chosen_logp, _ = C.forward(model, chosen, cfg.policy_device)
                rejected_logp, _ = C.forward(model, rejected, cfg.policy_device)
                if args.method == 'dpo':
                    with torch.no_grad(), model.disable_adapter():
                        ref_chosen, _ = C.forward(model, chosen, cfg.policy_device)
                        ref_rejected, _ = C.forward(model, rejected, cfg.policy_device)
                    loss = dpo_loss(chosen_logp.sum(), rejected_logp.sum(),
                                    ref_chosen.sum(), ref_rejected.sum(), method_cfg['beta'])
                else:
                    loss = simpo_loss(chosen_logp.mean(), rejected_logp.mean(),
                                      method_cfg['beta'], method_cfg['gamma'])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at record {record_index}')
                (loss / len(batch)).backward()
                loss_total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            print(f'method={args.method} epoch={epoch + 1}/{cfg.epochs} '
                  f'step={global_step}/{total_steps} loss={loss_total / len(batch):.6f}', flush=True)

    output = (args.output_dir or ROOT / 'checkpoints' / 'optimization_baselines' / args.method).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / 'adapter')
    processor.save_pretrained(output)
    if args.method == 'dpo' and not hasattr(model, 'disable_adapter'):
        raise RuntimeError('DPO reference adapter context was unavailable')
    (output / 'base_model_path.txt').write_text(cfg.base_model_path + '\n')
    (output / 'training_recipe.json').write_text(json.dumps({
        'method': args.method, 'steps': global_step, 'epochs': cfg.epochs,
        'preference_manifest_sha256': C.file_hash(manifest_path),
        'method_hyperparameters': method_cfg}, indent=2) + '\n')


if __name__ == '__main__':
    main()
