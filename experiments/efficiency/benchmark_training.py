#!/usr/bin/env python3
"""Measure wall time and allocated GPU-hours for one controlled training command."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def gpu_metadata(gpus):
    try:
        output = subprocess.check_output([
            'nvidia-smi', f'--id={gpus}', '--query-gpu=index,name,memory.total',
            '--format=csv,noheader,nounits'], text=True)
        return [line.strip() for line in output.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--stable_steps', type=int, default=100)
    parser.add_argument('--output', type=Path,
                        default=root / 'outputs/efficiency/training_cost.json')
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='Training command after --')
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    gpu_ids = [x.strip() for x in args.gpus.split(',') if x.strip()]
    if not command or not gpu_ids or args.stable_steps <= 0:
        parser.error('A command, at least one GPU, and positive stable_steps are required')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=root, env=env)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    elapsed = time.perf_counter() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    record = {
        'name': args.name,
        'stable_training_steps': args.stable_steps,
        'wall_seconds': elapsed,
        'wall_hours': elapsed / 3600.0,
        'allocated_gpu_count': len(gpu_ids),
        'allocated_gpu_hours': elapsed * len(gpu_ids) / 3600.0,
        'gpu_inventory': gpu_metadata(args.gpus)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.output.read_text()) if args.output.exists() else {'runs': []}
    existing['runs'] = [x for x in existing.get('runs', []) if x.get('name') != args.name]
    existing['runs'].append(record)
    args.output.write_text(json.dumps(existing, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
