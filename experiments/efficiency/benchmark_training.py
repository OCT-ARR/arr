#!/usr/bin/env python3
"""Measure training-step wall time and allocated GPU-hours for one training command."""
import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


def read_step_seconds(path, expected_steps):
    """Require measured, consecutive steps rather than a declared step count."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_steps:
        raise ValueError('The timing report must contain exactly the requested training steps')
    durations = []
    for expected, row in enumerate(rows, start=1):
        if type(row.get('step')) is not int or row['step'] != expected:
            raise ValueError('Timing report steps must be consecutive integers starting at one')
        duration = row.get('wall_seconds')
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('Each training-step duration must be finite and positive')
        durations.append(duration)
    return math.fsum(durations)


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
    if not command or not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or args.stable_steps <= 0:
        parser.error('A command, unique GPU identifiers, and positive stable_steps are required')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
    with tempfile.TemporaryDirectory(prefix='training_cost_') as temp_dir:
        timings = Path(temp_dir) / 'steps.jsonl'
        env['TRAINING_BENCHMARK_TIMINGS_FILE'] = str(timings)
        env['TRAINING_BENCHMARK_STEPS'] = str(args.stable_steps)
        process = subprocess.Popen(command, cwd=root, env=env)
        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        if return_code:
            raise SystemExit(f'{args.name}: training command failed (exit {return_code})')
        try:
            elapsed = read_step_seconds(timings, args.stable_steps)
        except (OSError, ValueError, TypeError, AttributeError):
            parser.error('Missing or invalid training-step timing report; see the efficiency README')
    record = {
        'name': args.name,
        'stable_training_steps': args.stable_steps,
        'timing_scope': 'training_steps_only',
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
