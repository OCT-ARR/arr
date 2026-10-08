#!/usr/bin/env python3
"""
Regenerate MESC generation outputs for FACES and FIRES (Baseline).

Run from repo root or this directory. The script writes JSONL files under:
  experiments/generation/outputs/generations/

Example:
  CUDA_VISIBLE_DEVICES=0,1 python experiments/generation/generate_mesc_outputs.py

If only one GPU is visible, both models run on `cuda:0` sequentially.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STICKERCONV_DIR = PROJECT_ROOT / "experiments" / "llm_blind_judge"

if str(STICKERCONV_DIR) not in sys.path:
    sys.path.insert(0, str(STICKERCONV_DIR))

import stickerconv_eval as base_eval  # noqa: E402


TEST_FILE = PROJECT_ROOT / "data" / "MESC-FT" / "test.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
GENERATION_DIR = OUTPUT_DIR / "generations"

CANDIDATE_MODELS = {
    "faces_main": PROJECT_ROOT / "checkpoints" / "faces_main",
    "grpo_baseline": PROJECT_ROOT / "checkpoints" / "grpo_baseline",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate MESC candidate outputs for FACES and FIRES (Baseline)."
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=0,
        help="0 means use the full MESC test set.",
    )
    parser.add_argument(
        "--force_regenerate",
        dest="force_regenerate",
        action="store_true",
        help="Regenerate outputs even if JSONL caches already exist.",
    )
    parser.add_argument(
        "--reuse_cache",
        dest="force_regenerate",
        action="store_false",
        help="Reuse existing JSONL caches when available.",
    )
    parser.set_defaults(force_regenerate=True)
    return parser.parse_args()


def configure_generation() -> None:
    visible_gpus = torch.cuda.device_count()
    if visible_gpus >= 2:
        candidate_devices = {
            "faces_main": "cuda:0",
            "grpo_baseline": "cuda:1",
        }
        parallel_generation = True
    elif visible_gpus == 1:
        candidate_devices = {
            "faces_main": "cuda:0",
            "grpo_baseline": "cuda:0",
        }
        parallel_generation = False
    else:
        candidate_devices = {
            "faces_main": "cpu",
            "grpo_baseline": "cpu",
        }
        parallel_generation = False

    base_eval.CANDIDATE_MODELS = CANDIDATE_MODELS
    base_eval.CANDIDATE_DEVICES = candidate_devices
    base_eval.PARALLEL_CANDIDATE_GENERATION = parallel_generation
    base_eval.CANDIDATE_PARALLEL_WORKERS = 0
    base_eval.RUN_GENERATION = True
    base_eval.RUN_POINTWISE = False
    base_eval.RUN_PAIRWISE = False


def main() -> None:
    args = parse_args()
    configure_generation()
    base_eval.FORCE_REGENERATE = args.force_regenerate

    if not TEST_FILE.is_file():
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")
    for name, path in CANDIDATE_MODELS.items():
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Missing model directory for {name}: {path}")

    GENERATION_DIR.mkdir(parents=True, exist_ok=True)

    samples = base_eval.load_test_samples(TEST_FILE, args.sample_limit)
    if not samples:
        raise RuntimeError(f"No valid samples loaded from {TEST_FILE}")

    print(f"Test file: {TEST_FILE}")
    print(f"Samples: {len(samples)}")
    print(f"Visible CUDA devices: {torch.cuda.device_count()}")
    print(f"Output dir: {GENERATION_DIR}")
    for name, path in CANDIDATE_MODELS.items():
        print(f"{name}: {path}")

    base_eval.generate_all_predictions(
        candidate_models=CANDIDATE_MODELS,
        samples=samples,
        generation_dir=GENERATION_DIR,
    )

    print("Done.")


if __name__ == "__main__":
    main()
