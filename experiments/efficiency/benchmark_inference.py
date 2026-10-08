#!/usr/bin/env python3
"""Benchmark FACES and FIRES generation latency under one controlled protocol."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.llm_blind_judge import stickerconv_eval as eval_module  # noqa: E402


DEFAULT_MODELS = {
    "FACES": ROOT / "checkpoints" / "faces_main",
    "FIRES": ROOT / "checkpoints" / "grpo_baseline",
}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def benchmark_model(
    name: str,
    model_path: Path,
    samples: list,
    device: str,
    warmup: int,
) -> dict:
    print(f"[load] {name}: {model_path}", flush=True)
    load_start = time.perf_counter()
    runner = eval_module.VisionLanguageRunner(str(model_path), device)
    synchronize(device)
    load_seconds = time.perf_counter() - load_start

    try:
        for sample in samples[:warmup]:
            runner.generate_chat(
                messages=eval_module.build_candidate_messages(sample),
                image_paths=sample.image_paths,
                max_new_tokens=eval_module.GEN_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                max_images=eval_module.MAX_IMAGES,
            )

        synchronize(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

        latencies: list[float] = []
        output_tokens: list[int] = []
        for index, sample in enumerate(samples, start=1):
            synchronize(device)
            start = time.perf_counter()
            output = runner.generate_chat(
                messages=eval_module.build_candidate_messages(sample),
                image_paths=sample.image_paths,
                max_new_tokens=eval_module.GEN_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                max_images=eval_module.MAX_IMAGES,
            )
            synchronize(device)
            elapsed = time.perf_counter() - start
            token_count = len(
                runner.processor.tokenizer.encode(output, add_special_tokens=False)
            )
            latencies.append(elapsed)
            output_tokens.append(token_count)
            if index % 10 == 0 or index == len(samples):
                print(
                    f"[{name}] {index}/{len(samples)} "
                    f"mean={statistics.mean(latencies):.3f}s",
                    flush=True,
                )

        peak_gb = None
        if device.startswith("cuda"):
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        total_seconds = sum(latencies)
        total_tokens = sum(output_tokens)
        return {
            "model": name,
            "model_path": str(model_path),
            "samples": len(samples),
            "warmup_samples": warmup,
            "load_seconds_excluded": load_seconds,
            "mean_seconds": statistics.mean(latencies),
            "std_seconds": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "median_seconds": statistics.median(latencies),
            "p95_seconds": percentile(latencies, 0.95),
            "total_generation_seconds": total_seconds,
            "total_output_tokens": total_tokens,
            "output_tokens_per_second": total_tokens / total_seconds,
            "mean_output_tokens": statistics.mean(output_tokens),
            "peak_allocated_gb": peak_gb,
        }
    finally:
        runner.close()
        del runner
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--faces_model", type=Path, default=DEFAULT_MODELS["FACES"])
    parser.add_argument("--fires_model", type=Path, default=DEFAULT_MODELS["FIRES"])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "efficiency" / "inference_latency.json",
    )
    args = parser.parse_args()

    for path in (args.faces_model, args.fires_model):
        if not path.is_dir():
            raise FileNotFoundError(path)
    all_samples = eval_module.load_test_samples(ROOT / "data" / "MESC-FT" / "test.jsonl", 0)
    if args.samples <= 0 or args.samples > len(all_samples):
        raise ValueError(f"--samples must be between 1 and {len(all_samples)}")
    sample_count = args.samples
    random.Random(args.seed).shuffle(all_samples)
    samples = all_samples[:sample_count]

    hardware = {"device": args.device}
    if args.device.startswith("cuda"):
        hardware["gpu_name"] = torch.cuda.get_device_name(args.device)
        hardware["gpu_total_gb"] = torch.cuda.get_device_properties(args.device).total_memory / (1024**3)

    results = []
    for name, path in (("FACES", args.faces_model), ("FIRES", args.fires_model)):
        results.append(
            benchmark_model(name, path, samples, args.device, max(0, args.warmup))
        )

    payload = {
        "protocol": {
            "test_file": str(ROOT / "data" / "MESC-FT" / "test.jsonl"),
            "seed": args.seed,
            "sample_ids": [sample.sample_id for sample in samples],
            "batch_size": 1,
            "do_sample": False,
            "max_new_tokens": eval_module.GEN_MAX_NEW_TOKENS,
            "max_images": eval_module.MAX_IMAGES,
            "excludes_model_loading_and_warmup": True,
        },
        "hardware": hardware,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
