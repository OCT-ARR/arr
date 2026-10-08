#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from api_baselines.clients import APIClientError, build_client
from api_baselines.data import extract_response, load_dataset
from api_baselines.env import load_dotenv
from api_baselines.io_utils import append_jsonl, ensure_dir, read_json, read_jsonl, write_json
from api_baselines.prompting import build_generation_prompt


def parse_list(value: str) -> list[str]:
    if value.lower() == "all":
        return ["all"]
    return [x.strip() for x in value.split(",") if x.strip()]


def load_completed_ids(path: Path, skip_errors: bool) -> set[str]:
    completed: set[str] = set()
    for row in read_jsonl(path):
        status = row.get("status")
        final_response = extract_response(
            str(row.get("final_response") or row.get("raw_output") or "")
        )
        if status == "ok" and final_response.strip():
            completed.add(row.get("sample_id", ""))
        elif skip_errors and status == "error":
            completed.add(row.get("sample_id", ""))
    return completed


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def call_with_retries(client, prompt: str, images: list[str], generation: dict[str, Any], retries: int, sleep_sec: float):
    last_error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            result = client.generate(prompt, images, generation)
            if not extract_response(result.text).strip():
                raise APIClientError("API returned an empty final response.")
            return result
        except Exception as exc:
            last_error = str(exc) if isinstance(exc, APIClientError) else type(exc).__name__
            if attempt <= retries:
                time.sleep(sleep_sec * attempt)
    raise APIClientError(last_error or "Unknown API error")


def run_model_on_dataset(
    model_name: str,
    model_config: dict[str, Any],
    dataset_name: str,
    dataset_config: dict[str, Any],
    prompt_config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    dataset_path = REPO_ROOT / dataset_config["path"]
    samples = load_dataset(
        dataset_name,
        dataset_path,
        limit=args.limit,
    )
    if args.offset:
        samples = samples[args.offset :]

    out_dir = ensure_dir(PROJECT_DIR / "outputs" / "raw" / dataset_name)
    out_path = out_dir / f"{model_name}.jsonl"
    meta_path = out_dir / f"{model_name}.meta.json"

    effective_prompt_config = dict(prompt_config)
    effective_prompt_config.update(model_config.get("prompt_overrides", {}))
    model_generation = dict(model_config.get("generation", {}))
    generation = {
        "temperature": args.temperature,
        "max_tokens": model_generation.get("max_tokens", args.max_tokens),
        "max_images": model_generation.get("max_images", args.max_images),
    }
    completed = load_completed_ids(out_path, skip_errors=args.skip_errors)
    pending = [s for s in samples if s.sample_id not in completed]

    write_json(
        meta_path,
        {
            "model_name": model_name,
            "model_config": {
                k: model_config[k]
                for k in ("name", "provider", "model", "generation", "prompt_overrides")
                if k in model_config
            },
            "dataset_name": dataset_name,
            "dataset_path": dataset_config["path"],
            "prompt_version": effective_prompt_config.get("version"),
            "generation": generation,
            "limit": args.limit,
            "offset": args.offset,
            "total_loaded": len(samples),
            "already_completed": len(completed),
            "pending": len(pending),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    print(f"\nModel: {model_name} | Dataset: {dataset_name}")
    print(f"Output: {out_path}")
    print(f"Loaded: {len(samples)} | Completed: {len(completed)} | Pending: {len(pending)}")

    if args.dry_run:
        for sample in pending[: min(2, len(pending))]:
            prompt = build_generation_prompt(sample, effective_prompt_config)
            print("\n--- DRY RUN SAMPLE ---")
            print(f"sample_id: {sample.sample_id}")
            print(f"images: {len(sample.images)}")
            print(prompt[:1500])
        return

    client = build_client(model_config)
    for sample in tqdm(pending, desc=f"{model_name}/{dataset_name}"):
        prompt = build_generation_prompt(sample, effective_prompt_config)
        started = time.time()
        try:
            result = call_with_retries(
                client,
                prompt,
                sample.images,
                generation,
                retries=args.retries,
                sleep_sec=args.retry_sleep,
            )
            final_response = extract_response(result.text)
            row = {
                "status": "ok",
                "sample_id": sample.sample_id,
                "source_index": sample.source_index,
                "dataset": dataset_name,
                "model_name": model_name,
                "provider": model_config["provider"],
                "api_model": model_config["model"],
                "prompt_version": effective_prompt_config.get("version"),
                "prompt_hash": hash_text(prompt),
                "image_count": len(sample.images),
                "raw_output": result.text,
                "final_response": final_response,
                "latency_sec": result.latency_sec,
                "wall_time_sec": time.time() - started,
                "usage": result.usage,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            row = {
                "status": "error",
                "sample_id": sample.sample_id,
                "source_index": sample.source_index,
                "dataset": dataset_name,
                "model_name": model_name,
                "provider": model_config["provider"],
                "api_model": model_config["model"],
                "prompt_version": effective_prompt_config.get("version"),
                "image_count": len(sample.images),
                "raw_output": "",
                "final_response": "",
                "latency_sec": None,
                "wall_time_sec": time.time() - started,
                "usage": {},
                "error": str(exc) if isinstance(exc, APIClientError) else type(exc).__name__,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        append_jsonl(out_path, row)
        if args.sleep:
            time.sleep(args.sleep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable API baseline generation.")
    parser.add_argument("--models", default="gpt_4o", help="Comma-separated model names or all.")
    parser.add_argument("--datasets", default="mesc", help="Comma-separated dataset names or all.")
    parser.add_argument("--models-config", default="configs/models.json", help="Path to model config JSON, relative to this folder unless absolute.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples per dataset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N loaded samples.")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep after each sample.")
    parser.add_argument("--skip-errors", action="store_true", help="Do not retry existing error rows.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling APIs.")
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    models_config_path = Path(args.models_config)
    if not models_config_path.is_absolute():
        models_config_path = PROJECT_DIR / models_config_path
    models_config = read_json(models_config_path)
    datasets_config = read_json(PROJECT_DIR / "configs" / "datasets.json")
    prompt_config = read_json(PROJECT_DIR / "configs" / "prompt.json")

    model_names = parse_list(args.models)
    dataset_names = parse_list(args.datasets)
    if model_names == ["all"]:
        model_names = list(models_config)
    if dataset_names == ["all"]:
        dataset_names = list(datasets_config)

    for model_name in model_names:
        if model_name not in models_config:
            raise SystemExit(f"Unknown model: {model_name}. Available: {', '.join(models_config)}")
        for dataset_name in dataset_names:
            if dataset_name not in datasets_config:
                raise SystemExit(f"Unknown dataset: {dataset_name}. Available: {', '.join(datasets_config)}")
            run_model_on_dataset(
                model_name,
                models_config[model_name],
                dataset_name,
                datasets_config[dataset_name],
                prompt_config,
                args,
            )


if __name__ == "__main__":
    main()
