#!/usr/bin/env python3
"""
Controlled pairwise evaluation for StickerConv outputs.

Reuses existing JSONL generations and runs controlled judge views:
1. full_context: conversation + images + think + response
2. response_only: conversation + images + response
3. length_controlled: conversation + images + prefix-truncated response

The pairwise protocol remains:
3 prompt variants + bidirectional swap + majority vote.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEFT_NAME = "faces_main"
RIGHT_NAME = "grpo_baseline"
PAIRWISE_MODES = ("full_context", "response_only", "length_controlled")


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    test_file: Path
    left_generation_file: Path
    right_generation_file: Path


DATASETS: Dict[str, DatasetConfig] = {
    "mesc": DatasetConfig(
        key="mesc",
        display_name="MESC",
        test_file=PROJECT_ROOT / "data" / "MESC-FT" / "test.jsonl",
        left_generation_file=PROJECT_ROOT
        / "experiments/llm_blind_judge"
        / "outputs"
        / "generations"
        / f"{LEFT_NAME}.jsonl",
        right_generation_file=PROJECT_ROOT
        / "experiments/llm_blind_judge"
        / "outputs"
        / "generations"
        / f"{RIGHT_NAME}.jsonl",
    ),
    "emotyda": DatasetConfig(
        key="emotyda",
        display_name="MELD-ESC300",
        test_file=PROJECT_ROOT
        / "data"
        / "MELD-ESC300"
        / "test.jsonl",
        left_generation_file=PROJECT_ROOT
        / "experiments/generalization"
        / "outputs_emotyda"
        / "generations"
        / f"{LEFT_NAME}.jsonl",
        right_generation_file=PROJECT_ROOT
        / "experiments/generalization"
        / "outputs_emotyda"
        / "generations"
        / f"{RIGHT_NAME}.jsonl",
    ),
}



for key, display_name, folder in [("stickerconv", "StickerConv", "StickerConv"), ("avamerg", "AvaMERG", "AvaMERG")]:
    generation_dir = PROJECT_ROOT / "experiments" / "generalization" / "outputs" / key / "generations"
    DATASETS[key] = DatasetConfig(key, display_name, PROJECT_ROOT / "data" / folder / "test.jsonl",
                                 generation_dir / f"{LEFT_NAME}.jsonl", generation_dir / f"{RIGHT_NAME}.jsonl")
DATASETS["meld"] = DATASETS["emotyda"]


def load_base_module():
    from experiments.llm_blind_judge import stickerconv_eval
    return stickerconv_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled StickerConv pairwise experiments on existing JSONL outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["mesc", "emotyda"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(PAIRWISE_MODES),
        default=["length_controlled"],
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=15,
        help="Word cap for length_controlled mode; the paper's main setting uses 15.",
    )
    parser.add_argument(
        "--length_control_strategy",
        choices=["cap", "matched"],
        default="cap",
        help=(
            "cap truncates each response independently to max_words; "
            "matched truncates both responses to the same per-sample length."
        ),
    )
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=SCRIPT_DIR / "outputs",
    )
    parser.add_argument(
        "--judge_model_path",
        type=Path,
        default=PROJECT_ROOT / "models" / "judge-32b",
    )
    parser.add_argument(
        "--judge_devices",
        nargs="+",
        default=["cuda:0"],
    )
    parser.add_argument("--judge_parallel_workers", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--left_name", default=LEFT_NAME)
    parser.add_argument("--right_name", default=RIGHT_NAME)
    parser.add_argument("--left_generation_file", type=Path)
    parser.add_argument("--right_generation_file", type=Path)
    return parser.parse_args()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def count_words(text: Any) -> int:
    raw = str(text or "").strip()
    return 0 if not raw else len(raw.split())


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    values = list(values)
    return float(statistics.mean(values)) if values else None


def median_or_none(values: Sequence[float]) -> Optional[float]:
    values = list(values)
    return float(statistics.median(values)) if values else None


def truncate_words(text: Any, limit: int) -> str:
    raw = str(text or "").strip()
    if limit <= 0 or not raw:
        return ""
    words = raw.split()
    return " ".join(words[:limit]).strip()


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "::".join(parts).encode("utf-8")
    return int(base_seed + (zlib.crc32(payload) % 100000))


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[Optional[float], Optional[float]]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def exact_two_sided_sign_test(left_wins: int, right_wins: int) -> float:
    decisive = left_wins + right_wins
    if decisive == 0:
        return 1.0
    tail = min(left_wins, right_wins)
    lower_prob = sum(math.comb(decisive, i) for i in range(tail + 1)) / (2**decisive)
    return float(min(1.0, 2.0 * lower_prob))


def load_predictions(base: Any, generation_file: Path, model_name: str, samples: Sequence[Any]) -> List[Any]:
    records = {}
    for line in generation_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status", "ok") != "ok":
            continue
        sample_id = int(row.get("source_index", row.get("sample_id")))
        records[sample_id] = row
    missing = [sample.sample_id for sample in samples if sample.sample_id not in records]
    if missing:
        raise ValueError(
            f"{generation_file} is missing {len(missing)} aligned samples. "
            f"First missing ids: {missing[:10]}"
        )

    predictions: List[Any] = []
    for sample in samples:
        row = records[sample.sample_id]
        raw_output = str(row.get("raw_output") or "")
        if not raw_output:
            raw_output = "<response>" + str(row.get("final_response") or row.get("response") or "") + "</response>"
        pred = base.build_model_prediction(sample.sample_id, model_name, raw_output)
        pred.think = str(row.get("think") or pred.think or "")
        pred.response = str(row.get("response") or pred.response or "")
        pred.format_valid = bool(row.get("format_valid", pred.format_valid))
        pred.think_missing = bool(row.get("think_missing", pred.think_missing))
        pred.response_missing = bool(row.get("response_missing", pred.response_missing))
        predictions.append(pred)
    return predictions


def clone_prediction(pred: Any, think: Optional[str] = None, response: Optional[str] = None) -> Any:
    new_think = pred.think if think is None else think
    new_response = pred.response if response is None else response
    return replace(
        pred,
        think=new_think,
        response=new_response,
        think_missing=(str(new_think).strip() == ""),
        response_missing=(str(new_response).strip() == ""),
        format_valid=(str(new_think).strip() != "" and str(new_response).strip() != ""),
    )


def mean_word_count(preds: Sequence[Any], field: str) -> Optional[float]:
    return mean_or_none([count_words(getattr(pred, field, "")) for pred in preds])


def mean_visible_words(preds: Sequence[Any]) -> Optional[float]:
    return mean_or_none(
        [count_words(getattr(pred, "think", "")) + count_words(getattr(pred, "response", "")) for pred in preds]
    )


def build_control_predictions(
    mode: str,
    left_preds: Sequence[Any],
    right_preds: Sequence[Any],
    max_words: int,
    length_control_strategy: str,
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    left_preds = list(left_preds)
    right_preds = list(right_preds)
    if len(left_preds) != len(right_preds):
        raise ValueError(
            f"Prediction count mismatch: {len(left_preds)} left vs {len(right_preds)} right"
        )

    left_targets: Optional[List[int]] = None
    right_targets: Optional[List[int]] = None
    if mode == "full_context":
        left_control = left_preds
        right_control = right_preds
        visible_rule = "conversation + images + think + response"
        length_target_rule = None
    elif mode == "response_only":
        left_control = [clone_prediction(pred, think="") for pred in left_preds]
        right_control = [clone_prediction(pred, think="") for pred in right_preds]
        visible_rule = "conversation + images + response"
        length_target_rule = None
    elif mode == "length_controlled":
        left_control = []
        right_control = []
        left_targets = []
        right_targets = []
        for left_pred, right_pred in zip(left_preds, right_preds):
            left_len = count_words(left_pred.response)
            right_len = count_words(right_pred.response)
            if length_control_strategy == "cap":
                left_target = min(max_words, left_len)
                right_target = min(max_words, right_len)
                length_target_rule = "left=min(max_words, len(left_response)); right=min(max_words, len(right_response))"
            elif length_control_strategy == "matched":
                shared_target = min(max_words, left_len, right_len)
                left_target = shared_target
                right_target = shared_target
                length_target_rule = "left=right=min(max_words, len(left_response), len(right_response))"
            else:
                raise ValueError(f"Unknown length_control_strategy: {length_control_strategy}")
            left_targets.append(left_target)
            right_targets.append(right_target)
            left_control.append(
                clone_prediction(
                    left_pred,
                    think="",
                    response=truncate_words(left_pred.response, left_target),
                )
            )
            right_control.append(
                clone_prediction(
                    right_pred,
                    think="",
                    response=truncate_words(right_pred.response, right_target),
                )
            )
        visible_rule = "conversation + images + prefix-truncated response"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    meta: Dict[str, Any] = {
        "mode": mode,
        "visible_text_rule": visible_rule,
        "think_visible": mode == "full_context",
        "think_removed": mode != "full_context",
        "response_length_control": mode == "length_controlled",
        "response_word_cap": max_words if mode == "length_controlled" else None,
        "length_control_strategy": length_control_strategy if mode == "length_controlled" else None,
        "length_target_rule": length_target_rule,
        "truncation_rule": "prefix_first_words" if mode == "length_controlled" else None,
        "mean_left_original_think_words": mean_word_count(left_preds, "think"),
        "mean_right_original_think_words": mean_word_count(right_preds, "think"),
        "mean_left_original_response_words": mean_word_count(left_preds, "response"),
        "mean_right_original_response_words": mean_word_count(right_preds, "response"),
        "mean_left_visible_think_words": mean_word_count(left_control, "think"),
        "mean_right_visible_think_words": mean_word_count(right_control, "think"),
        "mean_left_visible_response_words": mean_word_count(left_control, "response"),
        "mean_right_visible_response_words": mean_word_count(right_control, "response"),
        "mean_left_visible_total_words": mean_visible_words(left_control),
        "mean_right_visible_total_words": mean_visible_words(right_control),
    }
    if left_targets is not None and right_targets is not None:
        meta.update(
            {
                "mean_left_target_words": mean_or_none(left_targets),
                "mean_right_target_words": mean_or_none(right_targets),
                "median_left_target_words": median_or_none(left_targets),
                "median_right_target_words": median_or_none(right_targets),
                "min_left_target_words": float(min(left_targets)) if left_targets else None,
                "min_right_target_words": float(min(right_targets)) if right_targets else None,
                "max_left_target_words": float(max(left_targets)) if left_targets else None,
                "max_right_target_words": float(max(right_targets)) if right_targets else None,
            }
        )
    return left_control, right_control, meta


def experiment_dir_name(mode: str, max_words: int, length_control_strategy: str) -> str:
    if mode != "length_controlled":
        return mode
    return f"{mode}_{length_control_strategy}{max_words}"


def enrich_summary(
    base: Any,
    config: DatasetConfig,
    mode: str,
    samples: Sequence[Any],
    output_path: Path,
    summary_path: Path,
    control_meta: Dict[str, Any],
    judge_devices: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    records = list(base.ordered_cached_records(base.load_jsonl_records(output_path), samples))
    records.sort(key=lambda row: int(row["sample_id"]))
    if len(records) != len(samples):
        raise RuntimeError(f"{output_path} has {len(records)} records, expected {len(samples)}.")

    base_summary = base.summarize_pairwise(records, LEFT_NAME, RIGHT_NAME)
    left_indicator = [1.0 if row["winner_name"] == LEFT_NAME else 0.0 for row in records]
    decisive_indicator = [
        1.0 if row["winner_name"] == LEFT_NAME else 0.0
        for row in records
        if row["winner_name"] in {LEFT_NAME, RIGHT_NAME}
    ]
    left_ci_low, left_ci_high = bootstrap_ci(
        left_indicator,
        bootstrap_samples,
        stable_seed(
            seed,
            config.key,
            mode,
            str(control_meta.get("length_control_strategy")),
            str(control_meta.get("response_word_cap")),
            "left_win_rate",
        ),
    )
    decisive_ci_low, decisive_ci_high = bootstrap_ci(
        decisive_indicator,
        bootstrap_samples,
        stable_seed(
            seed,
            config.key,
            mode,
            str(control_meta.get("length_control_strategy")),
            str(control_meta.get("response_word_cap")),
            "decisive_left_share",
        ),
    )

    left_wins = int(base_summary["left_wins"])
    right_wins = int(base_summary["right_wins"])
    decisive_n = left_wins + right_wins
    summary: Dict[str, Any] = {
        **base_summary,
        "dataset": config.display_name,
        "dataset_key": config.key,
        "experiment": mode,
        "protocol": {
            "judge_model_path": str(base.JUDGE_MODEL_PATH),
            "judge_devices": list(judge_devices),
            "prompt_variants": 3,
            "bidirectional_swap": True,
            "majority_vote": True,
        },
        "sources": {
            "test_file": str(config.test_file),
            "left_generation_file": str(config.left_generation_file),
            "right_generation_file": str(config.right_generation_file),
        },
        "control": control_meta,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_ci95_left_win_rate": [left_ci_low, left_ci_high] if left_ci_low is not None else None,
        "bootstrap_ci95_decisive_left_share": [decisive_ci_low, decisive_ci_high]
        if decisive_ci_low is not None
        else None,
        "decisive_n": decisive_n,
        "decisive_left_share": (left_wins / decisive_n) if decisive_n else None,
        "sign_test_method": "two-sided exact binomial test on decisive (non-tie) samples",
        "sign_test_p_value": exact_two_sided_sign_test(left_wins, right_wins),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
    }
    write_json(summary_path, summary)
    return summary


def run_pairwise_experiment(
    base: Any,
    config: DatasetConfig,
    mode: str,
    samples: Sequence[Any],
    left_preds: Sequence[Any],
    right_preds: Sequence[Any],
    output_root: Path,
    judge_devices: Sequence[str],
    bootstrap_samples: int,
    seed: int,
    max_words: int,
    length_control_strategy: str,
    dry_run: bool,
    judge_model_path: Path,
    judge_parallel_workers: int,
    force_rerun: bool,
) -> Dict[str, Any]:
    mode_dir = experiment_dir_name(mode, max_words, length_control_strategy)
    mode_root = output_root / config.key / mode_dir / "pairwise"
    output_path = mode_root / f"{LEFT_NAME}_vs_{RIGHT_NAME}.jsonl"
    summary_path = mode_root / f"{LEFT_NAME}_vs_{RIGHT_NAME}_summary.json"
    if force_rerun and not dry_run:
        output_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    controlled_left, controlled_right, control_meta = build_control_predictions(
        mode=mode,
        left_preds=left_preds,
        right_preds=right_preds,
        max_words=max_words,
        length_control_strategy=length_control_strategy,
    )

    if dry_run:
        plan = {
            "dataset": config.display_name,
            "dataset_key": config.key,
            "experiment": mode,
            "experiment_dir": mode_dir,
            "sample_count": len(samples),
            "sources": {
                "test_file": str(config.test_file),
                "left_generation_file": str(config.left_generation_file),
                "right_generation_file": str(config.right_generation_file),
            },
            "control": control_meta,
            "output_path": str(output_path),
            "summary_path": str(summary_path),
        }
        write_json(mode_root / "dry_run_plan.json", plan)
        print(f"[DRY] {config.display_name} / {mode}: {len(samples)} samples -> {output_path}")
        return plan

    if not Path(judge_model_path).is_dir():
        raise FileNotFoundError(f"Judge model not found: {judge_model_path}")

    base.JUDGE_MODEL_PATH = Path(judge_model_path)
    base.JUDGE_PARALLEL_WORKERS = judge_parallel_workers
    mode_root.mkdir(parents=True, exist_ok=True)
    base.run_pairwise_parallel(
        samples=samples,
        left_predictions=controlled_left,
        right_predictions=controlled_right,
        left_name=LEFT_NAME,
        right_name=RIGHT_NAME,
        output_path=output_path,
        summary_path=summary_path,
        judge_devices=judge_devices,
    )
    summary = enrich_summary(
        base=base,
        config=config,
        mode=mode,
        samples=samples,
        output_path=output_path,
        summary_path=summary_path,
        control_meta=control_meta,
        judge_devices=judge_devices,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    print(
        f"[DONE] {config.display_name} / {mode}: "
        f"L={summary['left_wins']} R={summary['right_wins']} T={summary['ties']} "
        f"p={summary['sign_test_p_value']:.4g}"
    )
    return summary


def load_dataset_samples_and_predictions(
    base: Any,
    config: DatasetConfig,
    sample_limit: int,
) -> Tuple[List[Any], List[Any], List[Any]]:
    samples = base.load_test_samples(config.test_file, sample_limit)
    if not samples:
        raise RuntimeError(f"No samples loaded from {config.test_file}")
    left_preds = load_predictions(base, config.left_generation_file, LEFT_NAME, samples)
    right_preds = load_predictions(base, config.right_generation_file, RIGHT_NAME, samples)
    return samples, left_preds, right_preds


def main() -> None:
    global LEFT_NAME, RIGHT_NAME
    args = parse_args()
    LEFT_NAME, RIGHT_NAME = args.left_name, args.right_name
    if (args.left_generation_file or args.right_generation_file) and len(args.datasets) != 1:
        raise ValueError("Prediction file overrides require exactly one dataset")
    if args.sample_limit < 0:
        raise ValueError("--sample_limit must be >= 0")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive")
    if "length_controlled" in args.modes and args.max_words <= 0:
        raise ValueError("--max_words must be positive")

    base = load_base_module()
    results: List[Dict[str, Any]] = []

    for dataset_key in args.datasets:
        original_config = DATASETS[dataset_key]
        config = DatasetConfig(original_config.key, original_config.display_name, original_config.test_file,
                               args.left_generation_file or original_config.left_generation_file,
                               args.right_generation_file or original_config.right_generation_file)
        if not config.test_file.is_file():
            raise FileNotFoundError(f"Missing test file: {config.test_file}")
        if not config.left_generation_file.is_file():
            raise FileNotFoundError(f"Missing generation file: {config.left_generation_file}")
        if not config.right_generation_file.is_file():
            raise FileNotFoundError(f"Missing generation file: {config.right_generation_file}")

        samples, left_preds, right_preds = load_dataset_samples_and_predictions(
            base=base,
            config=config,
            sample_limit=args.sample_limit,
        )

        for mode in args.modes:
            summary = run_pairwise_experiment(
                base=base,
                config=config,
                mode=mode,
                samples=samples,
                left_preds=left_preds,
                right_preds=right_preds,
                output_root=args.output_root,
                judge_devices=args.judge_devices,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                max_words=args.max_words,
                length_control_strategy=args.length_control_strategy,
                dry_run=args.dry_run,
                judge_model_path=args.judge_model_path,
                judge_parallel_workers=args.judge_parallel_workers,
                force_rerun=args.force_rerun,
            )
            results.append(summary)

    if args.dry_run:
        plan_path = args.output_root / "dry_run_plan.json"
        write_json(plan_path, {"results": results})
        print(f"[DRY] Wrote plan to {plan_path}")
        return

    aggregate_path = args.output_root / "aggregate_summary.json"
    write_json(aggregate_path, {"results": results})
    print(f"[DONE] Wrote aggregate summary to {aggregate_path}")


if __name__ == "__main__":
    main()
