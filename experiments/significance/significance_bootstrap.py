#!/usr/bin/env python3
"""
Paired bootstrap significance tests for automatic metrics.

This script compares FACES against FIRES (Baseline) on the full
MESC test set and the full MELD-ESC300 test set. It first computes and caches
per-sample automatic metrics, then runs paired bootstrap resampling over test
instances.

Recommended invocation:
CUDA_VISIBLE_DEVICES=0 python experiments/significance/significance_bootstrap.py --mode all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

LOCAL_NLTK_PATH = PROJECT_ROOT / "nltk_data"
if LOCAL_NLTK_PATH.exists():
    import nltk

    nltk.data.path.insert(0, str(LOCAL_NLTK_PATH))


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    expected_count: int
    test_file: Path
    ours_file: Path
    baseline_file: Path
    cache_file: Path


DATASETS = {
    "mesc": DatasetConfig(
        name="MESC",
        expected_count=450,
        test_file=PROJECT_ROOT / "data" / "MESC-FT" / "test.jsonl",
        ours_file=PROJECT_ROOT
        / "experiments"
        / "llm_blind_judge"
        / "outputs"
        / "generations"
        / "faces_main.jsonl",
        baseline_file=PROJECT_ROOT
        / "experiments"
        / "llm_blind_judge"
        / "outputs"
        / "generations"
        / "grpo_baseline.jsonl",
        cache_file=SCRIPT_DIR / "cache" / "mesc_generate_data_per_sample_metrics.csv",
    ),
    "meld": DatasetConfig(
        name="MELD-ESC300",
        expected_count=300,
        test_file=PROJECT_ROOT
        / "data"
        / "MELD-ESC300"
        / "test.jsonl",
        ours_file=PROJECT_ROOT
        / "experiments"
        / "generalization"
        / "outputs_emotyda"
        / "generations"
        / "faces_main.jsonl",
        baseline_file=PROJECT_ROOT
        / "experiments"
        / "generalization"
        / "outputs_emotyda"
        / "generations"
        / "grpo_baseline.jsonl",
        cache_file=SCRIPT_DIR / "cache" / "meld_per_sample_metrics.csv",
    ),
}

LOCAL_ROBERTA_LARGE = SCRIPT_DIR / "models" / "roberta-large"


METRIC_COLUMNS = {
    "meteor": ("ours_meteor", "baseline_meteor", True, "METEOR"),
    "bertscore": ("ours_bertscore_f1", "baseline_bertscore_f1", True, "BERTScore-F1"),
    "diff_epitome": ("ours_diff_epitome", "baseline_diff_epitome", False, "DIFF-EPITOME"),
    "distinct3": ("ours_response", "baseline_response", True, "Distinct-3"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired bootstrap significance tests for FACES automatic metrics."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "cache", "bootstrap"],
        default="all",
        help="all = compute cache then bootstrap; cache = only compute metrics; bootstrap = reuse cache.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["mesc", "meld"],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_COLUMNS),
        default=["meteor", "bertscore", "diff_epitome"],
        help="Metrics to test. Distinct-3 is optional because it is corpus-level.",
    )
    parser.add_argument(
        "--include_distinct3",
        action="store_true",
        help="Shortcut to include Distinct-3 in addition to selected metrics.",
    )
    parser.add_argument(
        "--distinct3_aggregation",
        choices=["sample", "corpus"],
        default="sample",
        help=(
            "Distinct-3 aggregation. sample = mean per-example Distinct-3 "
            "(paper-style scale); corpus = unique 3-grams / total 3-grams "
            "over each bootstrap resample."
        ),
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=5000,
        help="Number of paired bootstrap resamples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Metric model device. Default: cuda:0 if available, otherwise cpu.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for BERTScore and DIFF-EPITOME.",
    )
    parser.add_argument(
        "--bertscore_model",
        default="roberta-large",
        help="BERTScore backbone. Use the same backbone as your paper if different.",
    )
    parser.add_argument(
        "--bertscore_num_layers",
        type=int,
        default=None,
        help=(
            "BERTScore layer count. Needed for local model paths. "
            "If omitted, the script uses the final layer (24) for roberta-large."
        ),
    )
    parser.add_argument(
        "--bertscore_rescale",
        action="store_true",
        help="Use BERTScore baseline rescaling. Leave off unless your paper used it.",
    )
    parser.add_argument(
        "--reward_model",
        default=str(
            PROJECT_ROOT
            / "checkpoints"
            / "reward_model"
            / "roberta_epitome_multitask"
            / "final"
        ),
        help="Local DIFF-EPITOME reward model path.",
    )
    parser.add_argument(
        "--force_cache",
        action="store_true",
        help="Recompute cached per-sample metrics even if cache files already exist.",
    )
    parser.add_argument(
        "--allow_count_mismatch",
        action="store_true",
        help="Do not fail when aligned sample count differs from the expected full-set size.",
    )
    parser.add_argument(
        "--output_csv",
        default=str(SCRIPT_DIR / "bootstrap_results.csv"),
        help="Path for final bootstrap result CSV.",
    )
    parser.add_argument("--mesc_test_file", default=None, help="Override MESC test JSONL.")
    parser.add_argument("--mesc_ours_file", default=None, help="Override MESC Ours JSONL.")
    parser.add_argument("--mesc_baseline_file", default=None, help="Override MESC FIRES (Baseline) JSONL.")
    parser.add_argument("--mesc_cache_file", default=None, help="Override MESC cache CSV.")
    parser.add_argument("--meld_test_file", default=None, help="Override MELD-ESC300 test JSONL.")
    parser.add_argument("--meld_ours_file", default=None, help="Override MELD-ESC300 Ours JSONL.")
    parser.add_argument("--meld_baseline_file", default=None, help="Override MELD-ESC300 FIRES (Baseline) JSONL.")
    parser.add_argument("--meld_cache_file", default=None, help="Override MELD-ESC300 cache CSV.")
    return parser.parse_args()


def resolve_path(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def resolve_dataset_config(dataset_key: str, args: argparse.Namespace) -> DatasetConfig:
    config = DATASETS[dataset_key]
    prefix = "mesc" if dataset_key == "mesc" else "meld"
    overrides = {}
    for field_name, arg_suffix in [
        ("test_file", "test_file"),
        ("ours_file", "ours_file"),
        ("baseline_file", "baseline_file"),
        ("cache_file", "cache_file"),
    ]:
        value = resolve_path(getattr(args, f"{prefix}_{arg_suffix}"))
        if value is not None:
            overrides[field_name] = value
    return replace(config, **overrides)


def resolve_bertscore_model(model_type: str) -> str:
    if model_type == "roberta-large" and LOCAL_ROBERTA_LARGE.is_dir():
        return str(LOCAL_ROBERTA_LARGE)
    resolved = resolve_path(model_type)
    if resolved is not None and resolved.exists():
        return str(resolved)
    return model_type


def resolve_bertscore_num_layers(model_type: str, num_layers: int | None) -> int | None:
    if num_layers is not None:
        return num_layers
    if model_type == "roberta-large":
        return 24
    if Path(model_type).name == "roberta-large":
        return 24
    return None


def extract_tag(text: str, tag: str) -> str:
    match = re.search(
        rf"<{tag}>(.*?)</{tag}>", text or "", flags=re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_tokenize(text: str) -> list[str]:
    text = normalize_text(text).lower()
    if not text:
        return []
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\b\w+\b", text)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_test_set(path: Path) -> pd.DataFrame:
    rows = []
    for sample_id, item in enumerate(load_jsonl(path)):
        prompt = ""
        reference = ""
        for turn in item.get("conversations", []):
            if turn.get("from") == "user":
                prompt = turn.get("value", "")
            elif turn.get("from") == "assistant":
                raw_ref = turn.get("value", "")
                reference = (extract_tag(raw_ref, "response") or raw_ref).strip()
        rows.append(
            {
                "sample_id": sample_id,
                "prompt": (prompt or "").strip(),
                "reference": reference,
            }
        )
    return pd.DataFrame(rows)


def load_generation_file(path: Path, response_col: str) -> pd.DataFrame:
    rows = []
    for item in load_jsonl(path):
        raw = item.get("response") or extract_tag(item.get("raw_output", ""), "response")
        if not raw:
            raw = item.get("raw_output", "")
        rows.append(
            {
                "sample_id": int(item["sample_id"]),
                response_col: (raw or "").strip(),
            }
        )
    df = pd.DataFrame(rows)
    if df["sample_id"].duplicated().any():
        duplicated = df[df["sample_id"].duplicated()]["sample_id"].tolist()[:10]
        raise ValueError(f"Duplicated sample_id in {path}: {duplicated}")
    return df


def align_dataset(config: DatasetConfig, allow_count_mismatch: bool) -> pd.DataFrame:
    refs = load_test_set(config.test_file)
    ours = load_generation_file(config.ours_file, "ours_response")
    base = load_generation_file(config.baseline_file, "baseline_response")

    df = refs.merge(ours, on="sample_id", how="inner").merge(
        base, on="sample_id", how="inner"
    )
    df = df.sort_values("sample_id").reset_index(drop=True)

    if len(df) != config.expected_count and not allow_count_mismatch:
        raise ValueError(
            f"{config.name}: aligned {len(df)} samples, expected {config.expected_count}. "
            "Use --allow_count_mismatch only if this is intentional."
        )

    missing_ref = df["reference"].eq("").sum()
    if missing_ref:
        raise ValueError(f"{config.name}: {missing_ref} empty references found.")
    return df


def compute_meteor_columns(df: pd.DataFrame) -> pd.DataFrame:
    ours_scores = []
    baseline_scores = []
    for ref, ours, base in tqdm(
        zip(df["reference"], df["ours_response"], df["baseline_response"]),
        total=len(df),
        desc="METEOR",
    ):
        rt = safe_tokenize(ref) or [""]
        ot = safe_tokenize(ours) or [""]
        ft = safe_tokenize(base) or [""]
        ours_scores.append(meteor_score([rt], ot) * 100.0)
        baseline_scores.append(meteor_score([rt], ft) * 100.0)
    df["ours_meteor"] = ours_scores
    df["baseline_meteor"] = baseline_scores
    return df


def compute_bertscore_columns(
    df: pd.DataFrame,
    model_type: str,
    num_layers: int | None,
    batch_size: int,
    device: str,
    rescale_with_baseline: bool,
) -> pd.DataFrame:
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise ImportError(
            "bert_score is not installed in this Python environment. "
            "Install it in the active environment before running significance tests."
        ) from exc

    refs = df["reference"].tolist()
    ours = df["ours_response"].tolist()
    base = df["baseline_response"].tolist()

    print(f"Computing BERTScore for Ours with {model_type} on {device}...")
    _, _, ours_f1 = bert_score(
        ours,
        refs,
        lang="en",
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=rescale_with_baseline,
        idf=False,
        verbose=True,
    )
    print(f"Computing BERTScore for FIRES (Baseline) with {model_type} on {device}...")
    _, _, baseline_f1 = bert_score(
        base,
        refs,
        lang="en",
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=rescale_with_baseline,
        idf=False,
        verbose=True,
    )
    df["ours_bertscore_f1"] = ours_f1.detach().cpu().numpy() * 100.0
    df["baseline_bertscore_f1"] = baseline_f1.detach().cpu().numpy() * 100.0
    return df


def compute_diff_epitome_columns(
    df: pd.DataFrame, reward_model_path: str, batch_size: int, device: str
) -> pd.DataFrame:
    print(f"Loading DIFF-EPITOME model from {reward_model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            reward_model_path, num_labels=3
        )
        .to(device)
        .eval()
    )

    prompts = df["prompt"].tolist()
    refs = df["reference"].tolist()
    ours = df["ours_response"].tolist()
    base = df["baseline_response"].tolist()

    ref_logits = []
    ours_logits = []
    baseline_logits = []
    with torch.no_grad():
        for start in tqdm(range(0, len(df), batch_size), desc="DIFF-EPITOME"):
            end = start + batch_size
            p_b = prompts[start:end]
            ref_b = refs[start:end]
            ours_b = ours[start:end]
            baseline_b = base[start:end]

            texts = (
                [p[-512:] + " " + r for p, r in zip(p_b, ref_b)]
                + [p[-512:] + " " + h for p, h in zip(p_b, ours_b)]
                + [p[-512:] + " " + h for p, h in zip(p_b, baseline_b)]
            )
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)
            logits = model(**inputs).logits.detach().cpu()
            chunk = len(p_b)
            ref_logits.append(logits[:chunk])
            ours_logits.append(logits[chunk : 2 * chunk])
            baseline_logits.append(logits[2 * chunk :])

    ref_logits_t = torch.cat(ref_logits, dim=0)
    ours_logits_t = torch.cat(ours_logits, dim=0)
    baseline_logits_t = torch.cat(baseline_logits, dim=0)

    df["ours_diff_epitome"] = (
        F.mse_loss(ours_logits_t, ref_logits_t, reduction="none").mean(dim=1).numpy()
    )
    df["baseline_diff_epitome"] = (
        F.mse_loss(baseline_logits_t, ref_logits_t, reduction="none").mean(dim=1).numpy()
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def compute_metric_cache(
    config: DatasetConfig,
    args: argparse.Namespace,
    metrics: set[str],
) -> pd.DataFrame:
    required_columns = {"sample_id", "prompt", "reference", "ours_response", "baseline_response"}
    for metric in metrics:
        if metric == "meteor":
            required_columns.update({"ours_meteor", "baseline_meteor"})
        elif metric == "bertscore":
            required_columns.update({"ours_bertscore_f1", "baseline_bertscore_f1"})
        elif metric == "diff_epitome":
            required_columns.update({"ours_diff_epitome", "baseline_diff_epitome"})

    if config.cache_file.exists() and not args.force_cache:
        cached = pd.read_csv(config.cache_file)
        if required_columns.issubset(set(cached.columns)):
            print(f"{config.name}: using existing cache {config.cache_file}")
            return cached
        missing = sorted(required_columns.difference(set(cached.columns)))
        print(
            f"{config.name}: cache exists but is missing columns {missing}; recomputing."
        )

    print(f"\n=== Building metric cache for {config.name} ===")
    df = align_dataset(config, args.allow_count_mismatch)
    print(f"{config.name}: aligned {len(df)} full-test samples.")

    if "meteor" in metrics:
        df = compute_meteor_columns(df)
    if "bertscore" in metrics:
        df = compute_bertscore_columns(
            df,
            model_type=resolve_bertscore_model(args.bertscore_model),
            num_layers=resolve_bertscore_num_layers(
                args.bertscore_model, args.bertscore_num_layers
            ),
            batch_size=args.batch_size,
            device=resolve_device(args.device),
            rescale_with_baseline=args.bertscore_rescale,
        )
    if "diff_epitome" in metrics:
        df = compute_diff_epitome_columns(
            df,
            reward_model_path=args.reward_model,
            batch_size=args.batch_size,
            device=resolve_device(args.device),
        )

    config.cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.cache_file, index=False)
    print(f"{config.name}: wrote cache to {config.cache_file}")
    return df


def build_ngram_cache(responses: Iterable[str], n: int = 3) -> tuple[list[set], np.ndarray]:
    ngram_sets = []
    ngram_counts = []
    for response in responses:
        tokens = safe_tokenize(response)
        if len(tokens) < n:
            ngram_sets.append(set())
            ngram_counts.append(0)
            continue
        grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        ngram_sets.append(set(grams))
        ngram_counts.append(len(grams))
    return ngram_sets, np.asarray(ngram_counts, dtype=np.int64)


def distinct_from_cache(ngram_sets: list[set], ngram_counts: np.ndarray, idx: np.ndarray) -> float:
    total = int(ngram_counts[idx].sum())
    if total == 0:
        return 0.0
    unique = set()
    for i in idx:
        unique.update(ngram_sets[int(i)])
    return (len(unique) / total * 100.0) if total else 0.0


def sample_distinct_n(response: str, n: int = 3) -> float:
    tokens = safe_tokenize(response)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return (len(set(grams)) / len(grams) * 100.0) if grams else 0.0


def mean_sample_distinct_n(responses: Iterable[str], n: int = 3) -> float:
    scores = [sample_distinct_n(response, n=n) for response in responses]
    return float(np.mean(scores)) if scores else 0.0


def paired_bootstrap(
    df: pd.DataFrame,
    metric: str,
    n_boot: int,
    seed: int,
    distinct3_aggregation: str,
) -> dict:
    ours_col, baseline_col, higher_is_better, display_name = METRIC_COLUMNS[metric]
    rng = np.random.default_rng(seed)
    n = len(df)

    if metric == "distinct3":
        ours_values = df[ours_col].tolist()
        baseline_values = df[baseline_col].tolist()
        if distinct3_aggregation == "sample":
            ours_scores = np.asarray(
                [sample_distinct_n(response, 3) for response in ours_values],
                dtype=np.float64,
            )
            baseline_scores = np.asarray(
                [sample_distinct_n(response, 3) for response in baseline_values],
                dtype=np.float64,
            )
            ours_mean = float(np.mean(ours_scores))
            baseline_mean = float(np.mean(baseline_scores))

            idx = rng.integers(0, n, size=(n_boot, n))
            print("Bootstrap Distinct-3 (sample-average)...")
            ours_boot = ours_scores[idx].mean(axis=1)
            baseline_boot = baseline_scores[idx].mean(axis=1)
            improvements = ours_boot - baseline_boot
        elif distinct3_aggregation == "corpus":
            print("Precomputing Distinct-3 n-gram caches...")
            ours_ngram_sets, ours_ngram_counts = build_ngram_cache(ours_values, 3)
            baseline_ngram_sets, baseline_ngram_counts = build_ngram_cache(baseline_values, 3)
            full_idx = np.arange(n)
            ours_mean = distinct_from_cache(ours_ngram_sets, ours_ngram_counts, full_idx)
            baseline_mean = distinct_from_cache(baseline_ngram_sets, baseline_ngram_counts, full_idx)

            improvements = np.empty(n_boot, dtype=np.float64)
            for b in tqdm(range(n_boot), desc="Bootstrap Distinct-3"):
                idx = rng.integers(0, n, size=n)
                improvements[b] = distinct_from_cache(
                    ours_ngram_sets, ours_ngram_counts, idx
                ) - distinct_from_cache(baseline_ngram_sets, baseline_ngram_counts, idx)
        else:
            raise ValueError(
                f"Unknown distinct3 aggregation '{distinct3_aggregation}'. "
                "Use 'sample' or 'corpus'."
            )
    else:
        missing = [col for col in (ours_col, baseline_col) if col not in df.columns]
        if missing:
            raise ValueError(
                f"Cache is missing {missing}. Re-run with --mode cache --force_cache "
                f"and include metric '{metric}'."
            )

        ours_arr = df[ours_col].to_numpy(dtype=np.float64)
        baseline_arr = df[baseline_col].to_numpy(dtype=np.float64)
        ours_mean = float(np.mean(ours_arr))
        baseline_mean = float(np.mean(baseline_arr))

        idx = rng.integers(0, n, size=(n_boot, n))
        print(f"Bootstrap {display_name}...")
        ours_boot = ours_arr[idx].mean(axis=1)
        baseline_boot = baseline_arr[idx].mean(axis=1)
        if higher_is_better:
            improvements = ours_boot - baseline_boot
        else:
            improvements = baseline_boot - ours_boot

    observed_improvement = (
        (ours_mean - baseline_mean) if higher_is_better else (baseline_mean - ours_mean)
    )
    # One-sided paired bootstrap p-value for "FACES improves over FIRES (Baseline)".
    # Add-one smoothing avoids returning an exact zero with finite resamples.
    p_value = float((np.sum(improvements <= 0.0) + 1) / (n_boot + 1))
    ci_low, ci_high = np.percentile(improvements, [2.5, 97.5])

    return {
        "metric": display_name,
        "metric_key": metric,
        "n": n,
        "baseline_mean": baseline_mean,
        "ours_mean": ours_mean,
        "improvement": float(observed_improvement),
        "p_value": p_value,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def format_p(p_value: float) -> str:
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.4f}"


def stable_seed(base_seed: int, *parts: str) -> int:
    key = "::".join(parts).encode("utf-8")
    return int(base_seed + zlib.crc32(key) % 100000)


def print_results(results: list[dict], alpha: float) -> None:
    if not results:
        return

    headers = [
        "Dataset",
        "Metric",
        "FIRES (Baseline)",
        "Ours",
        "Improvement",
        "95% CI",
        "p",
        "Sig.",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r["dataset"],
                r["metric"],
                f"{r['baseline_mean']:.4f}",
                f"{r['ours_mean']:.4f}",
                f"{r['improvement']:.4f}",
                f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]",
                format_p(r["p_value"]),
                "yes" if r["p_value"] < alpha else "no",
            ]
        )

    widths = [
        max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))
    ]
    print("\nPaired bootstrap results")
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    print("Improvement is defined so that positive values favor Ours.")


def main() -> None:
    args = parse_args()
    metrics = set(args.metrics)
    if args.include_distinct3:
        metrics.add("distinct3")

    if args.n_boot <= 0:
        raise ValueError("--n_boot must be positive.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1.")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Mode: {args.mode}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Metrics: {', '.join(sorted(metrics))}")
    print(f"Device: {resolve_device(args.device)}")
    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"),
    )

    results = []
    for dataset_key in args.datasets:
        config = resolve_dataset_config(dataset_key, args)
        print(f"\n[{config.name}]")
        print(f"  test : {config.test_file}")
        print(f"  ours : {config.ours_file}")
        print(f"  baseline: {config.baseline_file}")
        print(f"  cache: {config.cache_file}")

        if args.mode in {"all", "cache"}:
            df = compute_metric_cache(config, args, metrics)
        else:
            if not config.cache_file.is_file():
                raise FileNotFoundError(
                    f"Missing cache for {config.name}: {config.cache_file}. "
                    "Run with --mode cache first."
                )
            df = pd.read_csv(config.cache_file)

        if args.mode in {"all", "bootstrap"}:
            if len(df) != config.expected_count and not args.allow_count_mismatch:
                raise ValueError(
                    f"{config.name}: cache has {len(df)} rows, expected "
                    f"{config.expected_count}."
                )
            for metric in sorted(metrics):
                res = paired_bootstrap(
                    df=df,
                    metric=metric,
                    n_boot=args.n_boot,
                    seed=stable_seed(args.seed, dataset_key, metric),
                    distinct3_aggregation=args.distinct3_aggregation,
                )
                res["dataset"] = config.name
                res["significant"] = bool(res["p_value"] < args.alpha)
                results.append(res)

    if results:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        print_results(results, args.alpha)
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
