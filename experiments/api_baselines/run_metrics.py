#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from api_baselines.data import extract_response, load_dataset
from api_baselines.io_utils import ensure_dir, read_json, read_jsonl, write_json
from api_baselines.metrics import (
    compute_basic_metrics,
    compute_diff_epitome,
    compute_fbert,
    configure_local_nltk_data,
)


def parse_list(value: str) -> list[str]:
    if value.lower() == "all":
        return ["all"]
    return [x.strip() for x in value.split(",") if x.strip()]


def clean_final_response(value: object) -> str:
    text = extract_response(str(value or ""))
    lowered = text.lower()
    if "<think" in lowered or "<answer" in lowered or "</answer>" in lowered:
        return ""
    return text


def dedupe_valid_rows(rows: list[dict[str, object]], sample_by_id: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, int]]:
    latest_by_id: dict[str, dict[str, object]] = {}
    stats = {
        "raw_rows": len(rows),
        "invalid_rows": 0,
        "dirty_final_rows": 0,
        "duplicate_ok_rows": 0,
    }
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        final_response = clean_final_response(row.get("final_response", ""))
        if row.get("status") != "ok" or sample_id not in sample_by_id or not final_response:
            stats["invalid_rows"] += 1
            if row.get("status") == "ok" and sample_id in sample_by_id and str(row.get("final_response", "")).strip():
                stats["dirty_final_rows"] += 1
            continue
        row = dict(row)
        row["final_response"] = final_response
        if sample_id in latest_by_id:
            stats["duplicate_ok_rows"] += 1
        latest_by_id[sample_id] = row
    return sorted(
        latest_by_id.values(),
        key=lambda row: int(row.get("source_index", 0)),
    ), stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute automatic metrics for API baseline outputs.")
    parser.add_argument("--datasets", default="mesc", help="Comma-separated dataset names or all.")
    parser.add_argument("--models", default="all", help="Comma-separated model names or all available output files.")
    parser.add_argument("--without-fbert", action="store_true", help="Skip FBERT / BERTScore-F1.")
    parser.add_argument("--without-diff-epitome", action="store_true", help="Skip DIFF-E.")
    parser.add_argument("--fbert-batch-size", type=int, default=8)
    parser.add_argument("--fbert-model-type", default="roberta-large")
    parser.add_argument("--fbert-num-layers", type=int, default=24)
    parser.add_argument("--diff-batch-size", type=int, default=16)
    parser.add_argument(
        "--diff-epitome-model",
        default="checkpoints/reward_model/roberta_epitome_multitask/final",
        help="DIFF-EPITOME reward model path, relative to the repository root unless absolute.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N dataset samples.")
    parser.add_argument(
        "--summary-name",
        default="automatic_metrics_summary.csv",
        help="CSV filename under reports/ for the summary output.",
    )
    parser.add_argument(
        "--append-summary",
        action="store_true",
        help="Append new rows to an existing summary CSV instead of overwriting it.",
    )
    args = parser.parse_args()

    configure_local_nltk_data(REPO_ROOT)
    datasets_config = read_json(PROJECT_DIR / "configs" / "datasets.json")
    dataset_names = parse_list(args.datasets)
    if dataset_names == ["all"]:
        dataset_names = list(datasets_config)

    summaries: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        if dataset_name not in datasets_config:
            raise SystemExit(f"Unknown dataset: {dataset_name}. Available: {', '.join(datasets_config)}")

        dataset_path = REPO_ROOT / datasets_config[dataset_name]["path"]
        samples = load_dataset(dataset_name, dataset_path, limit=args.limit)
        sample_by_id = {sample.sample_id: sample for sample in samples}

        raw_dir = PROJECT_DIR / "outputs" / "raw" / dataset_name
        if not raw_dir.exists():
            print(f"No raw output directory found: {raw_dir}")
            continue

        model_names = parse_list(args.models)
        if model_names == ["all"]:
            model_names = sorted(p.stem for p in raw_dir.glob("*.jsonl"))

        for model_name in model_names:
            output_path = raw_dir / f"{model_name}.jsonl"
            rows = read_jsonl(output_path)
            ok_rows, row_stats = dedupe_valid_rows(rows, sample_by_id)
            if not ok_rows:
                print(f"Skipping {dataset_name}/{model_name}: no valid rows.")
                continue

            prompts = [sample_by_id[row["sample_id"]].context for row in ok_rows]
            references = [sample_by_id[row["sample_id"]].reference_response for row in ok_rows]
            hypotheses = [str(row.get("final_response", "")).strip() for row in ok_rows]
            metrics = compute_basic_metrics(references, hypotheses)
            if not args.without_fbert:
                metrics["FBERT"] = compute_fbert(
                    references,
                    hypotheses,
                    repo_root=REPO_ROOT,
                    batch_size=args.fbert_batch_size,
                    model_type=args.fbert_model_type,
                    num_layers=args.fbert_num_layers,
                )
            if not args.without_diff_epitome:
                metrics["DIFF-E."] = compute_diff_epitome(
                    prompts,
                    references,
                    hypotheses,
                    repo_root=REPO_ROOT,
                    batch_size=args.diff_batch_size,
                    reward_model_path=args.diff_epitome_model,
                )

            summary = {
                "dataset": dataset_name,
                "model_name": model_name,
                "valid_samples": len(ok_rows),
                "raw_rows": len(rows),
                "dataset_samples": len(samples),
                "duplicate_ok_rows": row_stats["duplicate_ok_rows"],
                "invalid_rows": row_stats["invalid_rows"],
                "dirty_final_rows": row_stats["dirty_final_rows"],
            }
            for key in ["B-2", "R-L", "M.", "FBERT", "Distinct-3", "DIFF-E."]:
                if key in metrics:
                    summary[key] = metrics[key]
            for key, value in metrics.items():
                if key not in summary:
                    summary[key] = value
            summaries.append(summary)
            out_json = PROJECT_DIR / "outputs" / "metrics" / dataset_name / f"{model_name}.json"
            write_json(out_json, summary)
            print(f"Saved metrics: {out_json}")

    if summaries:
        report_dir = ensure_dir(PROJECT_DIR / "reports")
        csv_path = report_dir / args.summary_name
        if args.append_summary and csv_path.exists():
            existing_rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
            fieldnames = list(existing_rows[0].keys()) if existing_rows else []
            for row in summaries:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            combined_rows = existing_rows + summaries
            ordered_rows = []
            for row in combined_rows:
                ordered = OrderedDict()
                for key in fieldnames:
                    ordered[key] = row.get(key, "")
                ordered_rows.append(ordered)
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(ordered_rows)
        else:
            fieldnames = list(summaries[0].keys())
            for row in summaries[1:]:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summaries)
        print(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
