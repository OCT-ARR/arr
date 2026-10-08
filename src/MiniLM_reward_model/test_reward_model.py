#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_epitome_multitask import load_and_merge_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the optional DIFF-EPITOME support-skill reward model."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "reward_model"
            / "roberta_epitome_multitask"
            / "final"
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "EPITOME-Reddit",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(
        device
    )
    model.eval()

    df = load_and_merge_data(args.data_dir)
    ds = Dataset.from_dict(
        {
            "text_a": df["seeker_post"].astype(str).tolist(),
            "text_b": df["response_post"].astype(str).tolist(),
            "labels": df.apply(
                lambda row: [
                    float(row["label_ER"]),
                    float(row["label_IP"]),
                    float(row["label_EX"]),
                ],
                axis=1,
            ).tolist(),
        }
    )
    test_ds = ds.train_test_split(test_size=0.1, seed=args.seed)["test"]

    preds_list, labels_list = [], []
    with torch.no_grad():
        for i in range(0, len(test_ds), args.batch_size):
            batch = test_ds[i : i + args.batch_size]
            inputs = tokenizer(
                batch["text_a"],
                batch["text_b"],
                truncation=True,
                max_length=args.max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            preds_list.extend(outputs.logits.cpu().numpy())
            labels_list.extend(batch["labels"])

    preds = np.array(preds_list)
    labels = np.array(labels_list)
    tasks = ["ER", "IP", "EX"]

    print(f"Evaluated samples: {len(test_ds)}")
    print(f"{'Task':<8} | {'PCC':>10} | {'MAE':>10}")
    print("-" * 36)
    for i, task_name in enumerate(tasks):
        pcc, _ = pearsonr(preds[:, i], labels[:, i])
        mae = mean_absolute_error(labels[:, i], preds[:, i])
        print(f"{task_name:<8} | {pcc:>10.4f} | {mae:>10.4f}")


if __name__ == "__main__":
    main()
