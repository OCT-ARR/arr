#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_and_merge_data(data_dir: Path) -> pd.DataFrame:
    files = {
        "ER": "emotional-reactions-reddit.csv",
        "IP": "interpretations-reddit.csv",
        "EX": "explorations-reddit.csv",
    }

    dfs: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing EPITOME annotation file: {path}")

        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        if key == "ER":
            cols = ["sp_id", "rp_id", "seeker_post", "response_post", "level"]
            df = df[cols].rename(columns={"level": "label_ER"})
        else:
            cols = ["sp_id", "rp_id", "level"]
            df = df[cols].rename(columns={"level": f"label_{key}"})
        dfs[key] = df.dropna()

    merged = dfs["ER"]
    merged = pd.merge(merged, dfs["IP"], on=["sp_id", "rp_id"], how="inner")
    merged = pd.merge(merged, dfs["EX"], on=["sp_id", "rp_id"], how="inner")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the optional DIFF-EPITOME support-skill reward model."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "EPITOME-Reddit",
        help="Directory containing the three EPITOME CSV annotation files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "reward_model" / "roberta_epitome_multitask",
        help="Directory where reward-model checkpoints will be written.",
    )
    parser.add_argument("--model_checkpoint", default="roberta-base")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training if supported by the local GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge_data(args.data_dir)
    print(f"Loaded merged EPITOME samples: {len(df)}")

    def combine_labels(row: pd.Series) -> list[float]:
        return [float(row["label_ER"]), float(row["label_IP"]), float(row["label_EX"])]

    ds = Dataset.from_dict(
        {
            "text_a": df["seeker_post"].astype(str).tolist(),
            "text_b": df["response_post"].astype(str).tolist(),
            "labels": df.apply(combine_labels, axis=1).tolist(),
        }
    )
    ds_split = ds.train_test_split(test_size=0.1, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    def preprocess_function(examples):
        return tokenizer(
            examples["text_a"],
            examples["text_b"],
            truncation=True,
            max_length=args.max_len,
            padding=False,
        )

    tokenized_ds = ds_split.map(preprocess_function, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint,
        num_labels=3,
        problem_type="regression",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=args.fp16,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    final_path = args.output_dir / "final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved DIFF-EPITOME reward model to: {final_path}")
    print("Output score order: [ER_score, IP_score, EX_score]")


if __name__ == "__main__":
    main()
