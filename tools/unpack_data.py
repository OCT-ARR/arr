#!/usr/bin/env python3
"""Decompress release data archives into the paths expected by the scripts."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


DATA_ARCHIVES = (
    "data/MESC-FT/train.jsonl.gz",
    "data/MESC-FT/val.jsonl.gz",
    "data/MESC-FT/test.jsonl.gz",
    "data/MELD-ESC300/test.jsonl.gz",
    "data/StickerConv/test.jsonl.gz",
    "data/AvaMERG/test.jsonl.gz",
    "data/EPITOME-Reddit/emotional-reactions-reddit.csv.gz",
    "data/EPITOME-Reddit/interpretations-reddit.csv.gz",
    "data/EPITOME-Reddit/explorations-reddit.csv.gz",
)


def unpack_one(path: Path, overwrite: bool) -> None:
    if path.suffix != ".gz":
        raise ValueError(f"Expected a .gz archive: {path}")
    target = path.with_suffix("")
    if target.exists() and not overwrite:
        print(f"[skip] {target} already exists")
        return
    if not path.exists():
        raise FileNotFoundError(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"[ok] {path} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack compressed release data files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing decompressed files.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    for rel_path in DATA_ARCHIVES:
        unpack_one(root / rel_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
