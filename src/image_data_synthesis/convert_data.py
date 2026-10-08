#!/usr/bin/env python3
"""Convert intermediate multimodal synthesis records into JSONL splits.

The expected input is a JSON list whose items contain:

- ``images``: a list of image paths;
- ``messages``: system, user, and assistant messages in order.

This utility is optional. The anonymous release already includes
``data/MESC-FT/{train,val,test}.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "raw" / "MESC-FT" / "synthesis_data.json"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "MESC-FT" / "converted.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Intermediate JSON file produced by local data synthesis.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSONL file in the repository conversation format.",
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root used to convert absolute image paths into repository-relative paths.",
    )
    parser.add_argument(
        "--keep_absolute_images",
        action="store_true",
        help="Keep absolute image paths instead of converting them to relative paths.",
    )
    return parser.parse_args()


def normalize_image_path(path_text: str, image_root: Path, keep_absolute: bool) -> str:
    path_text = str(path_text).strip()
    if not path_text:
        return path_text
    path = Path(path_text)
    if keep_absolute or not path.is_absolute():
        return path_text
    try:
        return path.resolve().relative_to(image_root.resolve()).as_posix()
    except ValueError:
        raise ValueError(
            f"Absolute image path is outside --image_root and would not be anonymous: {path}"
        )


def convert_records(
    raw_data: Iterable[Dict[str, Any]],
    image_root: Path,
    keep_absolute_images: bool,
) -> tuple[List[Dict[str, Any]], int]:
    converted: List[Dict[str, Any]] = []
    skipped = 0

    for item in raw_data:
        images = item.get("images", [])
        messages = item.get("messages", [])
        if not isinstance(images, list) or not isinstance(messages, list) or len(messages) < 3:
            skipped += 1
            continue

        try:
            system_text = str(messages[0]["content"])
            raw_user_text = str(messages[1]["content"])
            assistant_text = str(messages[2]["content"])
            normalized_images = [
                normalize_image_path(str(image), image_root, keep_absolute_images)
                for image in images
            ]
            user_text = raw_user_text.replace("<image>", "").strip()
            if normalized_images:
                user_text = f"{'<image>' * len(normalized_images)}{user_text}"
        except Exception:
            skipped += 1
            continue

        converted.append(
            {
                "system": system_text,
                "conversations": [
                    {
                        "from": "user",
                        "value": user_text,
                        "images": normalized_images,
                    },
                    {"from": "assistant", "value": assistant_text},
                ],
            }
        )

    return converted, skipped


def main() -> None:
    args = parse_args()
    input_file = args.input_file.expanduser()
    output_file = args.output_file.expanduser()

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, list):
        raise ValueError("Input file must contain a JSON list.")

    converted, skipped = convert_records(
        raw_data,
        image_root=args.image_root,
        keep_absolute_images=args.keep_absolute_images,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for entry in converted:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"input records: {len(raw_data)}")
    print(f"valid records: {len(converted)}")
    print(f"skipped records: {skipped}")
    print(f"output file: {output_file}")


if __name__ == "__main__":
    main()
