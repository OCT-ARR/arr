#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

from api_baselines.data import extract_response
from api_baselines.io_utils import append_jsonl, read_jsonl


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def repair_file(path: Path) -> tuple[int, int]:
    rows = read_jsonl(path)
    repaired = 0
    skipped = 0
    for row in rows:
        if row.get("status") != "ok":
            skipped += 1
            continue
        old_final = str(row.get("final_response", "")).strip()
        new_final = extract_response(str(row.get("raw_output") or old_final))
        if not new_final or new_final == old_final:
            skipped += 1
            continue
        new_row = dict(row)
        new_row["final_response"] = new_final
        new_row["repair_of_created_at_utc"] = row.get("created_at_utc")
        new_row["repaired_at_utc"] = datetime.now(timezone.utc).isoformat()
        append_jsonl(path, new_row)
        repaired += 1
    return repaired, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Append repaired final_response rows without deleting existing outputs.")
    parser.add_argument("--datasets", default="mesc,meld_esc300")
    parser.add_argument("--models", default="glm_4_1v_thinking_flashx")
    args = parser.parse_args()

    for dataset in parse_list(args.datasets):
        for model in parse_list(args.models):
            path = PROJECT_DIR / "outputs" / "raw" / dataset / f"{model}.jsonl"
            if not path.exists():
                print(f"Missing: {path}")
                continue
            repaired, skipped = repair_file(path)
            print(f"{dataset}/{model}: appended repaired rows={repaired}, skipped={skipped}")


if __name__ == "__main__":
    main()
