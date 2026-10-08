#!/usr/bin/env python3
"""Reconstruct the 300-example StickerConv paper evaluation split.

The loader accepts either the official LLM-release Parquet test file or its
legacy line-delimited JSON export. Sticker descriptions and emotion cues are
rendered as text because the original sticker pixels require separate access.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "raw" / "StickerConv" / "test.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "StickerConv"

# Same system prompt as build_emotyda_esc.py so the eval protocol matches.
SYSTEM_PROMPT = (
    "You are a professional therapist experienced in emotional support conversations. "
    "Based on the multimodal conversation history, you, as a therapist, first analyze "
    "the thought process and then provide a response to the client. The thought process "
    "involves four aspects: visual scene, client's emotion, client's situation, and "
    "response strategy; but whether each aspect is helpful for the conversation depends "
    "on the context. The thought process and response are enclosed within <think> "
    "</think> and <response> </response> tags, respectively, i.e., <think> thought "
    "process here </think><response> response here </response>"
)

NEGATIVE_ANNOS = {"Sadness", "Anger", "Fear", "Disgust"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-turns", type=int, default=2,
                   help="Need at least this many turns so there is context before the target.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sticker_or_none(value: Any) -> Optional[dict]:
    return value if isinstance(value, dict) and value else None


def _normalize_record(record: dict) -> dict:
    """Normalize current Parquet messages to the legacy paired-turn schema."""
    if isinstance(record.get("conversation"), list):
        return record

    messages = record.get("conversations")
    if messages is None:
        return {**record, "conversation": []}
    if hasattr(messages, "tolist"):
        messages = messages.tolist()

    turns = []
    pending_user: Optional[dict] = None
    for message in messages or []:
        role = clean(message.get("role", "")).lower()
        if role in {"user", "human"}:
            pending_user = message
        elif role in {"assistant", "system", "gpt"} and pending_user is not None:
            turns.append(
                {
                    "turn": len(turns) + 1,
                    "user_message": clean(pending_user.get("content", "")),
                    "system_message": clean(message.get("content", "")),
                    "user_sticker": _sticker_or_none(pending_user.get("image")),
                    "system_sticker": _sticker_or_none(message.get("image")),
                }
            )
            pending_user = None
    return {**record, "conversation": turns}


def load_records(src: Path) -> List[dict]:
    if not src.is_file():
        raise FileNotFoundError(f"StickerConv source not found: {src}")
    if src.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading StickerConv Parquet requires pandas and pyarrow") from exc
        raw_records = pd.read_parquet(src).to_dict(orient="records")
    else:
        raw_records = []
        with src.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))
    return [_normalize_record(record) for record in raw_records]


def sticker_tag(sticker: Optional[dict]) -> str:
    """Render a user sticker as an inline textual visual cue."""
    if not sticker:
        return ""
    desc = clean(sticker.get("description", ""))
    emo = clean(sticker.get("emotion", ""))
    parts = []
    if desc:
        parts.append(desc)
    if emo:
        parts.append(emo)
    if not parts:
        return ""
    return " [The user also sent a sticker. " + " ".join(parts) + "]"


def build_prompt(conversation: List[dict], target_idx: int) -> str:
    """Context = all turns before target_idx (both user & system), plus the
    target turn's user_message (+ its sticker cue). Ends with 'Therapist:'.
    """
    lines: List[str] = []
    for t in conversation[:target_idx]:
        um = clean(t.get("user_message", ""))
        sm = clean(t.get("system_message", ""))
        if um:
            lines.append(f"Client: {um}{sticker_tag(t.get('user_sticker'))}")
        if sm:
            lines.append(f"Therapist: {sm}")
    target = conversation[target_idx]
    um = clean(target.get("user_message", ""))
    lines.append(f"Client: {um}{sticker_tag(target.get('user_sticker'))}")
    lines.append("Therapist:")
    return "\n".join(lines)


def build_record(conversation: List[dict], target_idx: int) -> Optional[dict]:
    target = conversation[target_idx]
    gold = clean(target.get("system_message", ""))
    if not gold or not clean(target.get("user_message", "")):
        return None
    prompt = build_prompt(conversation, target_idx)
    return {
        "system": SYSTEM_PROMPT,
        "conversations": [
            {"from": "user", "value": prompt, "images": []},
            {"from": "assistant", "value": f"<response>{gold}</response>"},
        ],
    }


def has_negative(conversation: List[dict]) -> bool:
    for t in conversation:
        us = t.get("user_sticker")
        if us and us.get("origin_anno") in NEGATIVE_ANNOS:
            return True
    return False


def main() -> None:
    args = parse_args()
    recs = load_records(args.src)

    # A candidate = (record, target_idx). Target = last turn (max context).
    neg, other = [], []
    for r in recs:
        conv = r.get("conversation", [])
        if len(conv) < args.min_turns:
            continue
        target_idx = len(conv) - 1
        rec = build_record(conv, target_idx)
        if rec is None:
            continue
        (neg if has_negative(conv) else other).append(rec)

    rng = random.Random(args.seed)
    rng.shuffle(neg)
    rng.shuffle(other)
    selected = neg[: args.num_samples]
    if len(selected) < args.num_samples:
        selected.extend(other[: args.num_samples - len(selected)])

    print(f"Loaded conversations: {len(recs)}")
    print(f"Negative-affect candidates: {len(neg)}")
    print(f"Other candidates: {len(other)}")
    print(f"Selected: {len(selected)} / requested {args.num_samples}")
    if len(selected) < args.num_samples:
        print("WARNING: fewer samples than requested.")

    if args.dry_run:
        return

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"JSONL written to: {out_path}  ({len(selected)} lines)")


if __name__ == "__main__":
    main()
