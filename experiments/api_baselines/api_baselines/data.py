from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl


RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_OPEN_RE = re.compile(r"<answer>(.*)$", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
IMAGE_TAG_RE = re.compile(r"<image>")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset: str
    context: str
    images: list[str]
    reference_response: str
    raw_reference: str
    source_index: int


def extract_response(text: str) -> str:
    text = (text or "").strip()
    match = ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip()
    match = ANSWER_OPEN_RE.search(text or "")
    if match:
        return TAG_RE.sub("", match.group(1)).strip()
    match = RESPONSE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    if re.search(r"<think\b", text, flags=re.IGNORECASE) and not re.search(
        r"</think>", text, flags=re.IGNORECASE
    ):
        return ""
    text = THINK_RE.sub("", text).strip()
    return text.strip()


def clean_context(text: str) -> str:
    text = IMAGE_TAG_RE.sub("", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def derive_sample_id(item: dict[str, Any], idx: int) -> str:
    conversations = item.get("conversations", [])
    images: list[str] = []
    prompt = ""
    if conversations:
        first = conversations[0]
        prompt = first.get("value", "")
        images = first.get("images", []) or []

    for image_path in images:
        parts = Path(image_path).parts
        for part in reversed(parts):
            if re.match(r"dia\d+.*utt\d+", part):
                return part

    digest = hashlib.sha1(prompt.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"sample_{idx:05d}_{digest}"


def load_dataset(dataset_name: str, path: str | Path, limit: int | None = None) -> list[Sample]:
    samples: list[Sample] = []
    for idx, item in enumerate(read_jsonl(path)):
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue

        user_turn = next((t for t in conversations if t.get("from") == "user"), None)
        assistant_turn = next((t for t in conversations if t.get("from") == "assistant"), None)
        if not user_turn or not assistant_turn:
            continue

        sample_id = derive_sample_id(item, idx)
        raw_reference = assistant_turn.get("value", "")
        samples.append(
            Sample(
                sample_id=sample_id,
                dataset=dataset_name,
                context=clean_context(user_turn.get("value", "")),
                images=list(user_turn.get("images", []) or []),
                reference_response=extract_response(raw_reference),
                raw_reference=raw_reference,
                source_index=idx,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples
