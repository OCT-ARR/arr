#!/usr/bin/env python3
"""Build the MELD-ESC300 generalization set from MELD test data.

This script:
1. Reads MELD.Raw/test_sent_emo.csv.
2. Filters emotional-support-style samples from the MELD test split.
3. Uses the previous utterance video as visual context.
4. Extracts 5 frames per selected sample.
5. Writes data/MELD-ESC300/test.jsonl in the project format.

Raw MELD videos are external assets and are not included in this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "raw" / "MELD" / "MELD.Raw" / "test_sent_emo.csv"
DEFAULT_VIDEO_DIR = (
    REPO_ROOT
    / "raw"
    / "MELD"
    / "MELD.Raw"
    / "output_repeated_splits_test"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "MELD-ESC300"

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

POSITIVE_OR_NEUTRAL_TARGET_EMOTIONS = {"neutral", "joy", "surprise"}
NEGATIVE_CONTEXT_EMOTIONS = {"anger", "sadness", "fear", "disgust"}


@dataclass(frozen=True)
class UtteranceRow:
    sr_no: str
    utterance: str
    speaker: str
    emotion: str
    sentiment: str
    dialogue_id: int
    utterance_id: int


@dataclass(frozen=True)
class Candidate:
    dialogue_id: int
    target_idx: int
    visual_idx: int
    strict: bool
    video_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="MELD test_sent_emo.csv path.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="MELD test video directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--image-root-name",
        default="frames",
        help="Frame subdirectory name under --output-dir.",
    )
    parser.add_argument("--num-samples", type=int, default=300, help="Number of samples to build.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames extracted per sample.")
    parser.add_argument("--max-context-turns", type=int, default=6, help="Maximum turns before target response.")
    parser.add_argument("--min-response-words", type=int, default=4, help="Minimum target response length.")
    parser.add_argument("--max-response-words", type=int, default=40, help="Maximum target response length.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract frames even if the target frame files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report candidate counts; do not extract frames or write output.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = text.replace("\u0092", "'")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00ac\u00eds", "'s").replace("\u00ac\u00edt", "'t")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def read_meld_csv(path: Path) -> Dict[int, List[UtteranceRow]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    grouped: Dict[int, List[UtteranceRow]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dialogue_id = int(str(row["Dialogue_ID"]).strip())
                utterance_id = int(str(row["Utterance_ID"]).strip())
            except Exception:
                continue
            item = UtteranceRow(
                sr_no=str(row.get("Sr No.", "")).strip(),
                utterance=clean_text(str(row.get("Utterance", ""))),
                speaker=clean_text(str(row.get("Speaker", ""))),
                emotion=clean_text(str(row.get("Emotion", ""))).lower(),
                sentiment=clean_text(str(row.get("Sentiment", ""))).lower(),
                dialogue_id=dialogue_id,
                utterance_id=utterance_id,
            )
            if item.utterance:
                grouped.setdefault(dialogue_id, []).append(item)

    for rows in grouped.values():
        rows.sort(key=lambda x: x.utterance_id)
    return grouped


def locate_video(video_dir: Path, dialogue_id: int, utterance_id: int) -> Optional[Path]:
    names = [
        f"dia{dialogue_id}_utt{utterance_id}.mp4",
        f"final_videos_testdia{dialogue_id}_utt{utterance_id}.mp4",
    ]
    for name in names:
        path = video_dir / name
        if path.is_file() and not path.name.startswith("._"):
            return path
    return None


def has_negative_affect(row: UtteranceRow) -> bool:
    return row.sentiment == "negative" or row.emotion in NEGATIVE_CONTEXT_EMOTIONS


def target_is_support_like(row: UtteranceRow, min_words: int, max_words: int) -> bool:
    n_words = word_count(row.utterance)
    return (
        min_words <= n_words <= max_words
        and row.sentiment != "negative"
        and row.emotion in POSITIVE_OR_NEUTRAL_TARGET_EMOTIONS
    )


def collect_candidates(
    grouped: Dict[int, List[UtteranceRow]],
    video_dir: Path,
    min_words: int,
    max_words: int,
) -> Tuple[List[Candidate], List[Candidate]]:
    strict: List[Candidate] = []
    relaxed: List[Candidate] = []
    seen = set()

    for dialogue_id, rows in grouped.items():
        for idx in range(2, len(rows)):
            target = rows[idx]
            prev = rows[idx - 1]
            if not target_is_support_like(target, min_words, max_words):
                continue
            if target.speaker == prev.speaker:
                continue

            video_path = locate_video(video_dir, dialogue_id, prev.utterance_id)
            if video_path is None:
                continue

            key = (dialogue_id, target.utterance_id)
            if has_negative_affect(prev):
                strict.append(
                    Candidate(
                        dialogue_id=dialogue_id,
                        target_idx=idx,
                        visual_idx=idx - 1,
                        strict=True,
                        video_path=video_path,
                    )
                )
                seen.add(key)
                continue

            recent_context = rows[max(0, idx - 3) : idx]
            if key not in seen and any(has_negative_affect(row) for row in recent_context):
                relaxed.append(
                    Candidate(
                        dialogue_id=dialogue_id,
                        target_idx=idx,
                        visual_idx=idx - 1,
                        strict=False,
                        video_path=video_path,
                    )
                )
                seen.add(key)

    return strict, relaxed


def choose_candidates(strict: List[Candidate], relaxed: List[Candidate], n: int, seed: int) -> List[Candidate]:
    rng = random.Random(seed)
    strict = list(strict)
    relaxed = list(relaxed)
    rng.shuffle(strict)
    rng.shuffle(relaxed)
    selected = strict[:n]
    if len(selected) < n:
        selected.extend(relaxed[: n - len(selected)])
    selected.sort(key=lambda x: (x.dialogue_id, x.target_idx))
    return selected


def run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def probe_duration(video_path: Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    proc = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    if proc.returncode != 0:
        return None
    try:
        duration = float(proc.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def extract_single_frame(video_path: Path, timestamp: float, out_path: Path) -> Tuple[bool, str]:
    """Extract one frame, trying both fast and accurate seeking.

    Some MELD clips are shorter than 0.3s, so later timestamps may not decode
    reliably with fast seeking. Keeping this helper defensive prevents a single
    short clip from stopping the whole subset construction.
    """
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None

    if out_path.exists():
        out_path.unlink()

    attempts = [
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=448:-2",
            str(out_path),
        ],
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-ss",
            f"{timestamp:.3f}",
            "-frames:v",
            "1",
            "-vf",
            "scale=448:-2",
            str(out_path),
        ],
    ]

    errors: List[str] = []
    for cmd in attempts:
        proc = run_command(cmd)
        if proc.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
            return True, ""
        errors.append(proc.stderr.strip())
        if out_path.exists():
            out_path.unlink()
    return False, " | ".join(err for err in errors if err)


def extract_frames(video_path: Path, frame_dir: Path, num_frames: int, force: bool) -> List[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available on PATH.")

    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = [frame_dir / f"frame_{idx:04d}.jpg" for idx in range(1, num_frames + 1)]
    if not force and all(path.is_file() and path.stat().st_size > 0 for path in frame_paths):
        return frame_paths

    for path in frame_paths:
        if force and path.exists():
            path.unlink()

    duration = probe_duration(video_path)
    if duration is not None:
        timestamps = [(idx + 1) * duration / (num_frames + 1) for idx in range(num_frames)]
    else:
        timestamps = [float(idx) for idx in range(num_frames)]

    last_ok: Optional[Path] = None
    for timestamp, out_path in zip(timestamps, frame_paths):
        ok, error = extract_single_frame(video_path, timestamp, out_path)
        if ok:
            last_ok = out_path
            continue

        ok, fallback_error = extract_single_frame(video_path, 0.0, out_path)
        if ok:
            last_ok = out_path
            continue

        if last_ok is not None:
            shutil.copyfile(last_ok, out_path)
            continue

        message = error or fallback_error or "ffmpeg produced no frame"
        raise RuntimeError(f"Failed to extract frame from {video_path}: {message}")

    return frame_paths


def build_prompt(rows: List[UtteranceRow], target_idx: int, max_context_turns: int, num_images: int) -> str:
    start = max(0, target_idx - max_context_turns)
    context_rows = rows[start:target_idx]
    lines = [f"{row.speaker}: {row.utterance}" for row in context_rows]
    image_prefix = "<image>" * num_images
    return f"{image_prefix}" + "\n".join(lines) + "\nTherapist:"


def build_json_record(
    rows: List[UtteranceRow],
    candidate: Candidate,
    frame_paths: Sequence[Path],
    max_context_turns: int,
) -> Dict[str, object]:
    target = rows[candidate.target_idx]
    prompt = build_prompt(rows, candidate.target_idx, max_context_turns, len(frame_paths))
    rel_frames = [path.relative_to(REPO_ROOT).as_posix() for path in frame_paths]
    return {
        "system": SYSTEM_PROMPT,
        "conversations": [
            {
                "from": "user",
                "value": prompt,
                "images": rel_frames,
            },
            {
                "from": "assistant",
                "value": f"<response>{target.utterance}</response>",
            },
        ],
    }


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metadata(path: Path, grouped: Dict[int, List[UtteranceRow]], selected: Sequence[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "dialogue_id",
        "target_utterance_id",
        "previous_utterance_id",
        "target_speaker",
        "previous_speaker",
        "target_emotion",
        "target_sentiment",
        "previous_emotion",
        "previous_sentiment",
        "strict_filter",
        "video_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id, candidate in enumerate(selected):
            rows = grouped[candidate.dialogue_id]
            target = rows[candidate.target_idx]
            prev = rows[candidate.visual_idx]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "dialogue_id": candidate.dialogue_id,
                    "target_utterance_id": target.utterance_id,
                    "previous_utterance_id": prev.utterance_id,
                    "target_speaker": target.speaker,
                    "previous_speaker": prev.speaker,
                    "target_emotion": target.emotion,
                    "target_sentiment": target.sentiment,
                    "previous_emotion": prev.emotion,
                    "previous_sentiment": prev.sentiment,
                    "strict_filter": int(candidate.strict),
                    "video_path": str(candidate.video_path),
                }
            )


def main() -> None:
    args = parse_args()
    grouped = read_meld_csv(args.csv)
    strict, relaxed = collect_candidates(
        grouped=grouped,
        video_dir=args.video_dir,
        min_words=args.min_response_words,
        max_words=args.max_response_words,
    )
    selected = choose_candidates(strict, relaxed, args.num_samples, args.seed)

    print(f"Loaded dialogues: {len(grouped)}")
    print(f"Strict candidates: {len(strict)}")
    print(f"Relaxed candidates: {len(relaxed)}")
    print(f"Selected samples: {len(selected)} / requested {args.num_samples}")

    if len(selected) < args.num_samples:
        print("WARNING: fewer samples selected than requested. Consider relaxing filters.")

    if args.dry_run:
        return

    frames_root = args.output_dir / args.image_root_name
    jsonl_path = args.output_dir / "test.jsonl"
    metadata_path = args.output_dir / "metadata.csv"

    records: List[Dict[str, object]] = []
    for sample_id, candidate in enumerate(selected):
        rows = grouped[candidate.dialogue_id]
        prev = rows[candidate.visual_idx]
        frame_dir = frames_root / f"dia{candidate.dialogue_id}_utt{prev.utterance_id}"
        frame_paths = extract_frames(candidate.video_path, frame_dir, args.num_frames, args.force)
        record = build_json_record(
            rows=rows,
            candidate=candidate,
            frame_paths=frame_paths,
            max_context_turns=args.max_context_turns,
        )
        records.append(record)
        if (sample_id + 1) % 25 == 0 or sample_id + 1 == len(selected):
            print(f"Built {sample_id + 1}/{len(selected)} samples")

    write_jsonl(jsonl_path, records)
    write_metadata(metadata_path, grouped, selected)

    print(f"JSONL written to: {jsonl_path}")
    print(f"Metadata written to: {metadata_path}")
    print(f"Frames written to: {frames_root}")


if __name__ == "__main__":
    main()
