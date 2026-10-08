#!/usr/bin/env python3
"""Reconstruct the 300-example AvaMERG paper evaluation split.

The default protocol uses seed 42, five uniformly sampled speaker-video frames,
and up to six dialogue-context turns.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "raw" / "AvaMERG"
DEFAULT_SRC = RAW_DIR / "test.json"
DEFAULT_ZIP = RAW_DIR / "test_video.zip"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "AvaMERG"
ZIP_PREFIX = "test_video_v5_0"

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

VID_PATTERN = re.compile(r"dia(\d+)utt(\d+)_(\d+)\.mp4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-frames", type=int, default=5)
    p.add_argument("--max-context-turns", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_zip_index(zip_path: Path) -> Dict[Tuple[str, int], Tuple[int, str]]:
    """Map (conversation_id, utt_index) -> (profileID, zip_member_name)."""
    index: Dict[Tuple[str, int], Tuple[int, str]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = name.rsplit("/", 1)[-1]
            m = VID_PATTERN.fullmatch(base)
            if m:
                cid, utt, pid = m.group(1), int(m.group(2)), int(m.group(3))
                index[(cid, utt)] = (pid, name)
    return index


def pick_target_turn(record: dict, zip_index: Dict, cid: str, speaker_id: int
                     ) -> Optional[Tuple[int, int, dict]]:
    """Return (turn_position, utt_index, turn) for the first turn whose
    dialogue_history ends with a speaker line that has a matching speaker video.
    """
    for pos, t in enumerate(record.get("turns", [])):
        dh = t.get("dialogue_history", [])
        if not dh or dh[-1].get("role") != "speaker":
            continue
        utt = len(dh) - 1
        entry = zip_index.get((cid, utt))
        if entry and entry[0] == speaker_id:
            return pos, utt, t
    return None


def probe_duration(video_path: Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        d = float(proc.stdout.strip())
        return d if d > 0 else None
    except ValueError:
        return None


def extract_one(video: Path, ts: float, out: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if out.exists():
        out.unlink()
    for cmd in (
        [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{ts:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", "scale=448:-2", str(out)],
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-ss", f"{ts:.3f}",
         "-frames:v", "1", "-vf", "scale=448:-2", str(out)],
    ):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            return True
        if out.exists():
            out.unlink()
    return False


def extract_frames(video: Path, frame_dir: Path, n: int) -> List[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = [frame_dir / f"frame_{i:04d}.jpg" for i in range(1, n + 1)]
    dur = probe_duration(video)
    if dur:
        stamps = [(i + 1) * dur / (n + 1) for i in range(n)]
    else:
        stamps = [float(i) for i in range(n)]
    last_ok: Optional[Path] = None
    for ts, out in zip(stamps, frames):
        if extract_one(video, ts, out) or extract_one(video, 0.0, out):
            last_ok = out
        elif last_ok is not None:
            shutil.copyfile(last_ok, out)
        else:
            raise RuntimeError(f"failed to extract any frame from {video}")
    return frames


def build_context(turn: dict, max_turns: int, n_images: int) -> str:
    """dialogue_history -> conversation text, prefixed with N <image> tokens."""
    dh = turn.get("dialogue_history", [])
    dh = dh[-(2 * max_turns):]  # cap length
    lines = []
    for h in dh:
        role = "Client" if h.get("role") == "speaker" else "Therapist"
        utt = clean(h.get("utterance", ""))
        if utt:
            lines.append(f"{role}: {utt}")
    lines.append("Therapist:")
    return "<image>" * n_images + "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.src.is_file():
        raise FileNotFoundError(f"AvaMERG source not found: {args.src}")
    if not args.zip.is_file():
        raise FileNotFoundError(f"AvaMERG video zip not found: {args.zip}")

    data = json.load(args.src.open(encoding="utf-8"))
    print("Building zip index (this reads the zip central directory)...")
    zip_index = build_zip_index(args.zip)
    print(f"Indexed {len(zip_index)} video members.")

    # Collect usable conversations (>=1 turn with a matching speaker video).
    candidates = []  # (record, turn_pos, utt_index, turn)
    for r in data:
        if not r.get("turns"):
            continue
        cid = r["conversation_id"]
        sid = r["speaker_profile"]["ID"]
        picked = pick_target_turn(r, zip_index, cid, sid)
        if picked:
            pos, utt, t = picked
            candidates.append((r, pos, utt, t))

    print(f"Total conversations: {len(data)} | usable candidates: {len(candidates)}")
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.num_samples]
    print(f"Selected: {len(selected)} / requested {args.num_samples}")
    if args.dry_run:
        return

    out_dir = args.output_dir
    frames_root = out_dir / "frames"
    tmp_root = out_dir / "_tmp_video"
    frames_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test.jsonl"

    records = []
    with zipfile.ZipFile(args.zip) as zf:
        for i, (r, pos, utt, t) in enumerate(selected):
            cid = r["conversation_id"]
            member = zip_index[(cid, utt)][1]
            # extract single mp4 to tmp
            raw = tmp_root / f"dia{cid}utt{utt}.mp4"
            with zf.open(member) as src, raw.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            frame_dir = frames_root / f"dia{cid}utt{utt}"
            frames = extract_frames(raw, frame_dir, args.num_frames)
            raw.unlink()  # delete raw video immediately to save space

            gold = clean(t.get("response", ""))
            ctx = build_context(t, args.max_context_turns, len(frames))
            records.append({
                "system": SYSTEM_PROMPT,
                "conversations": [
                    {"from": "user", "value": ctx,
                     "images": [p.resolve().relative_to(PROJECT_ROOT).as_posix() if p.resolve().is_relative_to(PROJECT_ROOT) else str(p.resolve()) for p in frames]},
                    {"from": "assistant", "value": f"<response>{gold}</response>"},
                ],
            })
            if (i + 1) % 25 == 0 or i + 1 == len(selected):
                print(f"Built {i + 1}/{len(selected)}")

    shutil.rmtree(tmp_root, ignore_errors=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"JSONL written to: {out_path}  ({len(records)} lines)")
    print(f"Frames under: {frames_root}")


if __name__ == "__main__":
    main()
