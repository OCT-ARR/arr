#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_ffmpeg_binary() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def shift_time(time_str: object, offset_seconds: float = 0.0) -> str:
    text = str(time_str).replace(",", ".").strip('"')
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            time_value = datetime.strptime(text, fmt)
            break
        except ValueError:
            time_value = None
    if time_value is None:
        return "00:00:00.000"

    shifted = time_value + timedelta(seconds=offset_seconds)
    if shifted.year < 1900:
        shifted = datetime.strptime("00:00:00.000", "%H:%M:%S.%f")
    return shifted.strftime("%H:%M:%S.%f")[:-3]


def process_single_task(task_data: tuple[dict[str, str], str, str, bool]) -> tuple[str, str]:
    task, gpu_id, ffmpeg_binary, use_nvenc = task_data
    video_path = task["video_path"]
    start_time = task["start_time"]
    end_time = task["end_time"]
    out_path = task["out_path"]
    video_base_name = task["video_base_name"]
    out_name = Path(out_path).name
    clip_id = Path(out_path).stem

    output_file = Path(out_path)
    if output_file.exists() and output_file.stat().st_size > 0:
        return clip_id, f"skip existing [slot:{gpu_id}]: {video_base_name}/{out_name}"
    if output_file.exists() and output_file.stat().st_size == 0:
        output_file.unlink()

    cmd = [ffmpeg_binary]
    if use_nvenc:
        cmd.extend(
            [
                "-y",
                "-vsync",
                "0",
                "-hwaccel",
                "cuda",
                "-hwaccel_device",
                str(gpu_id),
                "-i",
                video_path,
                "-ss",
                start_time,
                "-to",
                end_time,
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-c:a",
                "aac",
            ]
        )
    else:
        cmd.extend(
            [
                "-ss",
                start_time,
                "-to",
                end_time,
                "-i",
                video_path,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-avoid_negative_ts",
                "1",
                "-y",
            ]
        )
    cmd.extend(["-loglevel", "error", out_path])

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return clip_id, f"done [slot:{gpu_id}]: {video_base_name}/{out_name}"
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else "unknown"
        return clip_id, f"failed [slot:{gpu_id}]: {out_name}. reason: {stderr}"
    except Exception as exc:  # pragma: no cover - defensive subprocess boundary
        return clip_id, f"failed [slot:{gpu_id}]: {out_name}. reason: {exc}"


def build_clip_tasks(
    csv_file: Path,
    video_dir: Path,
    output_dir: Path,
    offset_seconds: float,
) -> list[dict[str, str]]:
    if not csv_file.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {csv_file}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    df = pd.read_csv(csv_file)
    required_columns = {
        "Season",
        "Episode",
        "Dialogue_ID",
        "Utterance_ID",
        "Speaker",
        "StartTime",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    tasks: list[dict[str, str]] = []
    grouped = df.groupby(["Season", "Episode", "Dialogue_ID"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for (season, episode, dia_id), group_df in grouped:
        group_df = group_df.sort_values("Utterance_ID")
        rows = group_df.to_dict("records")

        video_base_name = f"S{int(season):02d}E{int(episode):02d}"
        video_path = video_dir / f"{video_base_name}.mp4"
        if not video_path.exists():
            continue

        real_start_time = None
        start_index = -1
        for idx, row in enumerate(rows):
            if row["Speaker"] != "Therapist":
                real_start_time = shift_time(row["StartTime"], offset_seconds)
                start_index = idx
                break
        if real_start_time is None:
            continue

        episode_dir = output_dir / video_base_name
        episode_dir.mkdir(parents=True, exist_ok=True)

        i = 0
        while i < len(rows):
            if i < start_index:
                i += 1
                continue

            row = rows[i]
            if row["Speaker"] == "Therapist":
                utterance_id = int(row["Utterance_ID"])
                end_time = shift_time(row["StartTime"], offset_seconds=0)
                if end_time > real_start_time:
                    out_name = (
                        f"{video_base_name}_dia{int(dia_id)}_u{utterance_id}.mp4"
                    )
                    tasks.append(
                        {
                            "video_path": str(video_path),
                            "start_time": real_start_time,
                            "end_time": end_time,
                            "out_path": str(episode_dir / out_name),
                            "video_base_name": video_base_name,
                        }
                    )

                j = i + 1
                while j < len(rows) and rows[j]["Speaker"] == "Therapist":
                    j += 1
                i = j
            else:
                i += 1

    return tasks


def parse_gpu_ids(raw: str) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return values or ["0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cut raw MESC-FT episode videos into dialogue-history clips. "
            "Raw videos and annotations are not included in the anonymous release."
        )
    )
    parser.add_argument(
        "--csv_file",
        type=Path,
        default=PROJECT_ROOT / "raw" / "MESC-FT" / "MESC.csv",
        help="Path to the source annotation CSV.",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=PROJECT_ROOT / "raw" / "MESC-FT" / "videos",
        help="Directory containing source episode videos named like S01E01.mp4.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "clips",
        help="Directory where cut clips will be written.",
    )
    parser.add_argument(
        "--gpu_ids",
        default="0",
        help="Comma-separated slot ids. Used only as worker labels unless --use_nvenc is set.",
    )
    parser.add_argument(
        "--workers_per_slot",
        type=int,
        default=4,
        help="Number of parallel ffmpeg workers per slot.",
    )
    parser.add_argument(
        "--offset_seconds",
        type=float,
        default=0.0,
        help="Optional offset applied to the first client utterance start time.",
    )
    parser.add_argument(
        "--use_nvenc",
        action="store_true",
        help="Use NVIDIA NVENC instead of stream copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ffmpeg_binary = get_ffmpeg_binary()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    print(f"ffmpeg: {ffmpeg_binary}")
    print(f"annotation csv: {args.csv_file}")
    print(f"video dir: {args.video_dir}")
    print(f"output dir: {args.output_dir}")

    tasks = build_clip_tasks(
        csv_file=args.csv_file,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        offset_seconds=args.offset_seconds,
    )
    if not tasks:
        print("No clip tasks were generated. Check the CSV/video paths and filenames.")
        return

    tasks_with_slots = [
        (task, gpu_ids[i % len(gpu_ids)], ffmpeg_binary, args.use_nvenc)
        for i, task in enumerate(tasks)
    ]
    workers = max(1, min(len(tasks_with_slots), len(gpu_ids) * args.workers_per_slot))
    print(f"Generated {len(tasks_with_slots)} clip tasks; using {workers} workers.")

    finished = 0
    with multiprocessing.Pool(processes=workers) as pool:
        for clip_id, message in pool.imap_unordered(process_single_task, tasks_with_slots):
            finished += 1
            print(f"[{finished}/{len(tasks_with_slots)}] {clip_id}: {message}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
