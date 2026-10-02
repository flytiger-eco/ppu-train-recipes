#!/usr/bin/env python3
"""
Video-MME Data Preparation Script

Generate JSONL required for Megatron-Bridge training from the original Video-MME dataset.
Processing pipeline:
  1. Read parquet metadata (questions/options/answers)
  2. For each video, trim first TRIM_SECONDS seconds -> mp4
  3. Extract corresponding audio (16kHz mono WAV) -> wav
  4. Parse SRT subtitles, extract text within the time window
  5. Generate JSONL in OpenAI conversation format

Usage:
  python prepare_videomme_data.py \
    --video-dir /path/to/Video-MME-uncom/data \
    --parquet   /path/to/Video-MME/videomme/test-00000-of-00001.parquet \
    --subtitle-dir /path/to/subtitle \
    --output-dir   /path/to/output \
    --trim-seconds 10

Dependencies: ffmpeg (video/audio trimming), pandas (parquet reading)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas pyarrow", file=sys.stderr)
    sys.exit(1)


def parse_srt(srt_path: Path, max_seconds: float) -> str:
    """Parse SRT subtitle file, extract text within max_seconds."""
    if not srt_path.exists():
        return ""
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    lines = []
    # Match timestamps: 00:00:00,220 --> 00:00:03,449
    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )
    blocks = re.split(r"\n\n+", text.strip())
    for block in blocks:
        block_lines = block.strip().split("\n")
        if len(block_lines) < 2:
            continue
        m = pattern.search(block_lines[0])
        if not m:
            continue
        start_s = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000
        if start_s > max_seconds:
            break
        # Get text lines after timestamp, remove HTML tags
        for line in block_lines[1:]:
            if pattern.search(line):
                continue
            clean = re.sub(r"<[^>]+>", "", line).strip()
            if clean:
                lines.append(f"[{m.group(0).replace(',', '.')}] {clean}")
    return "\n".join(lines)


def trim_video(src: Path, dst: Path, seconds: float) -> bool:
    """Trim video to first seconds seconds."""
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-t", str(seconds),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def extract_audio(src: Path, dst: Path, seconds: float) -> bool:
    """Extract audio as 16kHz mono WAV, trim to first seconds seconds."""
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-t", str(seconds),
        "-ar", "16000", "-ac", "1",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Video-MME Data Preparation")
    parser.add_argument("--video-dir", required=True,
                        help="Raw video directory (mp4 files)")
    parser.add_argument("--parquet", required=True,
                        help="Video-MME parquet file (containing questions/options/answers)")
    parser.add_argument("--subtitle-dir", default="",
                        help="SRT subtitle directory (optional)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (generates videos/, audios/, test.jsonl)")
    parser.add_argument("--trim-seconds", type=float, default=10,
                        help="Trim video/audio to first N seconds (default 10)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Process at most N videos (0=all)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    parquet_path = Path(args.parquet)
    output_dir = Path(args.output_dir)
    subtitle_dir = Path(args.subtitle_dir) if args.subtitle_dir else None

    out_videos = output_dir / "videos"
    out_audios = output_dir / "audios"
    out_videos.mkdir(parents=True, exist_ok=True)
    out_audios.mkdir(parents=True, exist_ok=True)

    # Read parquet
    df = pd.read_parquet(parquet_path)
    print(f"Read parquet: {len(df)} records")

    # Deduplicate by videoID (one video maps to multiple questions)
    video_ids = df["videoID"].unique().tolist()
    if args.max_samples > 0:
        video_ids = video_ids[:args.max_samples]
    print(f"After dedup: {len(video_ids)} videos")

    # Processing video/audio
    ok_count = 0
    for i, vid in enumerate(video_ids):
        # Find source video
        src_video = None
        for ext in (".mp4", ".webm", ".mkv"):
            candidate = video_dir / f"{vid}{ext}"
            if candidate.exists():
                src_video = candidate
                break
        if src_video is None:
            print(f"  [{i}] Skip {vid}: source video not found")
            continue

        dst_video = out_videos / f"{vid}.mp4"
        dst_audio = out_audios / f"{vid}.wav"

        print(f"  [{i}/{len(video_ids)}] {vid} ...", end=" ")
        v_ok = trim_video(src_video, dst_video, args.trim_seconds)
        a_ok = extract_audio(src_video, dst_audio, args.trim_seconds)
        if v_ok and a_ok:
            print("OK")
            ok_count += 1
        else:
            print(f"FAIL (video={v_ok}, audio={a_ok})")

    print(f"\nVideo/audio processing complete: {ok_count}/{len(video_ids)} successful")

    # Generate JSONL
    jsonl_path = output_dir / "test.jsonl"
    records = []
    for _, row in df.iterrows():
        vid = row["videoID"]
        video_path = str(out_videos / f"{vid}.mp4")
        audio_path = str(out_audios / f"{vid}.wav")

        # Build subtitle text
        subtitle_text = ""
        if subtitle_dir:
            srt_path = subtitle_dir / f"{vid}.srt"
            subtitle_text = parse_srt(srt_path, args.trim_seconds)

        # Build question and options
        question = row["question"]
        options = row["options"]
        option_labels = ["A", "B", "C", "D", "E", "F"]
        options_text = "\n".join(
            f"{option_labels[j]}. {opt}" for j, opt in enumerate(options)
        )
        full_question = f"{question}\n{options_text}"

        # Build conversation
        user_content = []
        user_content.append({"type": "video", "video": video_path})
        user_content.append({"type": "audio", "audio": audio_path})
        if subtitle_text:
            user_content.append({"type": "text", "text": f"[Subtitle]\n{subtitle_text}"})
        user_content.append({"type": "text", "text": full_question})

        record = {
            "conversation": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
            ],
            "metadata": {
                "data_source": "Video-MME",
                "video_id": vid,
                "question_id": row.get("question_id", ""),
                "domain": row.get("domain", ""),
                "task_type": row.get("task_type", ""),
            },
        }
        records.append(record)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"JSONL generation complete: {jsonl_path} ({len(records)} entries)")
    print(f"\nOutput directory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── videos/     ({len(list(out_videos.glob('*.mp4')))} mp4 files)")
    print(f"  ├── audios/     ({len(list(out_audios.glob('*.wav')))} wav files)")
    print(f"  └── test.jsonl  ({len(records)} entries)")


if __name__ == "__main__":
    main()
