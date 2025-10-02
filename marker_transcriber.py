#!/usr/bin/env python3
"""CLI utility to transcribe audio segments defined by markers."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from faster_whisper import WhisperModel


@dataclass
class Marker:
    name: str
    start: float
    end: float


def parse_time(value: Optional[str | float | int]) -> Optional[float]:
    """Convert a time representation into seconds."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        pass

    # Parse HH:MM:SS.mmm or MM:SS
    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid time format: {value}")

    seconds = 0.0
    for idx, part in enumerate(reversed(parts)):
        if idx == 0:
            seconds += float(part)
        elif idx == 1:
            seconds += 60 * float(part)
        elif idx == 2:
            seconds += 3600 * float(part)
    return seconds


def read_markers(marker_path: Path) -> List[dict]:
    """Read markers from CSV or JSON file using pandas or json module."""
    if marker_path.suffix.lower() == ".csv":
        df = pd.read_csv(marker_path)
        return df.to_dict(orient="records")

    if marker_path.suffix.lower() == ".json":
        with open(marker_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON marker file must contain a list of marker objects")
        return data

    raise ValueError("Marker file must be a .csv or .json file")


def normalize_markers(raw_markers: Iterable[dict], audio_duration: float) -> List[Marker]:
    markers: List[Marker] = []
    for entry in raw_markers:
        name = str(entry.get("name") or entry.get("title") or "").strip()
        if not name:
            raise ValueError("Marker entry missing 'name' field")
        start = parse_time(entry.get("start") or entry.get("start_time"))
        end = parse_time(entry.get("end") or entry.get("end_time"))
        if start is None:
            raise ValueError(f"Marker '{name}' missing start time")
        markers.append(Marker(name=name, start=start, end=end if end is not None else -1.0))

    markers.sort(key=lambda m: m.start)

    for idx, marker in enumerate(markers):
        if marker.end is not None and marker.end >= 0:
            continue
        next_start = markers[idx + 1].start if idx + 1 < len(markers) else audio_duration
        marker.end = next_start

    for marker in markers:
        if marker.end is None:
            marker.end = audio_duration
        marker.end = min(marker.end, audio_duration)
        if marker.end <= marker.start:
            raise ValueError(f"Marker '{marker.name}' has non-positive duration")

    return markers


def ffprobe_duration(audio_path: Path) -> float:
    """Return duration of audio using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Unable to determine duration for {audio_path}") from exc


def extract_segment(audio_path: Path, start: float, end: float) -> Path:
    duration = end - start
    if duration <= 0:
        raise ValueError("Segment duration must be positive")

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_file.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-ar",
        "16000",
        "-ac",
        "1",
        tmp_file.name,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Path(tmp_file.name)


def transcribe_segment(model: WhisperModel, segment_path: Path) -> str:
    segments, _ = model.transcribe(str(segment_path))
    texts: List[str] = []
    for segment in segments:
        piece = segment.text.strip()
        if piece:
            texts.append(piece)
    return " ".join(texts).strip()


def process_markers(
    audio_path: Path,
    markers: List[Marker],
    model: WhisperModel,
) -> List[dict]:
    results: List[dict] = []
    tmp_files: List[Path] = []
    try:
        for marker in markers:
            segment_path = extract_segment(audio_path, marker.start, marker.end)
            tmp_files.append(segment_path)
            text = transcribe_segment(model, segment_path)
            results.append(
                {
                    "name": marker.name,
                    "start_sec": marker.start,
                    "end_sec": marker.end,
                    "text": text,
                }
            )
    finally:
        for path in tmp_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    return results


def save_json(results: List[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


def save_markdown(results: List[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for item in results:
            handle.write(f"### {item['name']}\n\n")
            handle.write(f"{item['text']}\n\n")


def print_results(results: List[dict]) -> None:
    for item in results:
        print(f"{item['name']} ({item['start_sec']:.2f}s - {item['end_sec']:.2f}s):")
        print(item["text"])
        print()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio markers using faster-whisper")
    parser.add_argument("--audio", required=True, type=Path, help="Path to the audio file")
    parser.add_argument("--markers", required=True, type=Path, help="Path to markers CSV/JSON")
    parser.add_argument("--model", required=True, help="Name or path of the faster-whisper model")
    parser.add_argument("--out-json", required=True, type=Path, help="Output JSON file path")
    parser.add_argument("--out-md", required=True, type=Path, help="Output Markdown file path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    audio_path: Path = args.audio
    markers_path: Path = args.markers

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 1
    if not markers_path.exists():
        print(f"Marker file not found: {markers_path}", file=sys.stderr)
        return 1

    audio_duration = ffprobe_duration(audio_path)
    raw_markers = read_markers(markers_path)
    markers = normalize_markers(raw_markers, audio_duration)

    model = WhisperModel(args.model)

    results = process_markers(audio_path, markers, model)

    save_json(results, args.out_json)
    save_markdown(results, args.out_md)
    print_results(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
