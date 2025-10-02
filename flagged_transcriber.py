"""Flagged audio recording and transcription tool.

This script records audio from the default microphone while letting you insert
named markers ("flags") during the session. After recording, the audio between
markers is transcribed with OpenAI's Whisper models and labelled with the marker
names.

Usage example::

    python flagged_transcriber.py --model small --language en --output slides.txt

During recording, type commands such as ``flag Slide 1`` or ``flag Intro`` to
mark sections. Type ``stop`` when you are done. The tool will print the
transcriptions section by section and optionally save them to a text file.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import threading
import wave
from dataclasses import dataclass, field
from typing import List, Optional

import pyaudio
import whisper


CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16_000


@dataclass(order=True)
class Marker:
    """Represents a named marker dropped during the recording."""

    sample_index: int
    timestamp: float
    name: str = field(compare=False)


class FlaggedRecorder:
    """Capture audio from the microphone while tracking named markers."""

    def __init__(self, rate: int = RATE, chunk: int = CHUNK, format_: int = FORMAT, channels: int = CHANNELS):
        self.rate = rate
        self.chunk = chunk
        self.format = format_
        self.channels = channels

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        self.sample_width = self._pa.get_sample_size(self.format)
        self.frames: List[bytes] = []
        self.markers: List[Marker] = []

        self._total_samples = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Begin background recording."""

        if self._thread is not None:
            raise RuntimeError("Recorder already started")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def _record_loop(self) -> None:
        while not self._stop_event.is_set():
            data = self._stream.read(self.chunk, exception_on_overflow=False)
            with self._lock:
                self.frames.append(data)
                self._total_samples += len(data) // self.sample_width

    def add_marker(self, name: str) -> Marker:
        """Insert a marker at the current recording position."""

        with self._lock:
            sample_index = self._total_samples
        timestamp = sample_index / self.rate
        marker = Marker(sample_index=sample_index, timestamp=timestamp, name=name)
        self.markers.append(marker)
        self.markers.sort()  # Ensure markers remain ordered if added quickly.
        return marker

    def stop(self) -> None:
        """Stop the recording and close audio resources."""

        self._stop_event.set()
        if self._thread:
            self._thread.join()
            self._thread = None

        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total_samples

    def save_full_recording(self, path: str) -> None:
        """Persist the entire recording to ``path`` as a WAV file."""

        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(self.frames))

    def iter_marker_segments(self):
        """Yield (marker, start_sample, end_sample) tuples for each segment."""

        if not self.markers:
            return

        total_samples = self.total_samples
        ordered = sorted(self.markers)
        for idx, marker in enumerate(ordered):
            start_sample = marker.sample_index
            end_sample = ordered[idx + 1].sample_index if idx + 1 < len(ordered) else total_samples
            if end_sample <= start_sample:
                continue
            yield marker, start_sample, end_sample


def format_timestamp(seconds: float) -> str:
    mins, secs = divmod(seconds, 60)
    return f"{int(mins):02d}:{secs:05.2f}"


def transcribe_segments(recorder: FlaggedRecorder, model_name: str, language: Optional[str] = None):
    """Transcribe each flagged segment using Whisper."""

    if not recorder.markers:
        raise ValueError("No markers were added; cannot create flagged segments.")

    model = whisper.load_model(model_name)
    joined_audio = b"".join(recorder.frames)
    sample_width = recorder.sample_width

    transcripts = []
    for marker, start_sample, end_sample in recorder.iter_marker_segments():
        segment_bytes = joined_audio[start_sample * sample_width : end_sample * sample_width]
        if not segment_bytes:
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(recorder.channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(recorder.rate)
                wf.writeframes(segment_bytes)

            result = model.transcribe(tmp_path, language=language, fp16=False if model_name != "large" else None)
            transcripts.append(
                {
                    "name": marker.name,
                    "start": marker.timestamp,
                    "end": end_sample / recorder.rate,
                    "text": result.get("text", "").strip(),
                }
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return transcripts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record audio and transcribe flagged sections with Whisper.")
    parser.add_argument("--model", default="small", help="Whisper model size to load (e.g., tiny, base, small, medium, large).")
    parser.add_argument("--language", default=None, help="Optional language code to hint to Whisper.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the labelled transcript as a UTF-8 text file.",
    )
    args = parser.parse_args(argv)

    recorder = FlaggedRecorder()
    print("🎙️ Recording has started. Type commands to manage markers:")
    print("  flag <name>  -> add a marker at the current moment")
    print("  stop         -> finish recording and begin transcription")

    recorder.start()
    try:
        while True:
            try:
                command = input("> ").strip()
            except EOFError:
                command = "stop"
            except KeyboardInterrupt:
                print("\nDetected interrupt; stopping recording...")
                command = "stop"

            if not command:
                continue

            if command.lower() == "stop":
                break

            if command.lower().startswith("flag "):
                name = command[5:].strip()
                if not name:
                    print("⚠️  Please provide a name for the flag, e.g., 'flag Slide 1'.")
                    continue
                marker = recorder.add_marker(name)
                print(
                    f"✅ Added marker '{marker.name}' at {format_timestamp(marker.timestamp)} ({marker.timestamp:.2f}s)."
                )
            else:
                print("Unrecognised command. Use 'flag <name>' or 'stop'.")
    finally:
        recorder.stop()

    print("⏳ Transcribing flagged sections using Whisper...")
    try:
        transcripts = transcribe_segments(recorder, model_name=args.model, language=args.language)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    if not transcripts:
        print("No non-empty segments were found between markers.")
        return 0

    lines = []
    for segment in transcripts:
        start_label = format_timestamp(segment["start"])
        end_label = format_timestamp(segment["end"])
        header = f"{segment['name']} ({start_label} - {end_label})"
        body = segment["text"] or "[No speech detected]"
        formatted = f"{header}\n{body}\n"
        print("\n" + formatted)
        lines.append(formatted)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"💾 Transcript saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
