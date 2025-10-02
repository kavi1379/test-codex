# test-codex

This repository contains tools that help with live or recorded transcription
using OpenAI's Whisper model.

## GUI slide transcriber

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the transcriber and select a Word document to save slides:
   ```bash
   python live_gui_transcriber.py
   ```

The application records audio in short segments, transcribes it using Whisper,
and lets you manage slide notes and screenshots. Use the on-screen controls or
hotkeys (e.g., `Alt+S` to save the current slide).

## Flagged CLI recorder

`flagged_transcriber.py` lets you record audio from the microphone, drop named
markers ("flags") during the recording, and then transcribe each flagged
segment with Whisper.

Example usage:

```bash
python flagged_transcriber.py --model small --language en --output slides.txt
```

While recording:

- Type `flag <name>` (e.g., `flag Slide 1`) to mark the current moment.
- Type `stop` to finish recording and begin transcription.

The script prints the transcription for each marker-labelled segment and can
optionally save the formatted transcript to a text file via `--output`.
