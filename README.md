# test-codex

This repository contains a simple GUI application that performs live transcription using OpenAI's Whisper model.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the transcriber and select a Word document to save slides:
   ```bash
   python live_gui_transcriber.py
   ```

The application records audio in short segments, transcribes it using Whisper, and lets you manage slide notes and screenshots. Use the on-screen controls or hotkeys (e.g., `Alt+S` to save the current slide).
