# Voice Isolator Lite

A tiny desktop app that separates **clean speech from background noise** in an audio
file, powered by [Inworld's open-source voice-isolation model](https://github.com/inworld-ai/voice-isolator)
(64M parameters, 48 kHz).

It is deliberately **extremely lightweight**: a single Python file with a standard-library
Tkinter GUI. It **reuses the Python and PyTorch you already have installed** — no bundled
runtimes, no virtual environments, no duplicate downloads.

## Features

- One-click GUI: pick a noisy file, get a clean voice file
- Optional: also save the separated background/noise track
- Automatic device selection: CUDA GPU → Apple Silicon (MPS) → CPU
- Headless CLI mode for scripting and batch use
- Windows / macOS / Linux — anything that runs Python 3.10+ with Tkinter

## Requirements

- Python **3.10+** with Tkinter (included in the standard installer on Windows/macOS;
  on Debian/Ubuntu: `sudo apt install python3-tk`)
- PyTorch **2.1+** — *already installed? it will not be touched*. Not installed yet?
  the installer adds it for you.
- GPU is optional: a CUDA GPU speeds things up, but the model is small and CPU works fine.

## Quick start

**Windows**
1. Double-click `install_windows.bat` (installs the model package into your Python; never touches your PyTorch).
2. Double-click `run_windows.bat` — the app opens with no console window.

**macOS / Linux**
```bash
python install.py
python voice_isolator_app.py
```

The first isolation downloads the model weights (~250 MB) from Hugging Face and caches
them (`~/.cache/huggingface`); every later run is offline.

## Headless mode

The same file doubles as a CLI:

```bash
python voice_isolator_app.py meeting.wav                    # -> meeting_clean.wav
python voice_isolator_app.py meeting.wav --background      # also meeting_background.wav
python voice_isolator_app.py meeting.wav --device cpu      # force CPU
python voice_isolator_app.py meeting.wav -o out.flac       # output format follows the extension
```

Input formats are whatever [soundfile](https://github.com/bastibe/python-soundfile) reads
(WAV, FLAC, OGG, ...); stereo is downmixed to mono automatically.

## How it works

The app is a thin wrapper around the [`inworld-voice-isolation`](https://pypi.org/project/inworld-voice-isolation/)
package: `load_audio()` → `VoiceIsolator("inworld/voice-isolator").isolate()` →
`save_audio()`. The installer pins `torchaudio` to your installed PyTorch version so
`pip` can never upgrade or replace your existing PyTorch build.

## Troubleshooting

- **"CUDA available: no" in the app's status bar** — your PyTorch is the CPU-only build.
  Everything still works on CPU (the model is small); to use your GPU, reinstall PyTorch
  following [pytorch.org's instructions](https://pytorch.org/get-started/locally/).
- **GUI doesn't open** — Tkinter is missing (rare). Debian/Ubuntu:
  `sudo apt install python3-tk`.
- **Diagnosing the model package** — run `voice-isolate-doctor` (ships with the package).
- **First run is slow** — that's the one-time weight download (~250 MB).

## Credits & license

- Model: [inworld-ai/voice-isolator](https://github.com/inworld-ai/voice-isolator) by
  Inworld AI, Apache-2.0. Weights auto-download from Hugging Face
  ([inworld/voice-isolator](https://huggingface.co/inworld/voice-isolator)).
- This wrapper is a community project and is **not affiliated with Inworld AI**.
- Code in this repository: Apache-2.0 (see [LICENSE](LICENSE)).
