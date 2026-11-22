#!/usr/bin/env python3
"""Voice Isolator Lite — a tiny desktop app for Inworld's open-source voice-isolation model.

Separates clean speech from background noise in an audio file. It reuses the Python
interpreter and PyTorch already installed on your machine — no bundled runtimes, no
virtual environments, no duplicate downloads.

Usage:
    GUI:       python voice_isolator_app.py            (or double-click run_windows.bat)
    Headless:  python voice_isolator_app.py input.wav [-o output.wav] [--device cpu]
               [--background]

The model (64M params) is loaded through the `inworld-voice-isolation` package. Weights
(~250 MB) auto-download from Hugging Face on first use and are cached afterwards.

Model: https://github.com/inworld-ai/voice-isolator (Apache-2.0).
This wrapper is a community project and is not affiliated with Inworld AI.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import traceback

MODEL_NAME = "inworld/voice-isolator"
INSTALL_HINT = "python install.py   (or: python -m pip install inworld-voice-isolation)"

AUDIO_FILETYPES = [
    ("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.au *.w64"),
    ("All files", "*.*"),
]


# --------------------------------------------------------------------------- environment


def torch_summary() -> str:
    """One-line report of the interpreter/PyTorch environment, e.g. for the status bar."""
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed"
    cuda = "yes" if torch.cuda.is_available() else "no"
    return f"Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA available: {cuda}"


def resolve_device(choice: str) -> str:
    """Map a user-facing choice (auto/cuda/mps/cpu) to a concrete torch device string.

    Mirrors the model package's own fallback (CUDA, then Apple MPS on torch >= 2.6,
    then CPU) so the reported device is exactly the one used.
    """
    if choice != "auto":
        return choice
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        base = torch.__version__.split("+")[0].split(".")
        if tuple(int(p) for p in base[:2]) >= (2, 6):
            return "mps"
    return "cpu"


# ---------------------------------------------------------------------------------- core


def default_output(in_path: str) -> str:
    root, ext = os.path.splitext(in_path)
    return root + "_clean" + (ext or ".wav")


def default_background(in_path: str) -> str:
    root, ext = os.path.splitext(in_path)
    return root + "_background" + (ext or ".wav")


_ISOLATORS: dict[str, object] = {}  # device -> VoiceIsolator (loaded once, reused)


def _get_isolator(device: str):
    from inworld import voice_isolation

    iso = _ISOLATORS.get(device)
    if iso is None:
        iso = voice_isolation.VoiceIsolator(MODEL_NAME, device=device)
        _ISOLATORS[device] = iso
    return iso


def isolate_file(
    in_path: str,
    out_path: str | None = None,
    device: str = "auto",
    save_background: bool = False,
    log=print,
):
    """Isolate speech from `in_path`.

    Returns (output_path, background_path_or_None, elapsed_seconds, device_used).
    Raises ImportError if the model package is not installed.
    """
    from inworld import voice_isolation

    dev = resolve_device(device)
    out_path = out_path or default_output(in_path)

    log(f"Loading audio: {in_path}")
    wav, sr = voice_isolation.load_audio(in_path)

    log(f"Loading model on device '{dev}' (first run downloads ~250 MB, then cached)...")
    t0 = time.perf_counter()
    iso = _get_isolator(dev)

    log("Isolating voice...")
    voice = iso.isolate(wav, sr, output_sample_rate=sr)
    voice_isolation.save_audio(out_path, voice, sr, subtype="PCM_16")

    bg_path = None
    if save_background:
        bg_path = default_background(in_path)
        voice_isolation.save_audio(bg_path, wav - voice, sr, subtype="PCM_16")
        log(f"Background track saved: {bg_path}")

    elapsed = time.perf_counter() - t0
    log(f"Done in {elapsed:.1f}s -> {out_path}")
    return out_path, bg_path, elapsed, dev


# ------------------------------------------------------------------------------- headless


def run_headless(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="voice_isolator_app",
        description="Isolate clean speech from a noisy audio file (headless mode).",
    )
    parser.add_argument("input", help="input audio file (WAV/FLAC/OGG/...)")
    parser.add_argument("-o", "--output", help="output file (default: <input>_clean.wav)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument(
        "--background", action="store_true", help="also save the separated background track"
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 2

    print(torch_summary())
    try:
        isolate_file(
            args.input, args.output, device=args.device, save_background=args.background
        )
    except ImportError:
        print("The model package is not installed. Run:", file=sys.stderr)
        print(f"    {INSTALL_HINT}", file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------------------------------------ GUI


def run_gui() -> int:
    # Under `pythonw` (no console) stdout/stderr are None; some libraries (tqdm inside
    # huggingface_hub, used for the first-run weight download) fail writing to them.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class App:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.q: queue.Queue = queue.Queue()
            self.busy = False

            root.title("Voice Isolator Lite")
            root.minsize(600, 260)

            frm = ttk.Frame(root, padding=14)
            frm.grid(row=0, column=0, sticky="nsew")
            root.columnconfigure(0, weight=1)
            root.rowconfigure(0, weight=1)
            frm.columnconfigure(1, weight=1)

            # Input file
            ttk.Label(frm, text="Input audio:").grid(row=0, column=0, sticky="w", pady=3)
            self.in_var = tk.StringVar()
            ttk.Entry(frm, textvariable=self.in_var).grid(row=0, column=1, sticky="ew", padx=6)
            ttk.Button(frm, text="Browse...", command=self.pick_input).grid(row=0, column=2)

            # Output file
            ttk.Label(frm, text="Output file:").grid(row=1, column=0, sticky="w", pady=3)
            self.out_var = tk.StringVar()
            ttk.Entry(frm, textvariable=self.out_var).grid(row=1, column=1, sticky="ew", padx=6)
            ttk.Button(frm, text="Browse...", command=self.pick_output).grid(row=1, column=2)

            # Device
            ttk.Label(frm, text="Device:").grid(row=2, column=0, sticky="w", pady=3)
            self.dev_var = tk.StringVar(value="Auto")
            devices = ["Auto", "CUDA", "CPU"]
            import torch  # noqa: deferred; torch is a hard requirement to reach this line

            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                devices.insert(2, "MPS")
            self.dev_box = ttk.Combobox(
                frm, textvariable=self.dev_var, values=devices, state="readonly", width=8
            )
            self.dev_box.grid(row=2, column=1, sticky="w", padx=6)

            # Environment report
            auto_dev = resolve_device("auto")
            env = f"{torch_summary()}  |  Auto device -> {auto_dev}"
            ttk.Label(frm, text=env, foreground="#555").grid(
                row=3, column=0, columnspan=3, sticky="w", pady=(2, 6)
            )

            # Options
            self.bg_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                frm, text="Also save the background/noise track", variable=self.bg_var
            ).grid(row=4, column=0, columnspan=3, sticky="w")

            # Action row
            self.go = ttk.Button(frm, text="Isolate Voice", command=self.start)
            self.go.grid(row=5, column=0, pady=(10, 4))
            self.bar = ttk.Progressbar(frm, mode="indeterminate", length=220)

            # Status line
            self.status = ttk.Label(
                frm,
                text="Pick an input file, then click Isolate Voice.",
                wraplength=560,
                foreground="#333",
            )
            self.status.grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))

            self.root.after(100, self.poll)

        # --- file pickers ---------------------------------------------------

        def pick_input(self) -> None:
            path = filedialog.askopenfilename(title="Choose an audio file", filetypes=AUDIO_FILETYPES)
            if not path:
                return
            self.in_var.set(path)
            if not self.out_var.get():
                self.out_var.set(default_output(path))

        def pick_output(self) -> None:
            path = filedialog.asksaveasfilename(
                title="Save isolated voice as", defaultextension=".wav", filetypes=AUDIO_FILETYPES
            )
            if path:
                self.out_var.set(path)

        # --- run ------------------------------------------------------------

        def start(self) -> None:
            if self.busy:
                return
            in_path = self.in_var.get().strip()
            if not in_path or not os.path.isfile(in_path):
                messagebox.showwarning("No input", "Choose a valid input audio file first.")
                return
            device = self.dev_var.get().lower()
            self.busy = True
            self.go.config(state="disabled")
            self.bar.grid(row=5, column=1, sticky="ew", pady=(10, 4))
            self.bar.start(12)
            self.set_status("Starting...", "#333")
            threading.Thread(
                target=self.worker,
                args=(in_path, self.out_var.get().strip() or None, device, self.bg_var.get()),
                daemon=True,
            ).start()

        def worker(self, in_path, out_path, device, save_background) -> None:
            try:
                result = isolate_file(
                    in_path,
                    out_path,
                    device=device,
                    save_background=save_background,
                    log=lambda m: self.q.put(("status", m)),
                )
                self.q.put(("done", result))
            except ImportError:
                self.q.put(("nopackage", INSTALL_HINT))
            except Exception:  # noqa: BLE001 — show the failure to the user
                tail = traceback.format_exc().strip().splitlines()[-1]
                self.q.put(("error", tail))

        def poll(self) -> None:
            try:
                while True:
                    kind, payload = self.q.get_nowait()
                    if kind == "status":
                        self.set_status(str(payload), "#333")
                    elif kind == "done":
                        out_path, bg_path, elapsed, dev = payload
                        msg = f"Done in {elapsed:.1f}s on '{dev}'. Saved: {out_path}"
                        if bg_path:
                            msg += f"  (+ background: {bg_path})"
                        self.set_status(msg, "#0a7a0a")
                        self.finish()
                    elif kind == "nopackage":
                        self.set_status("The model package is not installed.", "#b00020")
                        messagebox.showerror(
                            "Package missing",
                            "The 'inworld-voice-isolation' package is required.\n\n"
                            f"Fix:\n    {payload}",
                        )
                        self.finish()
                    elif kind == "error":
                        self.set_status(f"Failed: {payload}", "#b00020")
                        messagebox.showerror("Isolation failed", payload)
                        self.finish()
            except queue.Empty:
                pass
            self.root.after(100, self.poll)

        def set_status(self, text: str, color: str) -> None:
            self.status.config(text=text, foreground=color)

        def finish(self) -> None:
            self.busy = False
            self.bar.stop()
            self.bar.grid_remove()
            self.go.config(state="normal")

    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


# ----------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        return run_headless(argv)
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
