#!/usr/bin/env python3
"""Smart installer for Voice Isolator Lite.

Installs the `inworld-voice-isolation` package into your *existing* Python
environment. If PyTorch is already installed, it is left completely untouched —
the matching torchaudio version is pinned so pip never upgrades or replaces
your PyTorch build (CPU and CUDA variants are different wheels).

Usage:  python install.py
"""

import subprocess
import sys

MIN_PYTHON = (3, 10)
PACKAGE = "inworld-voice-isolation"


def info(msg: str) -> None:
    print(f"[install] {msg}")


def fail(msg: str) -> None:
    print(f"[install] ERROR: {msg}")
    sys.exit(1)


def main() -> None:
    if sys.version_info < MIN_PYTHON:
        fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
            f"(you have {sys.version.split()[0]})."
        )

    try:
        import tkinter  # noqa: F401

        info(f"Python {sys.version.split()[0]}: OK, tkinter present.")
    except ImportError:
        info("WARNING: tkinter is missing — the GUI will not start.")
        info("         (Debian/Ubuntu fix: sudo apt install python3-tk)")

    try:
        import torch
    except ImportError:
        info("PyTorch not found — it will be installed as a dependency of the model package.")
        info("(PyPI default wheels are CPU-only on Windows/macOS; see pytorch.org for GPU builds.)")
        cmd = [sys.executable, "-m", "pip", "install", PACKAGE]
    else:
        torch_base = torch.__version__.split("+")[0]
        info(f"Found existing PyTorch {torch.__version__} — it will NOT be touched.")
        cmd = [sys.executable, "-m", "pip", "install", PACKAGE, f"torchaudio=={torch_base}"]

    info("Running: " + " ".join(cmd))
    if subprocess.call(cmd) != 0:
        fail("pip install failed (see output above).")

    try:
        from inworld import voice_isolation  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        fail(f"The package was installed but does not import: {exc}")
    info("OK: inworld-voice-isolation imports cleanly.")

    try:
        import torch

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        info(f"Device that will be used: {device}")
    except ImportError:
        pass

    info("All set! Launch the app with:  python voice_isolator_app.py")


if __name__ == "__main__":
    main()
