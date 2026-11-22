@echo off
rem Launches Voice Isolator Lite without a console window, using your installed Python.
cd /d "%~dp0"
where pythonw >nul 2>nul && (start "" pythonw voice_isolator_app.py) || (start "" python voice_isolator_app.py)
