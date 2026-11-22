@echo off
rem Installs the model package into your existing Python (never touches PyTorch).
cd /d "%~dp0"
where python >nul 2>nul && (python install.py) || (py -3 install.py)
echo.
pause
