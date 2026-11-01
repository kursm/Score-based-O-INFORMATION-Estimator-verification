@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Python environment .venv was not found. See README.md.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m soi_nuisance.gui
