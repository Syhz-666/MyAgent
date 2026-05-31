@echo off
setlocal

cd /d "%~dp0"
set PYTHONUTF8=1

echo Starting MyAgent Web UI...
echo Project directory: %CD%
echo.

where py >nul 2>nul
if errorlevel 1 (
    python web_ui.py
) else (
    py web_ui.py
)

if errorlevel 1 (
    echo.
    echo Web UI exited with an error. Please check the messages above.
)

pause
