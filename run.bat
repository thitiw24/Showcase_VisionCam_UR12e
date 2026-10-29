@echo off
title Knom_Propak - Vision & Robot Service
cd /d "%~dp0"

echo ======================================================================
echo   KNOM PROPAK - ROBOT VISION SERVICE
echo ======================================================================
echo Directory: %~dp0
echo.

:: Check if Python is available in system PATH
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in system PATH.
    echo Please install Python 3.10+ or add Python to your system PATH.
    pause
    exit /b 1
)

:loop
echo [RUNNER] Starting main.py at %time% ...
python main.py

echo.
echo [RUNNER] Program stopped at %time%
echo [RUNNER] Watchdog: Restarting automatically in 3 seconds... (Press Ctrl+C to abort)
timeout /t 3 /nobreak >nul
goto loop
