@echo off
title Install Autostart - Knom Propak
cd /d "%~dp0"

echo ======================================================================
echo   INSTALL AUTOSTART SHORTCUT TO WINDOWS STARTUP
echo ======================================================================
echo.

set "TARGET=%~dp0run.bat"
set "SHORTCUT=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Knom_Propak.lnk"

powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%TARGET%'; $s.WorkingDirectory = '%~dp0'; $s.Save()"

if exist "%SHORTCUT%" (
    echo [OK] Autostart shortcut successfully installed!
    echo Shortcut Path: %SHORTCUT%
    echo The vision program will now start automatically when logging into Windows.
) else (
    echo [ERROR] Failed to create Autostart shortcut.
)

echo.
pause
