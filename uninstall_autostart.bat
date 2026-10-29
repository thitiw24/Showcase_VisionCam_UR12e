@echo off
title Uninstall Autostart - Knom Propak

set "SHORTCUT=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Knom_Propak.lnk"

if exist "%SHORTCUT%" (
    del /f /q "%SHORTCUT%"
    echo [OK] Autostart shortcut successfully removed from Windows Startup!
) else (
    echo [INFO] Shortcut was not found in Windows Startup folder.
)

pause
