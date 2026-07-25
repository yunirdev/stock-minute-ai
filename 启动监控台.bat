@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0"

echo.
echo stock-minute-ai monitor launcher
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Project environment is missing. Running setup.bat...
    call setup.bat
    if errorlevel 1 exit /b 1
)

.venv\Scripts\python.exe --version >nul 2>&1
if errorlevel 1 (
    echo Project environment is invalid. Running setup.bat...
    call setup.bat
    if errorlevel 1 exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Project environment is unavailable after setup.
    exit /b 1
)

echo Starting NiceGUI monitor...
.venv\Scripts\python.exe -m trader.monitor_nice
set "MONITOR_EXIT=%ERRORLEVEL%"
if not "%MONITOR_EXIT%"=="0" (
    echo ERROR: NiceGUI monitor exited with code %MONITOR_EXIT%.
    echo To use browser mode, run: set QUANT_WEB=1
)
exit /b %MONITOR_EXIT%
