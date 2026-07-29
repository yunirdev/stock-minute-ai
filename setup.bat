@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo.
echo stock-minute-ai environment setup
echo.

echo [1/5] Checking Python 3.13 or newer...
python --version
if errorlevel 1 (
    echo ERROR: Python was not found. Install Python 3.13 or newer and add it to PATH.
    exit /b 1
)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.13 or newer is required.
    exit /b 1
)

echo.
echo [2/5] Checking uv...
uv --version
if errorlevel 1 (
    echo uv was not found. Installing it now...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\Programs\uv;%PATH%"
    uv --version
    if errorlevel 1 (
        echo ERROR: uv installation failed. See https://docs.astral.sh/uv/
        exit /b 1
    )
)

echo.
echo [3/5] Syncing locked dependencies and checking core imports...
uv sync
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    exit /b 1
)
.venv\Scripts\python.exe -c "import trader.main, trader.runtime, trader.paper_decision"
if errorlevel 1 (
    echo ERROR: Core module import failed.
    exit /b 1
)

echo.
echo [4/5] Checking local configuration...
if not exist ".env" (
    if not exist ".env.example" (
        echo ERROR: .env.example is missing.
        exit /b 1
    )
    copy /y ".env.example" ".env" >nul
    echo Created .env from .env.example. Add your API keys before starting the engine.
) else (
    echo Existing .env preserved.
)

echo.
echo [5/5] Checking optional Ollama installation...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo Ollama is optional and was not found.
) else (
    ollama --version
)

echo.
echo Setup completed successfully.
echo Monitor UI: uv run python -m trader.monitor_nice
echo Trading engine: uv run python -m trader.main
echo Tests: uv run python -m pytest tests -q
exit /b 0
