@echo off
setlocal enabledelayedexpansion

REM Always run from project root (where this .bat lives)
cd /d "%~dp0"

REM Activate venv if exists
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo [WARN] .venv not found. Using system Python/Streamlit.
)

REM Prefer venv streamlit if available
if exist ".venv\Scripts\streamlit.exe" (
  ".venv\Scripts\streamlit.exe" run "app\ui.py" --server.port 8501
) else (
  streamlit run "app\ui.py" --server.port 8501
)

if errorlevel 1 (
  echo.
  echo [ERROR] Streamlit exited with code %errorlevel%.
  pause
)
endlocal