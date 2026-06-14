@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

cd /d "%~dp0"
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo  🖥️  AI 交易监控平台
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo  正在启动...
echo.

REM 检查虚拟环境
if not exist ".venv\Scripts\python.exe" (
  echo ❌ 虚拟环境不存在。请先运行：
  echo.
  echo    python -m venv .venv
  echo    .venv\Scripts\pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

REM 启动 NiceGUI
.venv\Scripts\python.exe trader\monitor_nice.py
if errorlevel 1 (
  echo.
  echo ❌ 启动失败
  echo.
  echo 如果看到 WebView 相关错误，可改用浏览器模式：
  echo   set QUANT_WEB=1
  echo   .venv\Scripts\python.exe trader\monitor_nice.py
  echo.
  pause
)
