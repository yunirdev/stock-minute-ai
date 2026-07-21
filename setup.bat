@echo off
chcp 65001 >nul 2>&1
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║         stock-minute-ai  一键安装 / 修复环境         ║
echo ╚══════════════════════════════════════════════════════╝
echo.

:: ── 1. 检查 Python ───────────────────────────────────────────────────────────
echo [1/5] 检查 Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo      ✗ 未找到 Python。
    echo      请先安装 Python 3.13+: https://www.python.org/downloads/
    echo      安装时勾选 "Add Python to PATH"
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo      ✓ Python %PY_VER%

:: ── 2. 安装 / 检查 uv ────────────────────────────────────────────────────────
echo.
echo [2/5] 检查 uv 包管理器...
uv --version >nul 2>&1
if errorlevel 1 (
    echo      未找到 uv，正在安装...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    :: 刷新 PATH
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    set "PATH=%LOCALAPPDATA%\Programs\uv;%PATH%"
    uv --version >nul 2>&1
    if errorlevel 1 (
        echo      ✗ uv 安装失败，请手动安装: https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
)
for /f "tokens=2" %%v in ('uv --version 2^>^&1') do set UV_VER=%%v
echo      ✓ uv %UV_VER%

:: ── 3. 创建虚拟环境并安装依赖 ────────────────────────────────────────────────
echo.
echo [3/5] 安装 Python 依赖（根据 uv.lock，首次约 2-5 分钟）...
uv sync
if errorlevel 1 (
    echo      ✗ 依赖安装失败，请检查网络或查看上方错误。
    pause
    exit /b 1
)
echo      ✓ 依赖安装完成

:: ── 4. 创建 .env（如果不存在） ───────────────────────────────────────────────
echo.
echo [4/5] 检查配置文件...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo      ✓ 已从 .env.example 创建 .env
        echo      ★ 请用记事本打开 .env，填入你的 API Key！
    ) else (
        echo      ⚠ 未找到 .env.example，请手动创建 .env
    )
) else (
    echo      ✓ .env 已存在
)

:: ── 5. 检查 Ollama ────────────────────────────────────────────────────────────
echo.
echo [5/5] 检查 Ollama（本地 AI）...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo      ⚠ 未找到 Ollama。
    echo        如需本地 AI 评分，请安装: https://ollama.com/download
    echo        安装后用 ollama pull 拉取任意模型即可，系统会自动选择。
    echo        （无 Ollama 也能运行，AI 评分会显示 50 / StubLLM）
) else (
    echo      ✓ Ollama 已安装
    echo      已安装的模型：
    for /f "skip=1 tokens=1" %%m in ('ollama list 2^>nul') do (
        echo        · %%m
    )
    echo      系统启动时会自动选择最合适的模型（优先 .env 中的 OLLAMA_MODEL）。
)

:: ── 完成 ──────────────────────────────────────────────────────────────────────
echo.
echo ════════════════════════════════════════════════════════
echo   安装完成！启动命令：
echo.
echo     监控 UI：  uv run python -m nicegui trader/monitor_nice.py
echo     运行引擎： uv run python -m trader.main
echo     跑测试：   uv run python -m pytest tests/ -v
echo ════════════════════════════════════════════════════════
echo.
echo ★ 首次运行前请确认 .env 已填入 ALPACA_API_KEY / ALPACA_API_SECRET
echo.
pause
