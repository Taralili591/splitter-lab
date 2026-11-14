@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m labs.hr_chunking_lab --open %*
  exit /b
)
where uv >nul 2>nul
if not errorlevel 1 (
  uv run python -m labs.hr_chunking_lab --open %*
  exit /b
)
echo 请先按 README 安装 Python 3.11+ 并创建虚拟环境：
echo   py -3 -m venv .venv
echo   .venv\Scripts\python -m pip install -r requirements.txt
echo 也可以安装 uv 后，再双击这个文件。
pause
exit /b 1
