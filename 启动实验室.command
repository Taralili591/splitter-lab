#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m labs.hr_chunking_lab --open "$@"
elif command -v uv >/dev/null 2>&1; then
  exec uv run python -m labs.hr_chunking_lab --open "$@"
else
  print '请先按 README 安装 Python 3.11+ 并创建虚拟环境：'
  print '  python3 -m venv .venv'
  print '  .venv/bin/python -m pip install -r requirements.txt'
  print '也可以安装 uv 后，再双击这个文件。'
  read 'reply?按回车关闭窗口。'
  exit 1
fi
