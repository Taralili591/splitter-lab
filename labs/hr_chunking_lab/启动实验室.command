#!/bin/zsh
set -e
cd "$(dirname "$0")/../.."
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m labs.hr_chunking_lab --open "$@"
else
  exec uv run python -m labs.hr_chunking_lab --open "$@"
fi
