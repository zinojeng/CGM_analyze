#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
VENV_DIR="$ROOT_DIR/.venv"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "[錯誤] 找不到 python 或 python3 指令，請先安裝 Python。" >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] 建立虛擬環境於 $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PIP="$VENV_DIR/bin/pip"
VENV_PYTHON="$VENV_DIR/bin/python"

"$PIP" install --upgrade pip

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "[INFO] 安裝 requirements.txt 內的套件…"
    "$PIP" install -r "$REQUIREMENTS_FILE"
else
    echo "[WARN] 專案中沒有 requirements.txt，略過依賴安裝。"
fi

if ! "$VENV_DIR/bin/python" -m streamlit --version >/dev/null 2>&1; then
    echo "[INFO] 虛擬環境中未找到 Streamlit，安裝中…"
    "$PIP" install streamlit
fi

trap deactivate EXIT

echo "[INFO] 啟動 Streamlit 應用程式…"
cd "$ROOT_DIR"
exec "$VENV_DIR/bin/streamlit" run main.py
