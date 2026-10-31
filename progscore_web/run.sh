#!/bin/bash
# progscore Streamlit 启动脚本
cd "$(dirname "$0")"
if ! python3 -c "import streamlit" 2>/dev/null; then
  echo "首次运行：安装依赖..."
  pip install -r requirements.txt
fi
exec streamlit run app.py --server.port 8501 --server.address 0.0.0.0
